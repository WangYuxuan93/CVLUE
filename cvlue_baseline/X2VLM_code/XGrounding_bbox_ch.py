import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from utils.checkpointer import Checkpointer

from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from dataset.utils import collect_tensor_result, grounding_eval_bbox, grounding_eval_bbox_vlue

from models.model_grounding import XVLMPlusForGrounding

from optim import create_optimizer
from refTools.refer_python3 import REFER
from scheduler import create_scheduler
from utils.hdfs_io import hmkdir, hcopy, hexists


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    accumulate_steps = int(config.get('accumulate_steps', 2))
    for i, (image, text, target_bbox) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        target_bbox = target_bbox.to(device)

        _, loss_bbox, loss_giou = model(image, text_input.input_ids, text_input.attention_mask, target_bbox=target_bbox)
        loss = loss_bbox + loss_giou

        if accumulate_steps > 1:
            loss = loss / accumulate_steps
        
        # backward
        loss.backward()

        if (i+1) % accumulate_steps == 0:
            # update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def val(model, data_loader, tokenizer, device):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50

    result = []

    for image, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_coord = model(image, text_input.input_ids, text_input.attention_mask, target_bbox=None)

        assert len(ref_ids) == outputs_coord.shape[0]

        for r_id, coord in zip(ref_ids, outputs_coord):
            result.append({'ref_id': r_id.item(), 'pred': coord})

    return result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    grd_train_dataset, grd_test_dataset, grd_val_dataset = create_dataset('grounding_bbox_ch', config, args.evaluate)


    print("Creating model")
    model = XVLMPlusForGrounding(config=config)
    model.load_pretrained(args.checkpoint, config, load_bbox_pretrain=args.load_bbox_pretrain, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([grd_test_dataset], [False], num_tasks, global_rank)
        else:
            samplers = [None]

        test_loader = create_loader([grd_test_dataset], samplers,
                                    batch_size=[config['batch_size']],
                                    num_workers=[4], is_trains=[False], collate_fns=[None])


        result = val(model_without_ddp, test_loader[0], tokenizer, device)
        results = collect_tensor_result(result, filename='grounding_bbox_eval', local_wdir=args.result_dir,
                                        hdfs_wdir=args.output_hdfs,
                                        write_to_hdfs=world_size > 8)

        if utils.is_main_process():
            grounding_acc, avg_IoU = grounding_eval_bbox_vlue(results, config['test_file'][0])

            log_stats = {**{f'{k}': v for k, v in grounding_acc.items()}}
            print(log_stats)
            print('Avg_IoU: ', avg_IoU)

        dist.barrier()

    else:
        print("Start training")

        datasets = [grd_train_dataset, grd_val_dataset]

        train_dataset_size = len(grd_train_dataset)
        train_batch_size = config['batch_size']

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
        else:
            samplers = [None, None]

        train_loader, val_loader = create_loader(datasets, samplers,
                                                  batch_size=[config['batch_size'], config['batch_size']],
                                                  num_workers=[4, 4], is_trains=[True, False], collate_fns=[None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])

        accumulate_steps = int(config.get('accumulate_steps', 2))

        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size) / accumulate_steps)
        arg_sche['min_rate'] = config['min_lr'] / arg_opt['lr'] if 'min_lr' in config else 0
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_dir)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0


        print('-----zero shot-----')
        model_without_ddp = model
        if hasattr(model, 'module'):
            model_without_ddp = model.module
        result = val(model_without_ddp, val_loader, tokenizer, device)
        results = collect_tensor_result(result, filename='epoch%d' % -1, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                        write_to_hdfs=world_size > 8)
        grounding_acc, avg_IoU = grounding_eval_bbox_vlue(results, config['val_file'][0])


        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            model_without_ddp = model
            if hasattr(model, 'module'):
                    model_without_ddp = model.module
            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                # 'epoch': epoch,
            }

            # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=epoch, training_states=optimizer.state_dict())

            if epoch == 0 or epoch == 1 or epoch == 5 or epoch == 9:
                print('-----Evaluate on Valid Set-----')
                print('Epoch: ', epoch)
                result = val(model_without_ddp, val_loader, tokenizer, device)
                results = collect_tensor_result(result, filename='epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                        write_to_hdfs=world_size > 8)

                if utils.is_main_process():

                    print('Epoch: ', epoch)
                    grounding_acc, avg_IoU = grounding_eval_bbox_vlue(results, config['val_file'][0])
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'{k}': v for k, v in grounding_acc.items()},
                                'epoch': epoch}


            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if epoch == 9:
                if args.distributed:
                    num_tasks = utils.get_world_size()
                    global_rank = utils.get_rank()
                    samplers = create_sampler([grd_test_dataset], [False], num_tasks, global_rank)
                else:
                    samplers = [None]

                test_loader = create_loader([grd_test_dataset], samplers,
                                            batch_size=[config['batch_size']],
                                            num_workers=[4], is_trains=[False], collate_fns=[None])

                result = val(model_without_ddp, test_loader[0], tokenizer, device)
                results = collect_tensor_result(result, filename='grounding_bbox_eval', local_wdir=args.result_dir,
                                                hdfs_wdir=args.output_hdfs,
                                                write_to_hdfs=world_size > 8)

                if utils.is_main_process():
                    # if 'vlue_test' in config.keys() and config['vlue_test']:
                    grounding_acc, avg_IoU = grounding_eval_bbox_vlue(results, config['test_file'][0])

                    log_stats = {**{f'{k}': v for k, v in grounding_acc.items()}}
                    print(log_stats)
                    print('Avg_IoU: ', avg_IoU)

            dist.barrier()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/Grounding_bbox.yaml')
    parser.add_argument('--output_dir', type=str, default='output/refcoco_bbox')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")
    parser.add_argument('--load_bbox_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")

    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    yaml = yaml.YAML(typ='rt')

    config = yaml.load(open(args.config, 'r'))

    utils.update_config(config, args.override_cfg)
    if utils.is_main_process():
        print('config:', json.dumps(config))
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)
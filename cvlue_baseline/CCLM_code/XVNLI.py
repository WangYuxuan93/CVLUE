import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_classification import XVLM4XVNLI

import utils
from utils.hdfs_io import hexists, hmkdir
from utils.checkpointer import Checkpointer
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   

    for i, (image, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, targets = image.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', max_length=config['max_tokens'], truncation=True, return_tensors="pt").to(device)  
        
        loss = model(image, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        image, targets = image.to(device), targets.to(device)   
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(image, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    train_dataset, val_dataset, test_dataset_dict = create_dataset('xvnli', config)
    datasets = [train_dataset, val_dataset]

    train_dataset_size = len(train_dataset)
    train_batch_size = config['batch_size']
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
        print(f"### Test: {[(k, len(dataset)) for k, dataset in test_dataset_dict.items()]}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']] * 2,
                                                          num_workers=[4, 4], is_trains=[True, False],
                                                          collate_fns=[None, None])

    test_loader_dict = {}
    for k, v in test_dataset_dict.items():
        test_loader_dict[k] = create_loader([v], [None], batch_size=[config['batch_size']],
                                            num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    print("Creating model")
    model = XVLM4XVNLI(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        val_stats = evaluate(model, val_loader, tokenizer, device)
        if utils.is_main_process():
            print({f'val_{k}': v for k, v in val_stats.items()}, flush=True)
        dist.barrier()

        for language, test_loader in test_loader_dict.items():
            test_stats = evaluate(model, test_loader, tokenizer, device)
            if utils.is_main_process():
                print({f'test_{language}_{k}': v for k, v in test_stats.items()}, flush=True)
            dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_dir)

        max_epoch = config['schedular']['epochs']

        best = 0
        best_epoch = 0

        for epoch in range(max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
            if epoch >= config['start_eval'] and (epoch + 1) % config['eval_interval'] == 0:
                val_stats = evaluate(model, val_loader, tokenizer, device)
                if utils.is_main_process():
                    print({f'dev_{k}': v for k, v in val_stats.items()}, flush=True)

                for language, test_loader in test_loader_dict.items():
                    test_stats = evaluate(model, test_loader, tokenizer, device)
                    if utils.is_main_process():
                        print({f'test_{language}_{k}': v for k, v in test_stats.items()}, flush=True)
                        with open(os.path.join(args.output_dir, 'eval_result_{}.txt'.format(epoch)), 'a') as f:
                            f.write(json.dumps({f'test_{language}_{k}': v for k, v in test_stats.items()}))
                    dist.barrier()

                if utils.is_main_process():
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_stats.items()},
                                'epoch': epoch}

                    if float(val_stats['acc']) > best:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        checkpointer.save_checkpoint(epoch, save_obj, train_stats)
                        best = float(val_stats['acc'])
                        best_epoch = epoch
                    elif epoch >= max_epoch - 1:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        checkpointer.save_checkpoint(epoch, save_obj, train_stats)

                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d, score: %.4f\n" % (best_epoch, best))
            if args.output_hdfs and not args.fewshot:
                os.system(f"hdfs dfs -put {os.path.join(args.output_dir, 'model_state_epoch_{}.th'.format(best_epoch))} {args.output_hdfs}")
                os.system(f"hdfs dfs -put {os.path.join(args.output_dir, 'model_state_epoch_{}.th'.format(max_epoch-1))} {args.output_hdfs}")
                os.system(f"hdfs dfs -put {os.path.join(args.output_dir, 'log.txt')} {args.output_hdfs}")
                os.system(f"hdfs dfs -put {os.path.join(args.output_dir, 'eval_result_{}.txt'.format(best_epoch))} {args.output_hdfs}")
                os.system(f"hdfs dfs -put {os.path.join(args.output_dir, 'eval_result_{}.txt'.format(max_epoch))} {args.output_hdfs}")

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/NLVR.yaml')
    parser.add_argument('--output_dir', default='output/nlvr')
    parser.add_argument('--output_hdfs', default='')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--load_nlvr_pretrain', action='store_true')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--fewshot', default='', type=str, help="IGLUE fewshot. <lang>,<shot_num>, eg: ar,25")
    parser.add_argument('--lr', default=0., type=float, help="learning rate")
    parser.add_argument('--gmt', action='store_true', help="whether use google machine translation as test set")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.output_hdfs and not hexists(args.output_hdfs):
        hmkdir(args.output_hdfs)

    if args.fewshot: # fewshot eg: ar,25
        for i, train_file in enumerate(config['train_file']):
            config['train_file'][i] = train_file.format(*args.fewshot.split(','))
        for i, val_file in enumerate(config['val_file']):
            config['val_file'][i] = val_file.format(args.fewshot.split(',')[0])

    if args.lr != 0.:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    
    if args.gmt:
        config['test_file'] = config['gmt_test_file']

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
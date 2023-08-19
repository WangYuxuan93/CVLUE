#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import copy

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.models.clip.modeling_clip import CLIPForVG
from transformers.models.blip.modeling_blip import BlipForVG
from transformers.models.chinese_clip.modeling_chinese_clip import ChineseCLIPForVG
# chinese_clip

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type:str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    decoder_model_path: str = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    ref_id_column: Optional[str] = field(
        default="ref_id",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    image_column: Optional[str] = field(
        default="image",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    width_column: Optional[str] = field(
        default="width",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    height_column: Optional[str] = field(
        default="height",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    bbox_column: Optional[str] = field(
        default="bbox",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    predict_result_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


dataset_name_mapping = {
    "vg.py": ("ref_id", "image","text", "width", "height", "bbox"),
}


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # training_args.n_gpu = 1

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1] #csv/json
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_type == 'clip':
        model = CLIPForVG.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            decoder_model_path= model_args.decoder_model_path,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    elif model_args.model_type == 'blip':
        model = BlipForVG.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif model_args.model_type == 'chinese':
        model = ChineseCLIPForVG.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return



    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.ref_id_column is None:
        ref_id_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        ref_id_column = data_args.ref_id_column
        if ref_id_column not in column_names:
            raise ValueError(
                f"--ref_id_column' value '{data_args.ref_id_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.image_column is None:
        image_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.text_column is None:
        text_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:  
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.width_column is None:
        width_column = dataset_columns[3] if dataset_columns is not None else column_names[3]
    else:
        width_column = data_args.width_column
        if width_column not in column_names:
            raise ValueError(
                f"--width_column' value '{data_args.width_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.height_column is None:
        height_column = dataset_columns[4] if dataset_columns is not None else column_names[4]
    else:
        height_column = data_args.height_column
        if height_column not in column_names:
            raise ValueError(
                f"--height_column' value '{data_args.height_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.bbox_column is None:
        bbox_column = dataset_columns[5] if dataset_columns is not None else column_names[5]
    else:
        bbox_column = data_args.bbox_column
        if bbox_column not in column_names:
            raise ValueError(
                f"--bbox_column' value '{data_args.bbox_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.

    # config.vision_config.image_size = 224
    image_transformations = Transform(
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)


    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    new_image_size = config.vision_config.image_size

    def trans_bbox(bbox, width, height):
        x , y , w , h = bbox

        new_x = x / width * new_image_size
        new_y = y / height * new_image_size
        new_w = w / width * new_image_size
        new_h = h / height * new_image_size

        center_x = new_x + new_w / 2
        center_y = new_y - new_h / 2

        return torch.tensor([center_x / new_image_size, center_y / new_image_size, new_w / new_image_size, new_h / new_image_size], dtype=torch.float)


    def tokenize_texts(examples):
        texts = list(examples[text_column])
        text_inputs = tokenizer(texts, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["text_input_ids"] = text_inputs.input_ids
        examples["text_attention_mask"] = text_inputs.attention_mask

        target_bbox = examples[bbox_column]
        width = examples[width_column]
        height = examples[height_column]

        new_target_bbox = []
        for i in range(len(target_bbox)):
            new_target_bbox.append(trans_bbox(target_bbox[i], width[i], height[i]))
        examples["target_bbox"] = new_target_bbox

        return examples

    def transform_images(examples):
        images = [read_image('/work/data/cvlue/figs/' + image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open('/work/data/cvlue/figs/' + image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        train_dataset = train_dataset.map(
            function=tokenize_texts,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))

        test_dataset = test_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        test_dataset = test_dataset.map(
            function=tokenize_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    training_args.push_to_hub=False

    def compute_IOU(gold_bbox,pred_bbox):
        x1 = max(gold_bbox[0], pred_bbox[0])
        y1 = min(gold_bbox[1], pred_bbox[1])
        x2 = min(gold_bbox[0] + gold_bbox[2] , pred_bbox[0] + pred_bbox[2] )
        y2 = max(gold_bbox[1] - gold_bbox[3] , pred_bbox[1] - pred_bbox[3] )
        
        if x1 < x2 and y1 > y2:
            inter = (x2 - x1) * (y1 - y2)
        else:
            inter = 0
        
        union = gold_bbox[2] * gold_bbox[3] + pred_bbox[2] * pred_bbox[3] - inter

        return float(inter) / union

    def compute_metrics(p):

        # print('p',p)
        # print('p.predictions',p.predictions)
        # print('p.predictions.shape',p.predictions.shape)
        # print('p.label_ids',p.label_ids)


        pred_bbox = p.predictions
        gold_bbox = dataset["validation"][bbox_column]

        temp_pred_bbox = copy.deepcopy(pred_bbox)

        w = dataset["validation"][width_column]
        h = dataset["validation"][height_column]

        bbox_num = len(gold_bbox)
        acc_bbox_num = 0
        all_iou = 0.

        for i in range(len(gold_bbox)):
            temp_pred_bbox[i][0] = temp_pred_bbox[i][0] * w[i]
            temp_pred_bbox[i][1] = temp_pred_bbox[i][1] * h[i]
            temp_pred_bbox[i][2] = temp_pred_bbox[i][2] * w[i]
            temp_pred_bbox[i][3] = temp_pred_bbox[i][3] * h[i]

            temp_pred_bbox[i][0] = temp_pred_bbox[i][0] - temp_pred_bbox[i][2] / 2
            temp_pred_bbox[i][1] = temp_pred_bbox[i][1] + temp_pred_bbox[i][3] / 2

            new_gold_bbox = trans_bbox(gold_bbox[i], w[i], h[i])
            new_gold_bbox[0] = new_gold_bbox[0] * w[i]
            new_gold_bbox[1] = new_gold_bbox[1] * h[i]
            new_gold_bbox[2] = new_gold_bbox[2] * w[i]
            new_gold_bbox[3] = new_gold_bbox[3] * h[i]
            new_gold_bbox[0] = new_gold_bbox[0] - new_gold_bbox[2] / 2
            new_gold_bbox[1] = new_gold_bbox[1] + new_gold_bbox[3] / 2

            iou = compute_IOU(gold_bbox[i], temp_pred_bbox[i])
            all_iou += iou

            if iou > 0.5:
                acc_bbox_num += 1
        
        accuracy = acc_bbox_num / bbox_num
        last_iou = all_iou / bbox_num

        return {
            "accuracy": accuracy,
            "last_iou": last_iou,
        }
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        text_input_ids = torch.tensor([example["text_input_ids"] for example in examples], dtype=torch.long)
        
        text_attention_mask = torch.tensor([example["text_attention_mask"] for example in examples], dtype=torch.long)
        target_bbox = torch.tensor([example["target_bbox"] for example in examples], dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "target_bbox": target_bbox,
            "return_loss": True,
        }

    # print('training_args.use_legacy_prediction_loop',training_args.use_legacy_prediction_loop)
    # training_args.use_legacy_prediction_loop = True

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    

    if training_args.do_predict:
        result = trainer.predict(test_dataset, metric_key_prefix="predict")

        print('result: ', result)
        print('result.predictions: ', result.predictions)
        print('result.label_ids: ', result.label_ids)


        test_predict_bbox = result.predictions
        raw_test_dataset = dataset["test"]
        new_result = []
        for i in range(len(test_predict_bbox)):
            temp_line = {}
            temp_line['ref_id'] = raw_test_dataset[i]['ref_id']
            temp_line['image'] = raw_test_dataset[i]['image']
            temp_line['text'] = raw_test_dataset[i]['text']
            temp_line['width'] = raw_test_dataset[i]['width']
            temp_line['height'] = raw_test_dataset[i]['height']
            temp_line['bbox'] = []
            test_predict_bbox[i][0] = test_predict_bbox[i][0] - test_predict_bbox[i][2] / 2
            test_predict_bbox[i][1] = test_predict_bbox[i][1] + test_predict_bbox[i][3] / 2
            temp_line['bbox'].append(test_predict_bbox[i][0] * raw_test_dataset[i]['width'])
            temp_line['bbox'].append(test_predict_bbox[i][1] * raw_test_dataset[i]['height'])
            temp_line['bbox'].append(test_predict_bbox[i][2] * raw_test_dataset[i]['width'])
            temp_line['bbox'].append(test_predict_bbox[i][3] * raw_test_dataset[i]['height'])

            new_result.append(temp_line)

        # print('result: ', new_result)
        predict_result_path = data_args.predict_result_path
        with open(predict_result_path, 'w',encoding='utf-8') as f:
            json.dump(new_result, f,ensure_ascii=False,indent=2)


    # 11. Write Training Stats and push to hub.
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

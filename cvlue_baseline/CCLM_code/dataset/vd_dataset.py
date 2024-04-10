import os
import json
import random
from random import random as rand

from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question

from torchvision.transforms.functional import hflip
from dataset import build_tokenizer

class visdial_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root=None, split="train", max_ques_words=512, answer_list='',
                 text_encoder=''):

        self.careful_hflip = True

        self.split = split
        self.ann = []

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        elif not isinstance(ann_file, list):
            raise ValueError

        for f in ann_file:
            ann = json.load(open(f, 'r'))
            if isinstance(ann, list):
                self.ann += ann
            elif isinstance(ann, dict):
                # test set & few-shot train set
                for k, v in ann.items():
                    v['question_id'] = k
                    v['img_id'] = v.pop('imageId')
                    v['sent'] = v.pop('question')

                    # few-shot train set
                    if split == 'train':
                        v['label'] = {v['answer']: 0}

                    self.ann.append(v)


        self.transform = transform
        #self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.sep_token

        # if split == 'test':
        #if split == 'test' or split == 'valid':
        #    self.max_ques_words = 50  # do not limit question length during test

        with open(answer_list, 'r', encoding='utf-8') as f:

            string = f.read()
            tmp_answer_list = json.loads(string)


        self.id2answer = {i:ans for i, ans in enumerate(tmp_answer_list)}


    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]


        image_path = ann['img_path']

        image = Image.open(image_path).convert('RGB')


        image = self.transform(image)

        if self.split == 'test' or self.split == 'valid':
            question = pre_question(ann['sent'], self.max_ques_words)
            question_id = int(ann['question_id'])
            rnd_id = int(ann["rnd_id"])
            img_id = int(ann["img_id"])
            ans_opts = [self.id2answer[id] for id in ann["ans_opt_ids"]]

            return image, question_id, question, ans_opts, img_id, rnd_id

        elif self.split == 'train':
            question = pre_question(ann['sent'], self.max_ques_words)

            answer_weight = {}
            for answer in ann['label'].keys():
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann['label'])
                else:
                    answer_weight[answer] = 1 / len(ann['label'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
            answers = [self.id2answer[int(id)] for id in answers]

            # answers = [answer + self.eos_token for answer in answers]  # fix bug

            return image, question, answers, weights

        else:
            raise NotImplementedError

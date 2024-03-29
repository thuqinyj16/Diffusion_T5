import os
import json
import re
import string
import numpy as np
import random
from tqdm import tqdm
from transformers import T5Tokenizer,BartTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]
        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyQAPromptDataset_AE(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 input_ids2, attention_mask2,
                 decoder_input_ids2, decoder_attention_mask2,
                 task_prefix, task_names, all_task_prompts, ontology, in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100,
                 type1_num=25, type2_num=25,
                 general_num=50, task2id=None):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.tokenizer2 = BartTokenizer.from_pretrained('facebook/bart-large')
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        input_ids2 = [self.prompt_prefix + input_id2[0:-(prompt_num)] for input_id2 in input_ids2]
        self.input_ids2 = torch.LongTensor(input_ids2)
        attention_mask2 = [[1]*prompt_num + atten_mask2[0:-(prompt_num)] for atten_mask2 in attention_mask2]
        self.attention_mask2 = torch.LongTensor(attention_mask2)

        self.decoder_input_ids2 = torch.LongTensor(decoder_input_ids2)
        self.decoder_attention_mask2 = torch.LongTensor(decoder_attention_mask2)

        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        self.all_task_prompts = all_task_prompts
        self.task_prefix = task_prefix
        self.task_names = task_names
        #print("self.task_names:",self.task_names)
        #print("self.task2id:",task2id)
        self.ontology = ontology
        self.type1_num = type1_num
        self.type2_num = type2_num
        self.general_num = general_num
        self.task2id = task2id

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]==len(self.task_prefix)==len(self.ontology)
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        # print('new')
        # print(self.ontology[idx])
        # print(self.task2id[self.task_names[idx]])
        ontology = self.ontology[idx]
        task_id = torch.LongTensor([self.task2id[self.task_names[idx]]])
        #print("task_id's size:",task_id)
        ontology_type1 = torch.LongTensor(list(range(ontology[0] * self.type1_num, (ontology[0] + 1) * self.type1_num)))
        ontology_type2 = torch.LongTensor(list(range(ontology[1] * self.type2_num, (ontology[1] + 1) * self.type2_num)))
        ontology_tensor_general = torch.LongTensor(list(range(self.task2id[self.task_names[idx]] * self.general_num, (self.task2id[self.task_names[idx]] + 1) * self.general_num)))

        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], self.input_ids2[idx], self.attention_mask2[idx], task_id,random.choice(self.all_task_prompts[self.task_prefix[idx]]), ontology_type1, ontology_type2, ontology_tensor_general, self.task_prefix[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        task_prompt = random.choice(self.all_task_prompts[self.task_prefix[idx]])
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx].masked_fill_(self.decoder_input_ids[out_idx]==self.tokenizer2.pad_token_id,-100), self.decoder_attention_mask[out_idx], self.input_ids2[in_idx], self.attention_mask2[in_idx], self.decoder_input_ids2[out_idx].masked_fill_(self.decoder_input_ids2[out_idx] == self.tokenizer.pad_token_id, -100), self.decoder_attention_mask2[out_idx], task_id,task_prompt, ontology_type1, ontology_type2, ontology_tensor_general

class MyQAPromptDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))

        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)

class MyMetaLearningDataset(Dataset):
    def __init__(self,
                 train_input_ids, train_attention_mask, 
                 train_decoder_input_ids, train_decoder_attention_mask,
                 train_metadata_task, train_metadata_questions,
                 dev_input_ids, dev_attention_mask,
                 dev_decoder_input_ids, dev_decoder_attention_mask,
                 dev_metadata_task, dev_metadata_questions, 
                 inner_bsz,
                 is_training=False):

        self.train_input_ids = torch.LongTensor(train_input_ids)
        self.train_attention_mask = torch.LongTensor(train_attention_mask)

        self.train_decoder_input_ids = torch.LongTensor(train_decoder_input_ids)
        self.train_decoder_attention_mask = torch.LongTensor(train_decoder_attention_mask)

        self.dev_input_ids = torch.LongTensor(dev_input_ids)
        self.dev_attention_mask = torch.LongTensor(dev_attention_mask)

        self.dev_decoder_input_ids = torch.LongTensor(dev_decoder_input_ids)
        self.dev_decoder_attention_mask = torch.LongTensor(dev_decoder_attention_mask)

        self.train_metadata_task = train_metadata_task
        self.train_metadata_questions = train_metadata_questions

        self.dev_metadata_task = dev_metadata_task
        self.dev_metadata_questions = dev_metadata_questions

        self.inner_bsz = inner_bsz
        self.is_training = is_training

        assert len(self.train_input_ids)==len(self.train_attention_mask)==self.train_metadata_task[-1][-1]
        assert len(self.train_decoder_input_ids)==len(self.train_decoder_attention_mask)==self.train_metadata_questions[-1][-1]

        assert len(self.dev_input_ids)==len(self.dev_attention_mask)==self.dev_metadata_task[-1][-1]
        assert len(self.dev_decoder_input_ids)==len(self.dev_decoder_attention_mask)==self.dev_metadata_questions[-1][-1]

        assert len(self.train_metadata_task) == len(self.dev_metadata_task)

    def __len__(self):
        return len(self.train_metadata_task)

    def __getitem__(self, idx):
        # train
        if self.inner_bsz <= self.train_metadata_task[idx][1] - self.train_metadata_task[idx][0]:
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=True)

        train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask = [], [], [], []
        for train_in_index in train_in_indices:
            train_input_ids.append(self.train_input_ids[train_in_index])
            train_attention_mask.append(self.train_attention_mask[train_in_index])

            train_out_idx = np.random.choice(range(*self.train_metadata_questions[train_in_index]))

            train_decoder_input_ids.append(self.train_decoder_input_ids[train_out_idx])
            train_decoder_attention_mask.append(self.train_decoder_attention_mask[train_out_idx])

        train_input_ids = torch.stack(train_input_ids)
        train_attention_mask = torch.stack(train_attention_mask)
        train_decoder_input_ids = torch.stack(train_decoder_input_ids)
        train_decoder_attention_mask = torch.stack(train_decoder_attention_mask)

        # dev
        if self.inner_bsz <= self.dev_metadata_task[idx][1] - self.dev_metadata_task[idx][0]:
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=True)

        dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask = [], [], [], []
        for dev_in_index in dev_in_indices:
            dev_input_ids.append(self.dev_input_ids[dev_in_index])
            dev_attention_mask.append(self.dev_attention_mask[dev_in_index])

            dev_out_idx = np.random.choice(range(*self.dev_metadata_questions[dev_in_index]))

            dev_decoder_input_ids.append(self.dev_decoder_input_ids[dev_out_idx])
            dev_decoder_attention_mask.append(self.dev_decoder_attention_mask[dev_out_idx])

        dev_input_ids = torch.stack(dev_input_ids)
        dev_attention_mask = torch.stack(dev_attention_mask)
        dev_decoder_input_ids = torch.stack(dev_decoder_input_ids)
        dev_decoder_attention_mask = torch.stack(dev_decoder_attention_mask)

        return train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask, \
            dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask

class MyMetaLearningPromptDataset(Dataset):
    def __init__(self,
                 train_input_ids, train_attention_mask, 
                 train_decoder_input_ids, train_decoder_attention_mask,
                 train_metadata_task, train_metadata_questions,
                 dev_input_ids, dev_attention_mask,
                 dev_decoder_input_ids, dev_decoder_attention_mask,
                 dev_metadata_task, dev_metadata_questions, 
                 inner_bsz,
                 is_training=False, prompt_num=100):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]

        train_input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in train_input_ids]
        self.train_input_ids = torch.LongTensor(train_input_ids)
        train_attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in train_attention_mask]
        self.train_attention_mask = torch.LongTensor(train_attention_mask)

        self.train_decoder_input_ids = torch.LongTensor(train_decoder_input_ids)
        self.train_decoder_attention_mask = torch.LongTensor(train_decoder_attention_mask)

        dev_input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in dev_input_ids]
        self.dev_input_ids = torch.LongTensor(dev_input_ids)
        dev_attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in dev_attention_mask]
        self.dev_attention_mask = torch.LongTensor(dev_attention_mask)

        self.dev_decoder_input_ids = torch.LongTensor(dev_decoder_input_ids)
        self.dev_decoder_attention_mask = torch.LongTensor(dev_decoder_attention_mask)

        self.train_metadata_task = train_metadata_task
        self.train_metadata_questions = train_metadata_questions

        self.dev_metadata_task = dev_metadata_task
        self.dev_metadata_questions = dev_metadata_questions

        self.inner_bsz = inner_bsz
        self.is_training = is_training

        assert len(self.train_input_ids)==len(self.train_attention_mask)==self.train_metadata_task[-1][-1]
        assert len(self.train_decoder_input_ids)==len(self.train_decoder_attention_mask)==self.train_metadata_questions[-1][-1]

        assert len(self.dev_input_ids)==len(self.dev_attention_mask)==self.dev_metadata_task[-1][-1]
        assert len(self.dev_decoder_input_ids)==len(self.dev_decoder_attention_mask)==self.dev_metadata_questions[-1][-1]

        assert len(self.train_metadata_task) == len(self.dev_metadata_task)

    def __len__(self):
        return len(self.train_metadata_task)

    def __getitem__(self, idx):
        # train
        if self.inner_bsz <= self.train_metadata_task[idx][1] - self.train_metadata_task[idx][0]:
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=True)

        train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask = [], [], [], []
        for train_in_index in train_in_indices:
            train_input_ids.append(self.train_input_ids[train_in_index])
            train_attention_mask.append(self.train_attention_mask[train_in_index])

            train_out_idx = np.random.choice(range(*self.train_metadata_questions[train_in_index]))

            train_decoder_input_ids.append(self.train_decoder_input_ids[train_out_idx])
            train_decoder_attention_mask.append(self.train_decoder_attention_mask[train_out_idx])

        train_input_ids = torch.stack(train_input_ids)
        train_attention_mask = torch.stack(train_attention_mask)
        train_decoder_input_ids = torch.stack(train_decoder_input_ids)
        train_decoder_attention_mask = torch.stack(train_decoder_attention_mask)

        # dev
        if self.inner_bsz <= self.dev_metadata_task[idx][1] - self.dev_metadata_task[idx][0]:
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=True)

        dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask = [], [], [], []
        for dev_in_index in dev_in_indices:
            dev_input_ids.append(self.dev_input_ids[dev_in_index])
            dev_attention_mask.append(self.dev_attention_mask[dev_in_index])

            dev_out_idx = np.random.choice(range(*self.dev_metadata_questions[dev_in_index]))

            dev_decoder_input_ids.append(self.dev_decoder_input_ids[dev_out_idx])
            dev_decoder_attention_mask.append(self.dev_decoder_attention_mask[dev_out_idx])

        dev_input_ids = torch.stack(dev_input_ids)
        dev_attention_mask = torch.stack(dev_attention_mask)
        dev_decoder_input_ids = torch.stack(dev_decoder_input_ids)
        dev_decoder_attention_mask = torch.stack(dev_decoder_attention_mask)

        return train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask, \
            dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask


class MyMetaLearningDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        super(MyMetaLearningDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)
        self.collate_fn = self.dummy_collate
        self.args = args

    def dummy_collate(self, input_data):
        return input_data

    def inference_dataloader(self):
        bsz = self.args.predict_batch_size
        for idx, (start_idx, end_idx) in enumerate(self.dataset.metadata_rel):
            input_ids_for_this_rel = self.dataset.input_ids[start_idx: end_idx]
            masks_for_this_rel = self.dataset.attention_mask[start_idx: end_idx]
            for j in range(0, len(input_ids_for_this_rel), bsz):
                input_ids_this_batch = input_ids_for_this_rel[j: j+bsz]
                masks_for_this_batch = masks_for_this_rel[j: j+bsz]

                yield self.dataset.relation_ids[idx], self.dataset.relation_mask[idx], input_ids_this_batch, masks_for_this_batch

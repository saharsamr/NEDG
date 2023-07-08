import random

from torch.utils.data import Dataset
import torch
import re

from GNED.config import *


class WikiDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels, mask_entity=False, max_length=INPUT_GENERATION_MAX_LENGTH):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.mask_entity = mask_entity

    def __len__(self):

        return len(self.inputs)

    def masking_entities(self, input_ids, input_text):

        new_input_ids = input_ids
        new_input_text = input_text

        entity_start_token_id = self.tokenizer.convert_tokens_to_ids('<NE>')
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids('</NE>')

        entity_start_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_start_token_id]
        entity_end_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_end_token_id]

        for st, et in zip(entity_start_token_indices, entity_end_token_indices):
            new_input_ids[st+1:et] = [self.tokenizer.convert_tokens_to_ids('<mask>') for _ in range(st+1, et)]

        entity_names = re.findall(r'<NE>(.*?)</NE>', input_text)
        for name, st, et in zip(entity_names, entity_start_token_indices, entity_end_token_indices):
            new_input_text = new_input_text.replace(f'<NE>{name}</NE>', f'<NE>{"<mask>"*len(range(st+1, et))}</NE>')

        return new_input_ids, new_input_text

    def __getitem__(self, idx):

        input_encodings = self.tokenizer(
            self.inputs[idx], padding='max_length', truncation=True, max_length=self.max_len)
        output_encodings = self.tokenizer(
            self.labels[idx], padding='max_length', truncation=True, max_length=OUTPUT_GENERATION_MAX_LENGTH)
        input_text = self.inputs[idx]

        if self.mask_entity and random.random() < MASK_PROB:
            input_encodings['input_ids'], input_text = \
                self.masking_entities(input_encodings['input_ids'], self.inputs[idx])

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'input_text': input_text
        }

        if self.labels:
            item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




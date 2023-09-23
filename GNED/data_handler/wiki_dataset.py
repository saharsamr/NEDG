import random

from torch.utils.data import Dataset
import torch
import re

from GNED.config import *


class WikiDataset(Dataset):

    def __init__(
      self, tokenizer, inputs, labels, entity_names, mask_entity=False, max_length=None, is_gpt=False
    ):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.mask_entity = mask_entity
        self.entity_names = entity_names
        self.is_gpt = is_gpt
        if self.is_gpt:
            self.description_token = self.tokenizer.convert_tokens_to_ids('<dscrp>')

    def __len__(self):

        return len(self.inputs)

    def masking_entities(self, input_ids, input_text, method='complete'):

        new_input_ids = input_ids
        new_input_text = input_text

        entity_start_token_id = self.tokenizer.convert_tokens_to_ids('<NE>')
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids('</NE>')

        entity_start_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_start_token_id]
        entity_end_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_end_token_id]

        for st, et in zip(entity_start_token_indices, entity_end_token_indices):
            if method == 'complete':
                new_input_ids[st+1:et] = [self.tokenizer.convert_tokens_to_ids('<mask>') for _ in range(st+1, et)]
            if method == 'partial':
                new_input_ids[st+1:et] = [
                    self.tokenizer.convert_tokens_to_ids('<mask>') if random.random() < MASK_PROB else
                    new_input_ids[i] for i in range(st+1, et)]

        entity_names = re.findall(r'<NE>(.*?)</NE>', input_text)
        for name, st, et in zip(entity_names, entity_start_token_indices, entity_end_token_indices):
            new_input_text = new_input_text.replace(f'<NE>{name}</NE>', f'<NE>{"<mask>"*len(range(st+1, et))}</NE>')

        return new_input_ids, new_input_text

    def __getitem__(self, idx):

        if self.is_gpt:
            text = '<cntxt>' + self.inputs[idx] + '<dscrp>' + self.labels[idx]
            input_encodings = self.tokenizer(
                text, padding='max_length', truncation=True,
                max_length=INPUT_GENERATION_MAX_LENGTH+OUTPUT_GENERATION_MAX_LENGTH+2
            )
        else:
            input_encodings = self.tokenizer(
                self.inputs[idx], padding='max_length', truncation=True, max_length=INPUT_GENERATION_MAX_LENGTH)

        output_encodings = self.tokenizer(
            self.labels[idx], padding='max_length', truncation=True, max_length=OUTPUT_GENERATION_MAX_LENGTH)
        input_text = self.inputs[idx]
        entity_name = self.entity_names[idx]

        if self.mask_entity and MASKING_STRATEGY == 'Complete' and random.random() < MASK_PROB:
            input_encodings['input_ids'], input_text = \
                self.masking_entities(input_encodings['input_ids'], self.inputs[idx], method='complete')
        elif self.mask_entity and MASKING_STRATEGY == 'Partial':
            input_encodings['input_ids'], input_text = \
                self.masking_entities(input_encodings['input_ids'], self.inputs[idx], method='partial')

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'input_text': input_text,
            'entity_name': entity_name
        }

        if self.is_gpt:
            item['target_output'] = torch.tensor(output_encodings['input_ids'])
            description_token = input_encodings['input_ids'].index(self.description_token)
            item['actual_input'] = torch.tensor(input_encodings['input_ids'][:description_token+1])
            item['actual_attention_mask'] = torch.tensor(input_encodings['attention_mask'][:description_token+1])

        if self.labels:
            if self.is_gpt:
                item['labels'] = torch.tensor(input_encodings['input_ids'])
            else:
                item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




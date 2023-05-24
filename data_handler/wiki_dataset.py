from torch.utils.data import Dataset
import torch
from config import INPUT_GENERATION_MAX_LENGTH, OUTPUT_GENERATION_MAX_LENGTH


class WikiDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels, mask_entity=False, max_length=INPUT_GENERATION_MAX_LENGTH):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.mask_entity = mask_entity

    def __len__(self):

        return len(self.inputs)

    def masking_entities(self, input_ids, attention_mask):

        new_input_ids = input_ids
        new_attention_mask = attention_mask

        entity_start_token_id = self.tokenizer.convert_tokens_to_ids('<NE>')
        entity_end_token_id = self.tokenizer.convert_tokens_to_ids('</NE>')

        entity_start_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_start_token_id]
        entity_end_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_end_token_id]

        for st, et in zip(entity_start_token_indices, entity_end_token_indices):
            new_input_ids[st+1:et] = [self.tokenizer.convert_tokens_to_ids('<mask>') for _ in range(st+1,et)]
            new_attention_mask[st+1:et] = [0 for _ in range(st+1,et)]

        return new_input_ids, new_attention_mask

    def __getitem__(self, idx):

        input_encodings = self.tokenizer(
            self.inputs[idx], padding='max_length', truncation=True, max_length=self.max_len)
        output_encodings = self.tokenizer(
            self.labels[idx], padding='max_length', truncation=True, max_length=OUTPUT_GENERATION_MAX_LENGTH)

        if self.mask_entity:
            input_encodings['input_ids'], input_encodings['attention_mask'] = \
                self.masking_entities(input_encodings['input_ids'], input_encodings['attention_mask'])

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask'])
        }

        if self.labels:
            item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




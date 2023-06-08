from torch.utils.data import Dataset
import torch

from GNED.config import INPUT_CLASSIFICATION_MAX_LENGTH


class ClassificationDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels=None, max_length=INPUT_CLASSIFICATION_MAX_LENGTH):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        input_encodings = self.tokenizer.encode_plus(
            self.inputs[idx], add_special_tokens=True,
            padding='max_length', max_length=self.max_len,
            truncation=True
            )

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask'])
        }

        if self.labels:
            item['labels'] = torch.tensor(int(self.labels[idx]))

        return item

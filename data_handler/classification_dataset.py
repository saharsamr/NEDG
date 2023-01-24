from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels=None, max_length=256):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        input_encodings = self.tokenizer(self.inputs[idx], padding='longest', truncation=True, max_length=self.max_len)

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask'])
        }

        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])

        return item




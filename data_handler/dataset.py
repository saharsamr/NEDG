from torch.utils.data import Dataset
import torch


class WikiDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels=None, max_length=256):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        input_encodings = self.tokenizer(self.inputs[idx], padding=True, truncation=True, max_length=self.max_len)
        output_encodings = self.tokenizer(self.labels[idx], padding='max_length', truncation=True, max_length=40)
        item = {'input_ids': torch.tensor(input_encodings['input_ids'])}
        if self.labels:
            item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




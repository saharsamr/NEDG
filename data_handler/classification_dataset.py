from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):

    def __init__(self, tokenizer, inputs, labels=None, max_length=512):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        input_encodings = self.tokenizer.encode_plus(
            self.inputs[idx], add_special_tokens=True,
            padding='max_length', max_length=self.max_len
            )

        item = {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask'])
        }

        if self.labels:
            labels = [1.0, 0.0] if self.labels[idx] == 0 else [0.0, 1.0]
            item['labels'] = torch.tensor(labels)

        return item

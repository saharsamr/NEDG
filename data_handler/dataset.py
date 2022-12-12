from torch.utils.data import Dataset
import torch
from config import INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH


class WikiDataset(Dataset):

    def __init__(
      self, tokenizer, inputs, labels=None,
      input_max_length=INPUT_MAX_LENGTH, output_max_length=OUTPUT_MAX_LENGTH
    ):

        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.input_max_len = input_max_length
        self.output_max_len = output_max_length

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        input_encodings = self.tokenizer(self.inputs[idx], padding=True, truncation=True, max_length=self.input_max_len)
        output_encodings = self.tokenizer(self.labels[idx], padding=True, truncation=True, max_length=self.output_max_len)
        item = {'input_ids': torch.tensor(input_encodings['input_ids'])}
        if self.labels:
            item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




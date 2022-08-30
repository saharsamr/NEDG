from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.save_data import save_tokenized_datasets
import torch


def create_train_dev_test_datasets(data, tokenizer, max_length):

    X, Y = list(data['context']), list(data['description'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1)

    train_dataset = WikiDataset(tokenizer, x_train, y_train, max_length)
    dev_dataset = WikiDataset(tokenizer, x_dev, y_dev, max_length)
    test_dataset = WikiDataset(tokenizer, x_test, y_test, max_length)

    return train_dataset, dev_dataset, test_dataset


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
        output_encodings = self.tokenizer(self.labels[idx], padding=True, truncation=True, max_length=self.max_len)
        item = {'input_ids': torch.tensor(input_encodings['input_ids'])}
        if self.labels:
            item['labels'] = torch.tensor(output_encodings['input_ids'])

        return item




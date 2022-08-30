from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.save_data import save_tokenized_datasets
import torch


def create_train_dev_test_datasets(data, tokenizer):

    X, Y = list(data['context']), list(data['description'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1)

    train_dataset = WikiDataset(x_train, y_train)
    dev_dataset = WikiDataset(x_dev, y_dev)
    test_dataset = WikiDataset(x_test, y_test)

    return train_dataset, dev_dataset, test_dataset


class WikiDataset(Dataset):

    def __init__(self, inputs, labels=None):

        self.inputs = inputs
        self.labels = labels

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        item = {'input_ids': torch.tensor(self.inputs[idx])}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])

        return item




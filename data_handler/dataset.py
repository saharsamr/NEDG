from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.save_data import save_tokenized_datasets
import torch


def create_train_dev_test_datasets(data, tokenizer, max_len):

    X, Y = list(data['context']), list(data['description'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1)

    print('Tokenizing train data')
    x_train = tokenizer(x_train, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(x_train, 'input_train.pkl')
    y_train = tokenizer(y_train, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(y_train, 'output_train.pkl')
    print('Tokenizing dev data')
    x_dev = tokenizer(x_dev, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(x_dev, 'input_dev.pkl')
    y_dev = tokenizer(y_dev, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(y_dev, 'output_dev.pkl')
    print('Tokenizing test data')
    x_test = tokenizer(x_test, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(x_test, 'input_test.pkl')
    y_test = tokenizer(y_test, padding=True, truncation=True, max_length=max_len)
    save_tokenized_datasets(y_test, 'output_test.pkl')

    train_dataset = WikiDataset(x_train, y_train)
    dev_dataset = WikiDataset(x_dev, y_dev)
    test_dataset = WikiDataset(x_test, y_test)

    return train_dataset, dev_dataset, test_dataset


class WikiDataset(Dataset):

    def __init__(self, encodings, labels=None):

        self.encodings = encodings
        self.labels = labels

    def __len__(self):

        return len(self.encodings)

    def __getitem__(self, idx):

        item = {'input_ids': torch.tensor(self.encodings['input_ids'][idx])}
        if self.labels:
            item['labels'] = torch.tensor(self.labels['input_ids'][idx])

        return item




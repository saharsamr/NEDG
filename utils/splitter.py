import torch


def split_data(dataset, train_percentage, dev_percentage, test_percentage):

    train_len = int(train_percentage*len(dataset))
    dev_len = int(dev_percentage*len(dataset))
    test_len = len(dataset) - (train_len+dev_len)

    train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, dev_len, test_len])

    return train_dataset, dev_dataset, test_dataset


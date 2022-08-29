import pickle


def save_predictions(input, label, prediction):

    with open('preds.csv', 'w') as f:
        for i, l, p in zip(input, label, prediction):
            f.write(f'{i}||{l}||{p}\n')


def save_tokenized_datasets(dataset, name):

    with open(name, 'wb') as f:
        pickle.dump(dataset, f)

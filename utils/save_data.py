import pickle


def save_predictions(input_, label, prediction, delimiter='~'):

    with open('preds.csv', 'w') as f:
        for i, l, p in zip(input_, label, prediction):
            i = i.replace(delimiter, '')
            l = l.replace(delimiter, '')
            p = p.replace(delimiter, '')
            f.write(f'{i}{delimiter}{l}{delimiter}{p}\n')


def save_tokenized_datasets(dataset, name):

    with open(name, 'wb') as f:
        pickle.dump(dataset, f)

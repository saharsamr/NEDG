import pickle
from config import PRED_CLASSIFICATION_FILE_PATH, PRED_GENERATION_FILE_PATH


def save_generation_predictions(input_, label, prediction, delimiter='\1'):

    with open(PRED_GENERATION_FILE_PATH, 'w+') as f:
        for i, l, p in zip(input_, label, prediction):
            i = i.replace(delimiter, '')
            l = l.replace(delimiter, '')
            p = p.replace(delimiter, '')
            f.write(f'{i}{delimiter}{l}{delimiter}{p}\n')


def save_classification_prediction(input_, label, prediction, delimiter='\1'):

    with open(PRED_CLASSIFICATION_FILE_PATH, 'w') as f:
        for i, l, p in zip(input_, label, prediction):
            i = i.replace(delimiter, '')
            f.write(f'{i}{delimiter}{l}{delimiter}{p}\n')


def save_tokenized_datasets(dataset, name):

    with open(name, 'wb') as f:
        pickle.dump(dataset, f)

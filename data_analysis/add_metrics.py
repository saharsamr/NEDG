from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

from data_analysis.utils import compute_bleu, compute_rouge, compute_bertscore
from data_analysis.config import *


def add_metrics(df):
    descriptions = df['label']
    model_preds = df['model-pred']

    model_preds = [word_tokenize(pred) for pred in model_preds]
    descriptions = [[word_tokenize(description)] for description in descriptions]

    bleu = [
        compute_bleu([pred], [label], 1) for
        pred, label in tqdm(zip(model_preds, descriptions), total=len(model_preds))]

    rouge = [
        compute_rouge([pred], [label]) for
        pred, label in tqdm(zip(model_preds, descriptions), total=len(model_preds))]

    bert_score = compute_bertscore(model_preds, descriptions)

    df['bleu'] = bleu
    df['rouge'] = rouge
    df['bert-score'] = bert_score

    return df


if __name__ == '__main__':
    data = pd.read_csv(TEST_RESULTS, delimiter='\1', names=['context', 'label', 'model-pred', 'entity_name'], header=None)
    data = data.fillna('')
    data = add_metrics(data)
    data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)

from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

from data_analysis.utils import compute_bleu, compute_rouge
from data_analysis.config import *


def add_bleu_rouge(df):

    descriptions = df['label']
    CPE_preds = df['CPE-pred']
    CME_preds = df['CME-pred']

    CPE_preds = [word_tokenize(pred) for pred in CPE_preds]
    CME_preds = [word_tokenize(pred) for pred in CME_preds]
    descriptions = [[word_tokenize(description)] for description in descriptions]

    CPE_bleu = [
        compute_bleu([cpe_pred], [label], 1) for
        cpe_pred, label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_bleu = [
        compute_bleu([cme_pred], [label], 1) for
        cme_pred, label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]

    CPE_rouge = [
        compute_rouge([cpe_pred], [label]) for
        cpe_pred, label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_rouge = [
        compute_rouge([cme_pred], [label]) for
        cme_pred, label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]

    df['CPE-bleu'] = CPE_bleu
    df['CME-bleu'] = CME_bleu
    df['CPE-rouge'] = CPE_rouge
    df['CME-rouge'] = CME_rouge

    return df


data = pd.read_csv(TEST_CLASSIFICATION_FILE, delimiter='\1')
csme_data = pd.read_csv(CSME_TEST_FILE, delimiter='\1')
print(len(data), len(csme_data))
data = data.dropna()
csme_data = csme_data.dropna()
print(len(data), len(csme_data))
data = data[data['title'].isin(csme_data['title'])]
print(len(data))
# print(data.head)
# print(csme_data.head)
# data = add_bleu_rouge(data)
# data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)

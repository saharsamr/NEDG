from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

from data_analysis.utils import compute_bleu, compute_rouge
from data_analysis.config import *


def add_bleu_rouge(df):

    descriptions = df['label']
    CPE_preds = df['CPE-pred']
    CME_preds = df['CME-pred']
    CSME_preds = df['CSME-pred']

    CPE_preds = [word_tokenize(pred) for pred in CPE_preds]
    CME_preds = [word_tokenize(pred) for pred in CME_preds]
    CSME_preds = [word_tokenize(pred) for pred in CSME_preds]
    descriptions = [[word_tokenize(description)] for description in descriptions]

    CPE_bleu = [
        compute_bleu([cpe_pred], [label], 1) for
        cpe_pred, label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_bleu = [
        compute_bleu([cme_pred], [label], 1) for
        cme_pred, label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]
    CSME_bleu = [
        compute_bleu([csme_pred], [label], 1) for
        csme_pred, label in tqdm(zip(CSME_preds, descriptions), total=len(CSME_preds))]

    CPE_rouge = [
        compute_rouge([cpe_pred], [label]) for
        cpe_pred, label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_rouge = [
        compute_rouge([cme_pred], [label]) for
        cme_pred, label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]
    CSME_rouge = [
        compute_rouge([csme_pred], [label]) for
        csme_pred, label in tqdm(zip(CSME_preds, descriptions), total=len(CSME_preds))]

    df['CPE-bleu'] = CPE_bleu
    df['CME-bleu'] = CME_bleu
    df['CSME-bleu'] = CSME_bleu
    df['CPE-rouge'] = CPE_rouge
    df['CME-rouge'] = CME_rouge
    df['CSME-rouge'] = CSME_rouge

    return df


data = pd.read_csv(TEST_CLASSIFICATION_FILE, delimiter='\1')
csme_data = pd.read_csv(CSME_TEST_FILE, delimiter='\1')
data = data.dropna()
csme_data = csme_data.dropna()
print(len(data), len(csme_data))
data = data[data['title'].isin(csme_data['title'])]
csme_data = csme_data[csme_data['title'].isin(data['title'])]
data['CSME-pred'] = csme_data['CSME-pred']
data['CSME-context'] = csme_data['CSME-context']
data['CSME-bert'] = csme_data['CSME-bert']
data = add_bleu_rouge(data)
data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)

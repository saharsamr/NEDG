from datasets import load_metric
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from tqdm import tqdm


bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')
bertscore = load_metric('bertscore')


def compute_correlation(x, y):

    return {
        'spearman': scipy.stats.spearmanr(x, y),
        'pearson': scipy.stats.pearsonr(x, y)
    }


def compute_metrics_for_popularity(df):

    descriptions = df['description'].values
    CPE_preds = df['cpe-pred'].values
    CME_preds = df['cme-pred'].values

    CPE_preds = [word_tokenize(pred) for pred in CPE_preds]
    CME_preds = [word_tokenize(pred) for pred in CME_preds]
    descriptions = [[word_tokenize(description)] for description in descriptions]

    CPE_bert = compute_bertscore(CPE_preds, descriptions)
    CME_bert = compute_bertscore(CME_preds, descriptions)

    CPE_bleu = [
        compute_bleu([cpe_pred], [cpe_label], 1) for
        cpe_pred, cpe_label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_bleu = [
        compute_bleu([cme_pred], [cme_label], 1) for
        cme_pred, cme_label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]

    CPE_rouge = [
        compute_rouge([cpe_pred], [cpe_label]) for
        cpe_pred, cpe_label in tqdm(zip(CPE_preds, descriptions), total=len(CPE_preds))]
    CME_rouge = [
        compute_rouge([cme_pred], [cme_label]) for
        cme_pred, cme_label in tqdm(zip(CME_preds, descriptions), total=len(CME_preds))]

    df['CPE-bert'] = CPE_bert
    df['CME-bert'] = CME_bert
    df['CPE-bleu'] = CPE_bleu
    df['CME-bleu'] = CME_bleu
    df['CPE-rouge'] = CPE_rouge
    df['CME-rouge'] = CME_rouge

    return df


def compute_metrics(classification_result):

    metrics = {}

    cpe_data = classification_result[classification_result['CPE-pred'] != ''][['CPE-pred', 'label', 'popularity']].dropna()
    cpe_preds = [word_tokenize(pred) for pred in cpe_data['CPE-pred']]
    cpe_labels = [[word_tokenize(label)] for label in cpe_data['label']]
    metrics['CPE'] = {
        # 'bertscore': compute_bertscore(cpe_preds, cpe_labels),
        'bleu': [
            compute_bleu([cpe_pred], [cpe_label], 1) for
            cpe_pred, cpe_label in tqdm(zip(cpe_preds, cpe_labels), total=len(cpe_preds)) if len(cpe_label)],
        'popularity': cpe_data['popularity'].values,
    }

    cme_data = classification_result[classification_result['CME-pred'] != ''][['CME-pred', 'label', 'popularity']].dropna()
    cme_preds = [word_tokenize(pred) for pred in cme_data['CME-pred']]
    cme_labels = [[word_tokenize(label)] for label in cme_data['label']]
    metrics['CME'] = {
        'bertscore': compute_bertscore(cme_preds, cme_labels),
        'bleu': [
            compute_bleu([cme_pred], [cme_label], 1)['bleu']
            for cme_pred, cme_label in tqdm(zip(cme_preds, cme_labels), total=len(cme_preds)) if len(cme_label)],
        'popularity': cme_data['popularity'].values,

    }

    classification_result['Hybrid-pred'] = [
        sample['CPE-pred'] if sample['class-pred'] == 1 else sample['CME-pred']
        for _, sample in classification_result.iterrows()
    ]
    hybrid_data = classification_result[classification_result['Hybrid-pred'] != ''][['Hybrid-pred', 'label', 'popularity']].dropna()
    hybrid_preds = [word_tokenize(pred) for pred in hybrid_data['Hybrid-pred']]
    hybrid_labels = [[word_tokenize(label)] for label in hybrid_data['label']]
    metrics['Hybrid'] = {
        'bertscore': compute_bertscore(hybrid_preds, hybrid_labels),
        'bleu': [
            compute_bleu([hybrid_pred], [hybrid_label], 1)['bleu']
            for hybrid_pred, hybrid_label in tqdm(zip(hybrid_preds, hybrid_labels), total=len(hybrid_preds))
            if len(hybrid_label)],
        'popularity': hybrid_data['popularity'].values,
    }

    return metrics


def add_bleu_rouge_to_df(df):

    cpe_preds = [word_tokenize(pred) for pred in df['CPE-pred']]
    cme_preds = [word_tokenize(pred) for pred in df['CME-pred']]
    labels = [[word_tokenize(label)] for label in df['label']]

    df['CPE-bleu'] = [
        compute_bleu([cpe_pred], [cpe_label], 1) for
        cpe_pred, cpe_label in tqdm(zip(cpe_preds, labels), total=len(cpe_preds))]
    df['CME-bleu'] = [
        compute_bleu([cme_pred], [cme_label], 1) for
        cme_pred, cme_label in tqdm(zip(cme_preds, labels), total=len(cme_preds))]

    df['CPE-rouge'] = [
        compute_rouge([cpe_pred], [cpe_label]) for
        cpe_pred, cpe_label in tqdm(zip(cpe_preds, labels), total=len(cpe_preds))]
    df['CME-rouge'] = [
        compute_rouge([cme_pred], [cme_label]) for
        cme_pred, cme_label in tqdm(zip(cme_preds, labels), total=len(cme_preds))]

    return df


def remove_empty_preds(df):
    df = df[df['CPE-bert'] != 0]
    df = df[df['CME-bert'] != 0]
    return df


def compute_metrics_for_every_fraction(df):

    cpe_bert = np.mean(df["CPE-bert"].values)
    cme_bert = np.mean(df["CME-bert"].values)

    cpe_bleu = np.mean(df["CPE-bleu"].values)
    cme_bleu = np.mean(df["CME-bleu"].values)

    cpe_rouge = np.mean(df["CPE-rouge"].values)
    cme_rouge = np.mean(df["CME-rouge"].values)

    return cpe_bert, cme_bert, cpe_bleu, cme_bleu, cpe_rouge, cme_rouge


def compute_bleu(preds, labels, max_order):
    bleu_output = bleu_metric.compute(
        predictions=preds, references=labels, max_order=max_order
    )

    return bleu_output['bleu']


def compute_rouge(preds, labels):
    rouge_output = rouge_metric.compute(
        predictions=preds, references=labels,
        rouge_types=['rougeL']
    )

    return rouge_output['rougeL'][1][2]


def compute_bertscore(preds, labels):
    bertscore_output = bertscore.compute(
        predictions=preds, references=labels, lang='en', model_type='bert-base-uncased'
    )

    return bertscore_output['f1']

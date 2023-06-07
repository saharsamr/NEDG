from datasets import load_metric
from nltk.tokenize import word_tokenize
import scipy.stats
from tqdm import tqdm


bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')
bertscore = load_metric('bertscore')


def compute_correlation(x, y):

    return {
        'spearman': scipy.stats.spearmanr(x, y),
        'pearson': scipy.stats.pearsonr(x, y)
    }


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

    # cme_data = classification_result[classification_result['CME-pred'] != ''][['CME-pred', 'label', 'popularity']].dropna()
    # cme_preds = [word_tokenize(pred) for pred in cme_data['CME-pred']]
    # cme_labels = [[word_tokenize(label)] for label in cme_data['label']]
    # metrics['CME'] = {
    #     'bertscore': compute_bertscore(cme_preds, cme_labels),
    #     'bleu': [
    #         compute_bleu([cme_pred], [cme_label], 1)['bleu']
    #         for cme_pred, cme_label in tqdm(zip(cme_preds, cme_labels), total=len(cme_preds)) if len(cme_label)],
    #     'popularity': cme_data['popularity'].values,
    #
    # }
    #
    # classification_result['Hybrid-pred'] = [
    #     sample['CPE-pred'] if sample['class-pred'] == 1 else sample['CME-pred']
    #     for _, sample in classification_result.iterrows()
    # ]
    # hybrid_data = classification_result[classification_result['Hybrid-pred'] != ''][['Hybrid-pred', 'label', 'popularity']].dropna()
    # hybrid_preds = [word_tokenize(pred) for pred in hybrid_data['Hybrid-pred']]
    # hybrid_labels = [[word_tokenize(label)] for label in hybrid_data['label']]
    # metrics['Hybrid'] = {
    #     'bertscore': compute_bertscore(hybrid_preds, hybrid_labels),
    #     'bleu': [
    #         compute_bleu([hybrid_pred], [hybrid_label], 1)['bleu']
    #         for hybrid_pred, hybrid_label in tqdm(zip(hybrid_preds, hybrid_labels), total=len(hybrid_preds))
    #         if len(hybrid_label)],
    #     'popularity': hybrid_data['popularity'].values,
    # }

    return metrics


def compute_bleu(preds, labels, max_order):
    bleu_output = bleu_metric.compute(
        predictions=preds, references=labels, max_order=max_order
    )

    return bleu_output['bleu']


def compute_rouge(preds, labels):
    rouge_output = rouge_metric.compute(
        predictions=preds, references=labels,
        rouge_types=['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum']
    )

    return rouge_output['rougeL'][1][2]


def compute_bertscore(preds, labels):
    bertscore_output = bertscore.compute(
        predictions=preds, references=labels, lang='en', model_type='bert-base-uncased'
    )

    return bertscore_output

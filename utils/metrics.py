from datasets import load_metric
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json


def evaluate(pred_file, delimiter='~'):
    pred_data = pd.read_csv(pred_file, names=['context', 'label', 'pred'], delimiter=delimiter)
    pred_data.dropna(inplace=True)
    references = pred_data['label'].values
    predictions = pred_data['pred'].values

    references = [[word_tokenize(ref)] for ref in references]
    predictions = [word_tokenize(pred) for pred in predictions]

    bleu1 = compute_bleu(predictions, references, 1)
    bleu2 = compute_bleu(predictions, references, 2)
    bleu3 = compute_bleu(predictions, references, 3)
    bleu4 = compute_bleu(predictions, references, 4)
    bleu5 = compute_bleu(predictions, references, 5)
    rouge = compute_rouge(predictions, references)
    accuracy = compute_accuracy(predictions, references)
    bertscore = compute_bertscore(predictions, references)
    print(bleu1)
    print(bleu2)
    print(bleu3)
    print(bleu4)
    print(bleu5)
    print({'rouge1': rouge['rouge1']})
    print({'rouge2': rouge['rouge2']})
    print({'rouge3': rouge['rouge3']})
    print({'rouge4': rouge['rouge4']})
    print({'rougeL': rouge['rougeL']})
    print({'rougeLsum': rouge['rougeLsum']})
    print(accuracy)
    print(bertscore)


def compute_metrics(eval_preds):
    bleu_metric = load_metric('bleu')
    rouge_metric = load_metric('rouge')

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    bleu_output = bleu_metric.compute(
        predictions=predictions, references=labels, max_order=4
    )
    rouge_output = rouge_metric.compute(
        predictions=predictions, references=labels,
        rouge_types=['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum']
    )

    return {
        'bleu': bleu_output['score'],
        'rouge1': rouge_output['rouge1'][0]['fmeasure'],
        'rouge2': rouge_output['rouge2'][0]['fmeasure'],
        'rouge3': rouge_output['rouge3'][0]['fmeasure'],
        'rouge4': rouge_output['rouge4'][0]['fmeasure'],
        'rougeL': rouge_output['rougeL'][0]['fmeasure'],
        'rougeLsum': rouge_output['rougeLsum'][0]['fmeasure']
    }


def compute_bleu(preds, labels, max_order):
    bleu_metric = load_metric('bleu')
    bleu_output = bleu_metric.compute(
        predictions=preds, references=labels, max_order=max_order
    )

    return bleu_output


def compute_rouge(preds, labels):
    rouge_metric = load_metric('rouge')
    rouge_output = rouge_metric.compute(
        predictions=preds, references=labels,
        rouge_types=['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum']
    )

    return {
        'rouge1': rouge_output['rouge1'],
        'rouge2': rouge_output['rouge2'],
        'rouge3': rouge_output['rouge3'],
        'rouge4': rouge_output['rouge4'],
        'rougeL': rouge_output['rougeL'],
        'rougeLsum': rouge_output['rougeLsum']
    }


def compute_bertscore(preds, labels):
    bertscore = load_metric('bertscore')
    bertscore_output = bertscore.compute(
        predictions=preds, references=labels, lang='en', model_type='bert-base-uncased'
    )

    return {
        'precision': np.mean(bertscore_output['precision']),
        'recall': np.mean(bertscore_output['recall']),
        'f1': np.mean(bertscore_output['f1'])
    }


def compute_accuracy(preds, labels):
    correct, whole = 0, 0
    for pred, label in zip(preds, labels):
        if pred == label[0]:
            correct += 1
        whole += 1

    return {
        'accuracy': correct / whole
    }


def list_lowest_bertscores(file_path, delimiter='~'):
    data = pd.read_csv(file_path, names=['context', 'label', 'pred'], delimiter=delimiter)
    contexts = data['context'].values
    preds = data['pred'].values
    labels = data['label'].values

    bertscore = load_metric('bertscore')
    bertscore_output = bertscore.compute(
        predictions=preds, references=labels, lang='en', model_type='bert-base-uncased'
    )['f1']
    bertscore_dict = {}
    for context, pred, label, bert in tqdm(zip(contexts, preds, labels, bertscore_output)):
        if pred != label[0]:
            bertscore_dict[(context, pred, label)] = bert
    bertscores = sorted(bertscore_dict.items(), key=lambda x: x[1])[:100]

    result = {}
    for item in bertscores:
        (context, pred, label), score = item
        result[label] = {'context': context, 'prediction': pred, 'bertscore': score}

    with open('worst_outputs.json', 'w+') as f:
        json.dump(result, f)


def compare_lowest_bertscores(file_path1, file_path2, num_of_samples=100, delimiter='~'):
    
    data_we = pd.read_csv(file_path1, names=['context', 'label', 'pred'], delimiter=delimiter)
    data_woe = pd.read_csv(file_path2, names=['context', 'label', 'pred'], delimiter=delimiter)
    preds_we = data_we['pred'].values
    preds_woe = data_woe['pred']
    labels = data_we['label'].values

    bertscore = load_metric('bertscore')
    bertscore_output_we = bertscore.compute(
        predictions=preds_we, references=labels, lang='en', model_type='bert-base-uncased'
    )['f1']
    bertscore_output_woe = bertscore.compute(
        predictions=preds_woe, references=labels, lang='en', model_type='bert-base-uncased'
    )['f1']
    bertscore_dict = {}
    for pred_we, pred_woe, label, bert_we, bert_woe in \
      tqdm(zip(preds_we, preds_woe, labels, bertscore_output_we, bertscore_output_woe)):
        if pred_we != label[0]:
            bertscore_dict[(pred_we, pred_woe, label, bert_woe)] = bert_we
    bertscores = sorted(bertscore_dict.items(), key=lambda x: x[1])[:num_of_samples]

    result = {}
    for item in bertscores:
        (pred_we, pred_woe, label, bert_woe), bert_we = item
        result[label] = {'pred_we': pred_we, 'pred_woe': pred_woe, 'bertscore_we': bert_we, 'bertscore_woe': bert_woe}

    with open('worst_outputs.json', 'w+') as f:
        json.dump(result, f)

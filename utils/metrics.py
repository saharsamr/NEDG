from sklearn.metrics import accuracy_score, average_precision_score, \
f1_score, precision_score, recall_score, roc_auc_score
from datasets import load_metric
import numpy as np
import pandas as pd
from scipy import stats
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json


def evaluate_generation(pred_file, delimiter='\1'):
    pred_data = pd.read_csv(pred_file, names=['context', 'label', 'pred'], header=None, delimiter=delimiter)
    print('number of test sample before dropping nan values: ', len(pred_data))
    pred_data.dropna(inplace=True)
    print('number of test sample after dropping nan values: ', len(pred_data))
    references = pred_data['label'].values
    predictions = pred_data['pred'].values

    references = [[word_tokenize(ref)] for ref in references]
    predictions = [word_tokenize(pred) for pred in predictions]

    compute_generation_metrics(predictions, references)


def compute_generation_metrics(predictions, references):

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


def evaluate_classification(test_df):

    print('accuracy: ', accuracy_score(test_df['class-label'], test_df['class-pred']))
    print('avg precision score: ', average_precision_score(test_df['class-label'], test_df['class-pred']))
    print('f1 score: ', f1_score(test_df['class-label'], test_df['class-pred']))
    print('precision_score: ', precision_score(test_df['class-label'], test_df['class-pred']))
    print('recall_score: ', recall_score(test_df['class-label'], test_df['class-pred']))
    print('roc auc score: ', roc_auc_score(test_df['class-label'], test_df['class-pred']))

    references = [[word_tokenize(ref)] for ref in test_df['label']]
    predictions_hybrid = [
        word_tokenize(cpe_pred) if int(class_pred) == 1 else word_tokenize(cme_pred)
        for cpe_pred, cme_pred, class_pred in zip(test_df['CPE-pred'], test_df['CME-pred'], test_df['class-pred'])
    ]
    predictions_CPE = [word_tokenize(pred) for pred in test_df['CPE-pred']]
    predictions_CME = [word_tokenize(pred) for pred in test_df['CME-pred']]

    print('Hybrid Model Evaluation:')
    compute_generation_metrics(predictions_hybrid, references)
    print('Context With Entity Evaluation:')
    compute_generation_metrics(predictions_CPE, references)
    print('Context With Masked Entity Evaluation:')
    compute_generation_metrics(predictions_CME, references)


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


def compute_bertscores_from_file(file_path, delimiter='~'):

    data = pd.read_csv(file_path, names=['context', 'label', 'pred'], header=None, delimiter=delimiter)
    contexts = data['context'].values
    preds = data['pred'].values
    labels = data['label'].values

    bertscore = load_metric('bertscore')
    bertscore_output = bertscore.compute(
        predictions=preds, references=labels, lang='en', model_type='bert-base-uncased'
    )['f1']

    return contexts, preds, labels, bertscore_output


def compute_accuracy(preds, labels):
    correct, whole = 0, 0
    for pred, label in zip(preds, labels):
        if pred == label[0]:
            correct += 1
        whole += 1

    return {
        'accuracy': correct / whole
    }


# TODO: check this one (the delimiter has changed
def list_lowest_bertscores(file_path, delimiter='\1'):

    contexts, preds, labels, bertscore_output = compute_bertscores_from_file(file_path, delimiter=delimiter)
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
    
    _, preds_we, labels_we, bertscore_output_we = compute_bertscores_from_file(file_path1, delimiter=delimiter)
    _, preds_woe, labels_woe, bertscore_output_woe = compute_bertscores_from_file(file_path2, delimiter=delimiter)
    labels = labels_we

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

    with open('worst_outputs_comparison.json', 'w+') as f:
        json.dump(result, f)


def compare_bertscore_means(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)

    bert_score_we, bert_score_woe = [], []
    for key, value in data.items():
        if value['bertscore_we'] and value['bertscore_woe']:
            bert_score_we.append(value['bertscore_we'])
            bert_score_woe.append(value['bertscore_woe'])

    print(
        f'mean of bertscores in with-entity situation: {np.mean(bert_score_we)} '
        f'and in without entity situation:{np.mean(bert_score_woe)}\n'
        f'and the t-test results:\n'
        f'{stats.ttest_ind(bert_score_we, bert_score_woe)}'
    )

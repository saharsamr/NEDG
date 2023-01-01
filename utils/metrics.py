from datasets import load_metric
import numpy as np


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
        rouge_types=['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum', 'rougeSU4']
    )
    
    return {
        'bleu': bleu_output['score'],
        'rouge1': rouge_output['rouge1'][0]['fmeasure'],
        'rouge2': rouge_output['rouge2'][0]['fmeasure'],
        'rouge3': rouge_output['rouge3'][0]['fmeasure'],
        'rouge4': rouge_output['rouge4'][0]['fmeasure'],
        'rougeL': rouge_output['rougeL'][0]['fmeasure'],
        'rougeLsum': rouge_output['rougeLsum'][0]['fmeasure'],
        'rougeSU4': rouge_output['rougeSU4'][0]['fmeasure']
    }

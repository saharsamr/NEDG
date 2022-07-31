from datasets import load_metric
import numpy as np


def rouge(eval_preds):

  rouge = load_metric('rouge')
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  rouge_output = rouge.compute(
    predictions=predictions, references=labels, rouge_types=['rouge2']
  )['rouge2'].mid

  return rouge_output


def bleu(eval_preds):

  bleu = load_metric('bleu')
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  bleu_output = bleu.compute(
    predictions=predictions, references=labels, bleu_types=['bleu4']
  )['bleu4'].mid

  return bleu_output




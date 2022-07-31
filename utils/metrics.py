from datasets import load_metric


def rouge(predictions, labels):

  rouge = load_metric('rouge')
  rouge_output = rouge.compute
  (
    predictions=predictions, references=labels, rouge_types=['rouge2']
  )['rouge2'].mid

  return rouge_output


def bleu(predictions, labels):

  bleu = load_metric('bleu')
  bleu_output = bleu.compute(
    predictions=predictions, references=labels, bleu_types=['bleu4']
  )['bleu4'].mid

  return bleu_output




from transformers import TrainingArguments
from utils.metrics import rouge, bleu
from models.BART import BART
import pandas as pd


if __name__ == "__main__":

  data = pd.read_csv(
    'data/data.csv', delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
  )

  training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
  )

  def compute_metrics(predictions, labels):
    return {
      'rouge': rouge(predictions, labels),
      'bleu': bleu(predictions, labels)
    }

  print('Initialing the model...')
  model = BART(data, training_args, compute_metrics)
  model.set_learnable_params()
  print('Start training...')
  model.train()


from transformers import TrainingArguments
from utils.metrics import rouge, bleu
from models.BART import BART
import pandas as pd


if __name__ == "__main__":

  data = pd.read_csv(
    'NED/data/data.csv', delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
  )

  training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    evaluation_strategy='steps',
    eval_steps=10,
    metric_for_best_model='f1'
  )

  print('Initialing the model...')
  model = BART(data, training_args)
  model.set_learnable_params()
  print('Start training...')
  model.train()
  print('Start prediction...')
  model.pred()


from transformers import TrainingArguments
from utils.metrics import rouge, bleu
from utils.save_data import save_predictions
from models.BART import BART
import pandas as pd


if __name__ == "__main__":

  train = pd.read_csv('data/train.csv', delimiter='\1', header=0, names=['title', 'context', 'description'])
  test = pd.read_csv('data/test.csv', delimiter='\1', header=0, names=['title', 'context', 'description'])
  valid = pd.read_csv('data/valid.csv', delimiter='\1', header=0, names=['title', 'context', 'description'])

  train_x, train_y = train['context'], train['description']
  test_x, test_y = test['context'], test['description']
  valid_x, valid_y = valid['context'], valid['description']

  training_args = TrainingArguments(
    num_train_epochs=150,
    output_dir='./results',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy='epoch',
    logging_steps=1,
    # load_best_model_at_end=True,
    evaluation_strategy='epoch',
    do_eval=True,
    eval_steps=1,
    # save_strategy='epoch'
  )

  print('Initialing the model...')
  model = BART(
    training_args,
    train_x, train_y,
    test_x, test_y,
    valid_x, valid_y,
  )
  model.set_learnable_params()
  print('Start training...')
  model.train()
  print('Start prediction...')
  preds, inputs, labels = model.pred()

  save_predictions(inputs, labels, preds)




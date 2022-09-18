from transformers import TrainingArguments
from utils.save_data import save_predictions
from models.BART import BART
import pandas as pd
from config import TRAIN_FILE, TEST_FILE, VALID_FILE, \
  EPOCHS, BATCH_SIZE, WARMUP_STEPS, WEIGHT_DECAY, \
  LOGGING_DIR, OUTPUT_DIR


if __name__ == "__main__":

  train = pd.read_csv(TRAIN_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])
  test = pd.read_csv(TEST_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])
  valid = pd.read_csv(VALID_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])

  train_x, train_y = list(train['context']), list(train['description'])
  test_x, test_y = list(test['context']), list(test['description'])
  valid_x, valid_y = list(valid['context']), list(valid['description'])

  training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOGGING_DIR,
    logging_strategy='epoch',
    logging_steps=1,
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    do_eval=True,
    eval_steps=1,
    save_strategy='epoch',
    save_total_limit=3
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




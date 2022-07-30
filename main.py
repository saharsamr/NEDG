from transformers import TrainingArguments
from data_handler.dataset import WikiDataset
from utils.metrics import rouge, bleu
from models.BART import BART

if __name__ == "__main__":

  dataset = WikiDataset('./data/data.csv')
  train_dataset, dev_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

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

  model = BART(train_dataset, dev_dataset, test_dataset, training_args, compute_metrics)
  model.set_learnable_params()
  model.train()


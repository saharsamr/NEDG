from transformers import Trainer
from transformers import BertTokenizer, BertForSequenceClassification, \
    RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_metric
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

from GNED.data_handler.classification_dataset import ClassificationDataset
from GNED.config import CLASSIFICATION_SPECIAL_TOKENS, MODEL_CLASSIFICATION_PATH, LEARNING_RATE, \
    TEST_CLASSIFICATION_BATCH_SIZE, MODEL_CLASSIFICATION_NAME, INPUT_CLASSIFICATION_MAX_LENGTH


def compute_metrics(eval_preds):
    metric = load_metric('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class SequenceClassifier:

    def __init__(
      self, training_args,
      train_x, train_y,
      test_x, test_y,
      valid_x, valid_y,
      load=False
    ):

        if 'bert' in MODEL_CLASSIFICATION_NAME:
            self.tokenizer = BertTokenizer.from_pretrained(
                MODEL_CLASSIFICATION_NAME, problem_type='binary_classification',
                max_lenght=INPUT_CLASSIFICATION_MAX_LENGTH)
        elif 'roberta' in MODEL_CLASSIFICATION_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                MODEL_CLASSIFICATION_NAME, problem_type='binary_classification',
                max_lenght=INPUT_CLASSIFICATION_MAX_LENGTH)
        self.tokenizer.add_special_tokens({'additional_special_tokens': CLASSIFICATION_SPECIAL_TOKENS})

        if not load:
            if 'bert' in MODEL_CLASSIFICATION_NAME:
                self.model = BertForSequenceClassification.from_pretrained(MODEL_CLASSIFICATION_NAME, num_labels=2)
            elif 'roberta' in MODEL_CLASSIFICATION_NAME:
                self.model = RobertaForSequenceClassification.from_pretrained(MODEL_CLASSIFICATION_NAME, num_labels=2)
        else:
            if 'bert' in MODEL_CLASSIFICATION_NAME:
                self.model = BertForSequenceClassification.from_pretrained(MODEL_CLASSIFICATION_PATH, num_labels=2)
            elif 'roberta' in MODEL_CLASSIFICATION_NAME:
                self.model = RobertaForSequenceClassification.from_pretrained(MODEL_CLASSIFICATION_PATH, num_labels=2)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.model_max_length = INPUT_CLASSIFICATION_MAX_LENGTH
        self.model.cuda()

        print('Making datasets')
        self.train_dataset = ClassificationDataset(self.tokenizer, train_x, train_y)
        self.test_dataset = ClassificationDataset(self.tokenizer, test_x, test_y)
        self.valid_dataset = ClassificationDataset(self.tokenizer, valid_x, valid_y)

        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        accuracy = load_metric('accuracy')
        self.trainer = Trainer(
            model=self.model, args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            optimizers=(self.optimizer, None),
            compute_metrics=compute_metrics
        )

    def set_learnable_params(self, freeze_encoder=True):

        if 'bert' in MODEL_CLASSIFICATION_NAME:
            for param in self.model.bert.parameters():
                param.requires_grad = not freeze_encoder
            for name, param in self.model.bert.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = True
        elif 'roberta' in MODEL_CLASSIFICATION_NAME:
            for param in self.model.roberta.parameters():
                param.requires_grad = not freeze_encoder
            for name, param in self.model.roberta.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = True

    def train(self):
        self.trainer.train()

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_CLASSIFICATION_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch_preds = self.model(batch['input_ids'].cuda()).logits
                batch_labels = batch['labels']
                batch_input_ = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                for pred, label, input_ in zip(batch_preds, batch_labels, batch_input_):
                    predictions.extend([pred.argmax().cpu().numpy()])
                    inputs.extend([input_])
                    labels.extend([label])

        return predictions, inputs, labels

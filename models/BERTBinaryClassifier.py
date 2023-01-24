from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from data_handler.classification_dataset import ClassificationDataset
from config import ADDITIONAL_SPECIAL_TOKENS, MODEL_PATH, LEARNING_RATE, \
    TEST_BATCH_SIZE
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class BERTBinaryClassification:

    def __init__(
      self, training_args,
      train_x, train_y,
      test_x, test_y,
      valid_x, valid_y,
      load=False
    ):

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        if load:
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Making datasets')
        self.train_dataset = ClassificationDataset(self.tokenizer, train_x, train_y)
        self.test_dataset = ClassificationDataset(self.tokenizer, test_x, test_y)
        self.valid_dataset = ClassificationDataset(self.tokenizer, valid_x, valid_y)

        self.optimizer = AdamW(self.model.get_decoder().parameters(), lr=LEARNING_RATE)

        self.trainer = Trainer(
            model=self.model, args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            optimizers=(self.optimizer, None)
        )

    def train(self):
        self.model.train()

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                preds = self.model.generate(batch['input_ids'].cuda())
                predictions.extend(preds)
                input_ = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                inputs.extend(input_)
                label = batch['label']
                labels.extend(label)
        return predictions, inputs, labels

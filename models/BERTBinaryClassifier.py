from transformers import Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from data_handler.classification_dataset import ClassificationDataset
from config import ADDITIONAL_SPECIAL_TOKENS, MODEL_CLASSIFICATION_PATH, LEARNING_RATE, \
    TEST_CLASSIFICATION_BATCH_SIZE
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class BERTBinaryClassification:

    def __init__(
      self, training_args,
      train_x, train_y,
      test_x, test_y,
      load=False
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', problem_type='binary_classification', max_lenght=512, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        if not load:
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        else:
            self.model = BertForSequenceClassification.from_pretrained(MODEL_CLASSIFICATION_PATH, num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.model_max_length = 512
        self.model.cuda()

        print('Making datasets')
        self.train_dataset = ClassificationDataset(self.tokenizer, train_x, train_y)
        self.test_dataset = ClassificationDataset(self.tokenizer, test_x, test_y)

        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        self.trainer = Trainer(
            model=self.model, args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            optimizers=(self.optimizer, None)
        )

    def set_learnable_params(self, freeze_encoder=True):

        for param in self.model.bert.parameters():
            param.requires_grad = not freeze_encoder

    def train(self):
        self.trainer.train()

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_CLASSIFICATION_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                preds = self.model(batch['input_ids'].cuda())
                predictions.extend([preds.logits[0].argmax().cpu().numpy()])
                input_ = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                inputs.extend(input_)
                label = batch['labels']
                labels.extend([label[0].argmax().cpu().numpy()])
        return predictions, inputs, labels

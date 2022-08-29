from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig, TrainingArguments, Trainer
from transformers.optimization import AdamW
from transformers import Trainer
from data_handler.dataset import create_train_dev_test_datasets
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
import torch


class BART:

    def __init__(
      self, data, trainer_args, model_name='facebook/bart-large-cnn'
    ):

        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<NE>', '</NE>']})
        self.config = BartConfig.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model_name = model_name

        print('Making datasets')
        self.data = data
        self.train_dataset, self.dev_dataset, self.test_dataset = create_train_dev_test_datasets(
            self.data, self.tokenizer, self.config.max_length
        )

        self.optimizer = AdamW(self.model.parameters())

        self.trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            # optimizers=[self.optimizer],
            # tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics_func,
        )

    def train(self):
        self.trainer.train()

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=False):

        for param in self.model.get_encoder().parameters():
            param.requires_grad = not freeze_encoder
        for param in self.model.get_decoder().parameters():
            param.requires_grad = not freeze_decoder

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        inputs, labels, predictions = [], [], []
        with torch.no_grad():
            for batch in test_dataloader:
                ids = self.model.generate(batch['input_ids'].cuda(), max_length=256)
                preds = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                predictions.extend(preds)
                input = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                inputs.extend(input)
                label = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels.extend(label)
        return predictions, inputs, labels

    def save(self):
        pass

    def load(self):
        pass

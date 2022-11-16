from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig, TrainingArguments, Trainer
from transformers.optimization import AdamW
from transformers import Trainer
from data_handler.dataset import WikiDataset
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
import torch
from config import MODEL_NAME, ADDITIONAL_SPECIAL_TOKENS


class BART:

    def __init__(
      self, trainer_args,
      train_x, train_y,
      test_x, test_y,
      valid_x, valid_y,
      model_name=MODEL_NAME
    ):

        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model_name = model_name

        print('Making datasets')
        self.train_dataset = WikiDataset(self.tokenizer, train_x, train_y)
        self.test_dataset = WikiDataset(self.tokenizer, test_x, test_y)
        self.valid_dataset = WikiDataset(self.tokenizer, valid_x, valid_y)

        self.optimizer = AdamW(self.model.parameters())

        self.trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            # optimizers=[self.optimizer],
            # tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics_func,
        )

    def train(self):
        self.trainer.train()

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=True):

        for part in [self.model.encoder, self.model.decoder]:
            for param in part.embed_positions.parameters():
                param.requires_grad = False
            for param in part.embed_tokens.parameters():
                param.requires_grad = False

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

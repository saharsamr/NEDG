from transformers import GPT2TokenizerFast, GPT2Model
from transformers.optimization import AdamW
from transformers import EarlyStoppingCallback
from transformers import Trainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from GNED.data_handler.wiki_dataset import WikiDataset
from GNED.config import *


class GPT2:

    def __init__(
      self, trainer_args,
      train_x, train_y,
      test_x, test_y,
      valid_x, valid_y,
      train_entity_names,
      test_entity_names,
      valid_entity_names,
      model_name=MODEL_GENERATION_NAME,
      model_load_path=MODEL_GENERATION_PATH,
      load=False
    ):

        self.model_name = model_name
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            self.model_name, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=False, truncation=True,
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        if load:
            self.model = GPT2Model.from_pretrained(model_load_path)
        else:
            self.model = GPT2Model.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Making datasets')
        self.train_dataset = WikiDataset(self.tokenizer, train_x, train_y, train_entity_names, mask_entity=True)
        self.test_dataset = WikiDataset(self.tokenizer, test_x, test_y, test_entity_names, mask_entity=False)
        self.valid_dataset = WikiDataset(self.tokenizer, valid_x, valid_y, valid_entity_names, mask_entity=False)

        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        self.trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer, None),
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(f'{OUTPUT_DIR}/final_model/')

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=True):

        for param in self.model.parameters():
            param.requires_grad = True

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_GENERATION_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions, entity_names = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                ids = self.model.generate(
                    batch['input_ids'].cuda(), min_length=OUTPUT_GENERATION_MIN_LENGTH
                    , max_length=OUTPUT_GENERATION_MAX_LENGTH
                )
                preds = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                predictions.extend(preds)
                # input_ = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
                inputs.extend(batch['input_text'])
                label = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels.extend(label)
                entity_names.extend(batch['entity_name'])
        return predictions, inputs, labels, entity_names

    def save(self):
        pass

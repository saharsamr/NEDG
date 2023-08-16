from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers.optimization import AdamW
from transformers import EarlyStoppingCallback
from transformers import Trainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from GNED.data_handler.wiki_dataset import WikiDataset
from GNED.utils.metrics import compute_metrics
from GNED.config import MODEL_GENERATION_NAME, ADDITIONAL_SPECIAL_TOKENS, \
    MODEL_GENERATION_PATH, OUTPUT_GENERATION_MAX_LENGTH, LEARNING_RATE, INPUT_GENERATION_MAX_LENGTH, \
    OUTPUT_GENERATION_MIN_LENGTH, TEST_GENERATION_BATCH_SIZE


class BART:

    def __init__(
      self, trainer_args,
      train_x, train_y,
      test_x, test_y,
      valid_x, valid_y,
      model_name=MODEL_GENERATION_NAME,
      model_load_path=MODEL_GENERATION_PATH,
      load=False
    ):

        self.model_name = model_name
        self.tokenizer = BartTokenizerFast.from_pretrained(
            self.model_name, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        if load:
            self.model = BartForConditionalGeneration.from_pretrained(model_load_path)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Making datasets')
        self.train_dataset = WikiDataset(self.tokenizer, train_x, train_y, mask_entity=True)
        self.test_dataset = WikiDataset(self.tokenizer, test_x, test_y, mask_entity=False)
        self.valid_dataset = WikiDataset(self.tokenizer, valid_x, valid_y, mask_entity=False)

        self.optimizer = AdamW(self.model.get_decoder().parameters(), lr=LEARNING_RATE)

        self.trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer, None),
            # compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def train(self):
        self.trainer.train()

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=True):

        for param in self.model.get_encoder().parameters():
            param.requires_grad = not freeze_encoder
        for param in self.model.get_decoder().parameters():
            param.requires_grad = not freeze_decoder

        for part in [self.model.get_encoder()]:
            for param in part.embed_positions.parameters():
                param.requires_grad = False
            for param in part.embed_tokens.parameters():
                param.requires_grad = True

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_GENERATION_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions = [], [], []
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
        return predictions, inputs, labels

    def save(self):
        pass

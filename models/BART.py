import numpy as np
from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers.optimization import AdamW
from data_handler.wiki_dataset import WikiDataset
from utils.metrics import compute_metrics
from transformers import EarlyStoppingCallback
from transformers import Trainer
from torch.utils.data import DataLoader
import torch
from config import MODEL_GENERATION_NAME, ADDITIONAL_SPECIAL_TOKENS, \
    MODEL_GENERATION_PATH, MODEL_GENERATION_PATH_2, OUTPUT_GENERATION_MAX_LENGTH, LEARNING_RATE, INPUT_GENERATION_MAX_LENGTH, \
    OUTPUT_GENERATION_MIN_LENGTH, TEST_GENERATION_BATCH_SIZE, MODEL_GENERATION_NAME_2
from tqdm import tqdm


class BART:

    def __init__(
            self, trainer_args,
            train_x, train_y,
            test_x, test_y,
            valid_x, valid_y,
            model_name=MODEL_GENERATION_NAME,
            load=False,
            model_name_2=MODEL_GENERATION_NAME_2,
            load_2=False,
    ):

        self.model_name_1 = model_name
        self.model_name_2 = model_name_2
        self.tokenizer = BartTokenizerFast.from_pretrained(
            self.model_name_1, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        if load:
            self.model_1 = BartForConditionalGeneration.from_pretrained(MODEL_GENERATION_PATH)
        else:
            self.model_1 = BartForConditionalGeneration.from_pretrained(self.model_name_1)
        self.model_1.resize_token_embeddings(len(self.tokenizer))

        if load_2:  # initialize second model
            self.model_2 = BartForConditionalGeneration.from_pretrained(MODEL_GENERATION_PATH_2)
        else:
            self.model_2 = BartForConditionalGeneration.from_pretrained(self.model_name_2)
        self.model_2.resize_token_embeddings(len(self.tokenizer))

        print('Making datasets')
        self.train_dataset = WikiDataset(self.tokenizer, train_x, train_y)
        self.test_dataset = WikiDataset(self.tokenizer, test_x, test_y)
        self.valid_dataset = WikiDataset(self.tokenizer, valid_x, valid_y)

        self.optimizer_1 = AdamW(self.model_1.get_decoder().parameters(), lr=LEARNING_RATE)
        self.optimizer_2 = AdamW(self.model_2.get_decoder().parameters(), lr=LEARNING_RATE)
        self.trainer = Trainer(
            model=self.model_1,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer_1, None),
            # compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        self.trainer_2 = Trainer(  # new trainer for second model
            model=self.model_2,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer_2, None),
        )

    def forward(self, inputs):
        with torch.no_grad():
            outputs1 = self.model1(inputs)
            outputs2 = self.model2(inputs)
        return {'outputs1': outputs1, 'outputs2': outputs2}

    def train(self):
        self.trainer.train()

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=True):

        for part in [self.model_1.get_encoder(), self.model_1.get_decoder()]:
            for param in part.embed_positions.parameters():
                param.requires_grad = False
            for param in part.embed_tokens.parameters():
                param.requires_grad = False

        for param in self.model_1.get_encoder().parameters():
            param.requires_grad = not freeze_encoder
        for param in self.model_1.get_decoder().parameters():
            param.requires_grad = not freeze_decoder

    def pred(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=TEST_GENERATION_BATCH_SIZE, shuffle=False)
        inputs, labels, predictions = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                ids = self.model_1.generate(
                    batch['input_ids'].cuda(), min_length=OUTPUT_GENERATION_MIN_LENGTH,
                    max_length=OUTPUT_GENERATION_MAX_LENGTH
                )
                preds = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                predictions.extend(preds)
                input_ = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                inputs.extend(input_)
                label = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels.extend(label)
        return predictions, inputs, labels
    def attention_weights_specific_layer(self, input_text: str, i, **kwargs):
        with torch.no_grad():
            encoded_input = self.tokenizer.encode_plus(
                input_text,
                return_tensors="pt",
                max_length=600,
                padding="longest",
                truncation=True
            )
            if torch.cuda.is_available():
                self.model_1.cuda()
                self.model_2.cuda()
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            outputs_1 = self.model_1(
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions_1 = outputs_1.encoder_attentions
            decoder_attentions_1 = outputs_1.decoder_attentions

            outputs2 = self.model_2(  # generate outputs from second model
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions_2 = outputs2.encoder_attentions
            decoder_attentions_2 = outputs2.decoder_attentions

            return encoder_attentions_1[i], decoder_attentions_1[i], encoder_attentions_2[i], decoder_attentions_2[i]

    def attention_weights(self, input_text: str, **kwargs):
        with torch.no_grad():
            encoded_input = self.tokenizer.encode_plus(
                input_text,
                return_tensors="pt",
                max_length=600,
                padding="longest",
                truncation=True
            )
            if torch.cuda.is_available():
                self.model_1.cuda()
                self.model_2.cuda()
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            outputs_1 = self.model_1(
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions_1 = outputs_1.encoder_attentions
            decoder_attentions_1 = outputs_1.decoder_attentions

            outputs2 = self.model_2(  # generate outputs from second model
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions_2 = outputs2.encoder_attentions
            decoder_attentions_2 = outputs2.decoder_attentions

            return encoder_attentions_1, decoder_attentions_1, encoder_attentions_2, decoder_attentions_2

    def Pearson_correlation(self, compareModel):
        # Get attention weights for each model
        encoder_att1, decoder_att1 = self.model_2.attention_weights(self.model_1.attention_weights, return_dict=True)
        encoder_att2, decoder_att2 = self.model_1.attention_weights(self.model_2.attention_weights, return_dict=True)

        # Flatten attention weights into 1D arrays
        att1_flat = np.ravel(encoder_att1.numpy())
        att2_flat = np.ravel(encoder_att2.numpy())

        # Compute Pearson correlation coefficient
        corr_coef, p_value = np.corrcoef(att1_flat, att2_flat)

        return corr_coef
    def save(self):
        pass

# in metric lists:
#  'pearson': PearsonCorrelation('pearson', 'outputs1', 'outputs2')(model_output)

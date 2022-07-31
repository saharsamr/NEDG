from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, TrainingArguments, Trainer
from transformers.optimization import AdamW
from utils.trainers import CrossEntropyTrainer


class BART:

    def __init__(
      self, train_dataset, dev_dataset, test_dataset,
      trainer_args, compute_metrics_func, model_name='facebook/bart-large'
    ):

        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<NE>', '</NE>']})
        self.config = BartConfig.from_pretrained(model_name)
        self.model_name = model_name

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.optimizer = AdamW(self.model.parameters())

        self.trainer = CrossEntropyTrainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            optimizers=[self.optimizer],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_func,
        )

    def train(self):
        self.trainer.train()

    def set_learnable_params(self, freeze_encoder=True, freeze_decoder=False):

        for param in self.model.get_encoder().parameters():
            param.requires_grad = not freeze_encoder
        for param in self.model.get_decoder().parameters():
            param.requires_grad = not freeze_decoder

    def pred(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

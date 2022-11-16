from transformers import Trainer
from torch import nn


# TODO: it takes just two arguments and lables is extra, also, its model calls is totally false
class CrossEntropyTrainer(Trainer):

    def compute_loss(self, model, inputs, labels, return_outputs=False):

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

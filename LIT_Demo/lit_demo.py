from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp import dev_server
from lit_nlp import server_flags

import sys
import tempfile
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


class IMDBData(lit_dataset.Dataset):

    LABELS = ["0", "1"]

    def __init__(self):
        imdb = load_dataset("imdb")
        dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

        self._examples = [{
            "text": sample['text'],
            "label": sample['label']
        } for sample in dataset]

    def spec(self) -> lit_types.Spec:
        return {
            "text": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS)
        }


class NewsClassificationModel(lit_model.Model):

    LABELS = ["0", "1"]

    def __init__(self, model_path):

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    @staticmethod
    def max_minibatch_size():
        return 32

    def predict_minibatch(self, inputs):

        encoded_input = self.tokenizer.batch_encode_plus(
            [ex["TITLE"] for ex in inputs],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=128,
            padding="longest",
            truncation="longest_first"
        )

        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                encoded_input[tensor] = encoded_input[tensor].cuda()

        with torch.no_grad():  # remove this if you need gradients.
            out: transformers.modeling_outputs.SequenceClassifierOutput = \
                self.model(**encoded_input)

        batched_outputs = {
            "probas": torch.nn.functional.softmax(out.logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "ntok": torch.sum(encoded_input["attention_mask"], dim=1),
            "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
        }

        detached_outputs = {k: v.cpu().numpy() for k, v in batched_outputs.items()}

        for output in utils.unbatch_preds(detached_outputs):
            ntok = output.pop("ntok")
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[1:ntok - 1])
            yield output

    def input_spec(self) -> lit_types.Spec:
        return {
            "text": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS, required=False)
        }

    def output_spec(self) -> lit_types.Spec:
        return {
            "tokens": lit_types.Tokens(),
            "probas": lit_types.MulticlassPreds(parent="label", vocab=self.LABELS),
            "cls_emb": lit_types.Embeddings()
        }


def get_wsgi_app() -> Optional[dev_server.LitServerType]:

    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "quickstart_sst_demo:get_wsgi_app() called with unused "
            "args: %s", unused)
    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:

    model_path = _MODEL_PATH.value or tempfile.mkdtemp()
    logging.info("Working directory: %s", model_path)

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    FLAGS.set_default("development_demo", True)
    FLAGS.set_default("server_type", "external")
    FLAGS.set_default("demo_mode", True)
    _MODEL_PATH = flags.DEFINE_string("results/model", None,
                                      "Path to save trained model.")

    datasets = {'news_test': IMDBData()}
    models = {"imdb_classifier": NewsClassificationModel('/content/model/')}

    app.run(main)
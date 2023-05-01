from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp import dev_server
from lit_nlp import server_flags

import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration

import pandas as pd

INPUT_GENERATION_MAX_LENGTH = 600
FLAGS = flags.FLAGS


class WikiDataset(lit_dataset.Dataset):

    def __init__(self, file_path):

        data = pd.read_csv(file_path, delimiter='\1', header=None, names=['title', 'context', 'description'])[:100]
        self._examples = [{
            "title": row['title'],
            "context": row['context'],
            "description": row['description']
        } for _, row in data.iterrows()]

    def spec(self) -> lit_types.Spec:

        return {
            "title": lit_types.TextSegment(),
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment()
        }


class BartModel(lit_model.Model):

    def __init__(self, model_path, model_name):

        super().__init__()
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

        self.enc_layers, self.dec_layers = None, None

    def predict_minibatch(self, inputs):

        encoded_input = self.tokenizer.batch_encode_plus(
            [ex["context"] for ex in inputs],  # changed
            return_tensors="pt",
            max_length=600,
            padding="longest",
            truncation=True
        )
        encoded_output = self.tokenizer.batch_encode_plus(
            [ex["description"] for ex in inputs],  # changed
            return_tensors="pt",
            max_length=20,
            padding="longest",
            truncation=True
        )

        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                encoded_input[tensor] = encoded_input[tensor].cuda()

        with torch.no_grad():  # remove this if you need gradients.
            out = self.model(**encoded_input, output_attentions=True)
            ids = self.model.generate(encoded_input['input_ids'])

        batched_outputs = {
            "probas": torch.nn.functional.softmax(out.logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "input_ntok": torch.sum(encoded_input["attention_mask"], dim=1),
            "target_ids": encoded_output["input_ids"],
            "target_ntok": torch.sum(encoded_output["attention_mask"], dim=1),
            "decoder_layer_1_attention": out.decoder_attentions[0],
            "encoder_layer_1_attention": out.encoder_attentions[0]
        }

        detached_outputs = {k: v.cpu().numpy() if type(v) != list else torch.tensor(v).cpu().numpy() for k, v in batched_outputs.items()}

        detached_outputs["output_text"] = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        for output in utils.unbatch_preds(detached_outputs):
            input_ntok = output.pop("input_ntok")
            output["input_tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[1:input_ntok - 1])
            output_ntok = output.pop("target_ntok")
            output["target_tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("target_ids")[1:output_ntok - 1])
            yield output

    def input_spec(self) -> lit_types.Spec:

        return {
            "title": lit_types.TextSegment(),
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment()
        }

    def output_spec(self) -> lit_types.Spec:

        spec = {
            "input_tokens": lit_types.Tokens(),
            "target_tokens": lit_types.Tokens(),
            "output_text": lit_types.GeneratedText(parent="description"),
            "encoder_layer_1_attention": lit_types.AttentionHeads(align_in="input_tokens", align_out="input_tokens"),
            "decoder_layer_1_attention": lit_types.AttentionHeads(align_in="target_tokens", align_out="target_tokens")
        }

        return spec


def get_wsgi_app() -> Optional[dev_server.LitServerType]:

    FLAGS.set_default("server_type", "default")
    FLAGS.set_default("host", "0.0.0.0")
    FLAGS.set_default("demo_mode", True)

    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "quickstart_sst_demo:get_wsgi_app() called with unused "
            "args: %s", unused)

    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:

    datasets = {'wiki_dataset': WikiDataset('../data/HumanConcatenated/test_human_masked_ne_with_context.csv')}
    models = {"bart": BartModel('../results/ConcatedCME', 'facebook/bart-large-cnn')}

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

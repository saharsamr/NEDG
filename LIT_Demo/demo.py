from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp import dev_server
from lit_nlp import server_flags

import re
import sys
import attr
from typing import Optional, Sequence, List

from absl import app
from absl import flags
from absl import logging

import torch
import tensorflow as tf
from transformers import BartTokenizerFast, BartForConditionalGeneration

import pandas as pd

INPUT_GENERATION_MAX_LENGTH = 600
FLAGS = flags.FLAGS


def masked_token_mean(vectors, masks):

    masks = tf.cast(masks, tf.float32)
    weights = masks / tf.reduce_sum(masks, axis=1, keepdims=True)
    return tf.reduce_sum(vectors * tf.expand_dims(weights, axis=-1), axis=1)


class WikiDataset(lit_dataset.Dataset):

    def __init__(self, file_path):

        data = pd.read_csv(file_path, delimiter='\1', header=None, names=['title', 'context', 'description'])
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


@attr.s(auto_attribs=True, kw_only=True)
class BARTModelConfig(object):

    inference_batch_size: int = 1
    beam_size: int = 4
    max_gen_length: int = 20
    num_to_generate: int = 1
    token_top_k: int = 10
    output_attention: bool = True


class BartModel(lit_model.Model):

    def __init__(self, model_path, model_name, **config_kw):

        super().__init__()
        self.config = BARTModelConfig(**config_kw)
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.cntx_max_len = 600
        self.def_max_len = 20


    def predict_minibatch(self, inputs):

        encoded_context = self.tokenizer.batch_encode_plus(
            [sample['context'] for sample in inputs], padding='max_length', truncation=True, max_length=self.cntx_max_len)
        encoded_def = self.tokenizer.batch_encode_plus(
            [sample['description'] for sample in inputs], padding='max_length', truncation=True, max_length=self.def_max_len)

        self.model.cuda()
        for tensor in encoded_context:
            encoded_context[tensor] = encoded_context[tensor].cuda()
        with torch.no_grad():
            ids = self.model.generate(encoded_context['input_ids'], min_length=3, max_length=self.def_max_len)

        batched_outputs = {
            "probas": torch.nn.functional.softmax(ids.logits, dim=-1),
            "input_ids": encoded_context["input_ids"],
            "ntok": torch.sum(encoded_context["attention_mask"], dim=1),
            "cls_emb": ids.hidden_states[-1][:, 0],  # last layer, first token
        }

        assert len(ids.attentions) == self.model.config.num_hidden_layers
        for i, layer_attention in enumerate(ids.attentions):
            batched_outputs[f"layer_{i}/attention"] = layer_attention

        detached_outputs = {
            k: v.cpu().detach().numpy() for k, v in batched_outputs.items()}

        for output in utils.unbatch_preds(detached_outputs):
            ntok = output.pop("ntok")
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[:ntok])

            for key in output:
                if not re.match(r"layer_(\d+)/attention", key):
                    continue
                output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
                output[key] = output[key].copy()
            yield output

    def input_spec(self):

        return {
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment(required=False),
        }

    def output_spec(self):

        spec = {
            "output_text": lit_types.GeneratedText(parent="description"),
            "input_tokens": lit_types.Tokens(parent="context"),
            "encoder_final_embedding": lit_types.Embeddings(),
            "target_tokens": lit_types.Tokens(parent="description"),
            "pred_tokens": lit_types.TokenTopKPreds(align="target_tokens"),
        }

        for i in range(self.model.config.num_hidden_layers):
            spec[f"layer_{i}/attention"] = lit_types.AttentionHeads(
                align_in="input_tokens", align_out="input_tokens")
        return spec


def get_wsgi_app() -> Optional[dev_server.LitServerType]:

    FLAGS.set_default("server_type", "default")
    FLAGS.set_default("host", "0.0.0.0")
    FLAGS.set_default("demo_mode", True)
    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "quickstart_sst_demo:get_wsgi_app() called with unused "
            "args: %s", unused)
    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:

    datasets = {'news_test': WikiDataset('../data/HumanConcatenated/test_human_ne_no_context.csv')}
    models = {"imdb_classifier":BartModel('../results/ConcatedCME', 'facebook/bart-large-cnn')}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

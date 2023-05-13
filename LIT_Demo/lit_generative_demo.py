import numpy as np
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp import dev_server
from lit_nlp import server_flags

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


class GenerativeModel(lit_model.Model):

    def __init__(self, model_path, model_name, **config_kw):

        super().__init__()
        self.config = BARTModelConfig(**config_kw)
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
        self.model_1 = BartForConditionalGeneration.from_pretrained(model_path)

    @property
    def num_layers(self):
        return self.model_1.config.num_layers

    def _encode_texts(self, texts: List[str]):
        return self.tokenizer.batch_encode_plus(
            texts,
            padding="longest",
            truncation=True)

    # def _force_decode(self, encoded_inputs, encoded_targets):
    #
    #     results = self.model(
    #         input_ids=encoded_inputs["input_ids"],
    #         decoder_input_ids=encoded_targets["input_ids"],
    #         attention_mask=encoded_inputs["attention_mask"],
    #         decoder_attention_mask=encoded_targets["attention_mask"])
    #
    #     model_probs = tf.nn.softmax(results.logits, axis=-1)
    #     top_k = tf.math.top_k(
    #         model_probs, k=self.config.token_top_k, sorted=True, name=None)
    #     batched_outputs = {"input_ids": encoded_inputs["input_ids"],
    #                        "input_ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
    #                        "target_ids": encoded_targets["input_ids"],
    #                        "target_ntok": tf.reduce_sum(encoded_targets["attention_mask"], axis=1),
    #                        "top_k_indices": top_k.indices, "top_k_probs": top_k.values,
    #                        "encoder_final_embedding": masked_token_mean(
    #                            results.encoder_last_hidden_state, encoded_inputs["attention_mask"])}
    #
    #     if self.config.output_attention:
    #         for i in range(len(results.decoder_attentions)):
    #             batched_outputs[
    #                 f"decoder_layer_{i + 1:d}_attention"] = results.decoder_attentions[i]
    #         for i in range(len(results.encoder_attentions)):
    #             batched_outputs[
    #                 f"encoder_layer_{i + 1:d}_attention"] = results.encoder_attentions[i]
    #
    #     return batched_outputs

    def _force_decode(self, encoded_inputs, encoded_targets):

        input_ids = torch.tensor(encoded_inputs["input_ids"]).cuda()
        attention_mask = torch.tensor(encoded_inputs["attention_mask"]).cuda()
        decoder_input_ids = torch.tensor(encoded_targets["input_ids"]).cuda()
        decoder_attention_mask = torch.tensor(encoded_targets["attention_mask"]).cuda()

        self.model_1.cuda()
        results = self.model_1(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask)

        model_probs = torch.nn.functional.softmax(results.logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(
            model_probs, k=self.config.token_top_k, dim=-1, sorted=True)
        batched_outputs = {"input_ids": encoded_inputs["input_ids"],
                           "input_ntok": attention_mask.sum(dim=1),
                           "target_ids": encoded_targets["input_ids"],
                           "target_ntok": decoder_attention_mask.sum(dim=1),
                           "top_k_indices": top_k_indices,
                           "top_k_probs": top_k_values,
                           "encoder_final_embedding": masked_token_mean(
                               results.encoder_last_hidden_state, attention_mask)}

        if self.config.output_attention:
            for i, decoder_attention in enumerate(results.decoder_attentions):
                batched_outputs[f"decoder_layer_{i + 1:d}_attention"] = decoder_attention
            for i, encoder_attention in enumerate(results.encoder_attentions):
                batched_outputs[f"encoder_layer_{i + 1:d}_attention"] = encoder_attention

        return batched_outputs

    def _postprocess(self, preds):

        input_ntok = preds.pop("input_ntok")
        input_ids = preds.pop("input_ids")[:input_ntok]
        preds["input_tokens"] = self.tokenizer.convert_ids_to_tokens(input_ids)

        target_ntok = preds.pop("target_ntok")
        target_ids = preds.pop("target_ids")[:target_ntok]
        preds["target_tokens"] = self.tokenizer.convert_ids_to_tokens(target_ids)

        token_topk_preds = [[("N/A", 1.)]]
        pred_ids = preds.pop("top_k_indices")[:target_ntok]  # <int>[num_tokens, k]
        pred_probs = preds.pop(
            "top_k_probs")[:target_ntok]  # <float32>[num_tokens, k]
        for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
            token_pred_words = self.tokenizer.convert_ids_to_tokens(token_pred_ids)
            token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
        preds["pred_tokens"] = token_topk_preds

        candidates = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in preds.pop("generated_ids")
        ]
        if self.config.num_to_generate > 1:
            preds["output_text"] = [(s, None) for s in candidates]
        else:
            preds["output_text"] = candidates[0]

        for key in preds:
            if not re.match(r"\w+_layer_(\d+)/attention", key):
                continue
            if key.startswith("encoder_"):
                ntok = input_ntok
            elif key.startswith("decoder_"):
                ntok = target_ntok
            else:
                raise ValueError(f"Invalid attention key: '{key}'")

            preds[key] = preds[key][:, :ntok, :ntok].transpose((0, 2, 1))
            preds[key] = preds[key].copy()

        return preds

    def max_minibatch_size(self) -> int:
        return self.config.inference_batch_size

    def predict_minibatch(self, inputs):

        encoded_inputs = self._encode_texts([ex["context"] for ex in inputs])
        encoded_targets = self._encode_texts(
            [ex.get("description", "") for ex in inputs])

        batched_outputs = self._force_decode(encoded_inputs, encoded_targets)
        self.model_1.config.output_hidden_states = False
        generated_ids = self.model_1.generate(
            encoded_inputs.input_ids,
            num_beams=self.config.beam_size,
            attention_mask=encoded_inputs.attention_mask,
            max_length=self.config.max_gen_length,
            num_return_sequences=self.config.num_to_generate)

        batched_outputs["generated_ids"] = tf.reshape(
            generated_ids,
            [-1, self.config.num_to_generate, generated_ids.shape[-1]])
        self.model_1.config.output_hidden_states = True

        detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
        unbatched_outputs = utils.unbatch_preds(detached_outputs)
        return list(map(self._postprocess, unbatched_outputs))

    # def predict_minibatch(self, inputs):
    #
    #     encoded_inputs = self._encode_texts([ex["context"] for ex in inputs])
    #     encoded_targets = self._encode_texts(
    #         [ex.get("description", "") for ex in inputs])
    #
    #     if torch.cuda.is_available():
    #         self.model.cuda()
    #         # for tensor in encoded_inputs:
    #         #     encoded_inputs[tensor] = encoded_inputs[tensor].cuda()
    #
    #     ids = self.model.generate(
    #         encoded_inputs, min_length=3, max_length=20
    #     )
    #     preds = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
    #
    #     batched_outputs = {"input_ids": encoded_inputs["input_ids"],
    #                        "input_ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
    #                        "target_ids": encoded_targets["input_ids"],
    #                        "target_ntok": tf.reduce_sum(encoded_targets["attention_mask"], axis=1)}
    #     detached_outputs = {k: v.cpu().numpy() for k, v in batched_outputs.items()}
    #
    #     for output in utils.unbatch_preds(detached_outputs):
    #         ntok = output.pop("ntok")
    #         output["tokens"] = self.tokenizer.convert_ids_to_tokens(
    #             output.pop("input_ids")[1:ntok - 1])
    #         yield output

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
        if self.config.num_to_generate > 1:
            spec["output_text"] = lit_types.GeneratedTextCandidates(
                parent="description")

        # if self.config.output_attention:
        #     for i in range(self.num_layers):
        #         spec[f"encoder_layer_{i + 1:d}_attention"] = lit_types.AttentionHeads(
        #             align_in="input_tokens", align_out="input_tokens")
        #         spec[f"decoder_layer_{i + 1:d}_attention"] = lit_types.AttentionHeads(
        #             align_in="target_tokens", align_out="target_tokens")
        return spec

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
    models = {"imdb_classifier": GenerativeModel('../results/ConcatedCME', 'facebook/bart-large-cnn')}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

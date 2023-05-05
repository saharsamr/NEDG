from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration


def masked_token_mean(vectors, masks):

    denom = torch.sum(masks, -1, keepdim=True)
    feat = torch.sum(vectors * masks.unsqueeze(-1), dim=1) / denom
    return feat


class BartModel(lit_model.Model):

    def __init__(self, model_path, model_name):

        super().__init__()
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=600, padding=True, truncation=True,
        )
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

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
            "encoder_layer_1_attention": out.encoder_attentions[0],
            "encoder_final_embedding": masked_token_mean(
                out.encoder_last_hidden_state, encoded_input["attention_mask"])
        }

        detached_outputs = {k: v.cpu().numpy() if type(v) != list else torch.tensor(v).cpu().numpy() for k, v in
                            batched_outputs.items()}

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
            "target_text": lit_types.TextSegment(),
            "input_tokens": lit_types.Tokens(),
            "input_gradients": lit_types.TokenGradients(align="input_tokens"),
            "target_tokens": lit_types.Tokens(),
            "output_text": lit_types.GeneratedText(parent="description"),
            "encoder_layer_1_attention": lit_types.AttentionHeads(align_in="input_tokens", align_out="input_tokens"),
            "decoder_layer_1_attention": lit_types.AttentionHeads(align_in="target_tokens", align_out="target_tokens"),
            "encoder_final_embedding": lit_types.Embeddings()
        }

        return spec

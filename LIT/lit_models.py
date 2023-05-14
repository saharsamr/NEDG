import numpy as np
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration


def masked_token_mean(vectors, masks):

    denom = torch.sum(masks, -1, keepdim=True)
    feat = torch.sum(vectors * masks.unsqueeze(-1), dim=1) / denom
    return feat


def masking_entities(tokenizer, input_ids_list, attention_mask_list):

    new_input_ids_list, new_attention_mask_list = [], []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):

        new_input_ids = input_ids.tolist()
        new_attention_mask = attention_mask.tolist()

        entity_start_token_id = tokenizer.convert_tokens_to_ids('<NE>')
        entity_end_token_id = tokenizer.convert_tokens_to_ids('</NE>')

        entity_start_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_start_token_id]
        entity_end_token_indices = [i for i, tok_id in enumerate(new_input_ids) if tok_id == entity_end_token_id]

        for st, et in zip(entity_start_token_indices, entity_end_token_indices):
            new_input_ids[st+1:et] = [tokenizer.convert_tokens_to_ids('<mask>') for _ in range(st+1, et)]
            new_attention_mask[st+1:et] = [0 for _ in range(st+1, et)]

        new_input_ids_list.append(new_input_ids)
        new_attention_mask_list.append(new_attention_mask)

    return torch.tensor(new_input_ids_list), torch.tensor(new_attention_mask_list)


class BartModel(lit_model.Model):

    def __init__(self, model_path, model_name, mask_entity=False):

        super().__init__()
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=600, padding=True, truncation=True,
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<NE>', '</NE>', '<CNTXT>', '</CNTXT>']})
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.mask_entity = mask_entity

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

        old_masks = encoded_input['attention_mask']
        if self.mask_entity:
            encoded_input['input_ids'], encoded_input['attention_mask'] = \
                masking_entities(self.tokenizer, encoded_input['input_ids'], encoded_input['attention_mask'])

        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                try:
                    encoded_input[tensor].requires_grad = True
                except:
                    pass
                encoded_input[tensor] = encoded_input[tensor].cuda()

        with torch.torch.set_grad_enabled(True):  # remove this if you need gradients.
            out = self.model(**encoded_input, output_attentions=True, output_hidden_states=True)
            ids = self.model.generate(encoded_input['input_ids'])

        encoded_input['attention_mask'] = old_masks.cuda()

        batched_outputs = {
            "probas": torch.nn.functional.softmax(out.logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "input_ntok": torch.sum(encoded_input["attention_mask"], dim=1),
            "target_ids": encoded_output["input_ids"],
            "target_ntok": torch.sum(encoded_output["attention_mask"], dim=1),
            "decoder_layer_1_attention": out.decoder_attentions[0],
            "encoder_layer_1_attention": out.encoder_attentions[0],
            "encoder_final_embedding": masked_token_mean(
                out.encoder_last_hidden_state, encoded_input["attention_mask"]),
            'token_embeddings': out.encoder_last_hidden_state,
            'encoder_attentions': out.encoder_attentions,
            'decoder_attentions': out.decoder_attentions

        }

        # <torch.float32>[batch_size, num_tokens, emb_dim]
        scalar_pred_for_gradients = torch.max(
            batched_outputs["probas"], dim=1, keepdim=False, out=None)[0]
        batched_outputs["input_emb_grad"] = torch.autograd.grad(
            scalar_pred_for_gradients,
            out.decoder_hidden_states[0],
            grad_outputs=torch.ones_like(scalar_pred_for_gradients))[0]

        detached_outputs = {k: v.detach().cpu().numpy() if type(v) != list else torch.tensor(v).cpu().numpy() for k, v in
                            batched_outputs.items()}

        detached_outputs["output_text"] = self.tokenizer.batch_decode(ids, skip_special_tokens=True)

        for output in utils.unbatch_preds(detached_outputs):
            input_ntok = output.pop("input_ntok")
            output["input_tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[1:input_ntok - 1])
            output_ntok = output.pop("target_ntok")
            output["target_tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("target_ids")[1:output_ntok - 1])
            output["token_grad_sentence"] = output["input_emb_grad"][1:input_ntok - 1]
            output['input_token_embedding'] = output['token_embeddings'][1:input_ntok - 1]
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
            "target_tokens": lit_types.Tokens(),
            'input_token_embedding': lit_types.TokenEmbeddings(align='input_tokens'),
            "token_grad_sentence": lit_types.TokenGradients(align="input_tokens", grad_for='input_token_embedding'),
            "output_text": lit_types.GeneratedText(parent="description"),
            "encoder_layer_1_attention": lit_types.AttentionHeads(align_in="input_tokens", align_out="input_tokens"),
            "decoder_layer_1_attention": lit_types.AttentionHeads(align_in="target_tokens", align_out="target_tokens"),
            "encoder_final_embedding": lit_types.Embeddings()
        }

        return spec

    def attention_weights_last_layer(self, input_text: str, **kwargs):
        with torch.no_grad():
            encoded_input = self.tokenizer.encode_plus(
                input_text,
                return_tensors="pt",
                max_length=600,
                padding="longest",
                truncation=True
            )
            if torch.cuda.is_available():
                self.model.cuda()
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            outputs = self.model(
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions = outputs.encoder_attentions
            decoder_attentions = outputs.decoder_attentions
            return encoder_attentions[-1], decoder_attentions[-1]

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
                self.model.cuda()
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            outputs = self.model(
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True,
                **kwargs
            )
            encoder_attentions = outputs.encoder_attentions
            decoder_attentions = outputs.decoder_attentions
            return encoder_attentions, decoder_attentions
        
    def Pearson_correlation(self, compareModel):
        # Example input
        att_input = ''

        # Get attention weights for each model
        encoder_att1, decoder_att1 = self.attention_weights(att_input, return_dict=True)
        encoder_att2, decoder_att2 = compareModel.attention_weights(att_input, return_dict=True)

        # Flatten attention weights into 1D arrays
        att1_flat = np.ravel(encoder_att1.numpy())
        att2_flat = np.ravel(encoder_att2.numpy())

        # Compute Pearson correlation coefficient
        corr_coef, p_value = np.corrcoef(att1_flat, att2_flat)

        return corr_coef

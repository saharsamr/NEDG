from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration


batch_counter = 0


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
            new_input_ids[st + 1:et] = [tokenizer.convert_tokens_to_ids('<mask>') for _ in range(st + 1, et)]
            new_attention_mask[st + 1:et] = [0 for _ in range(st + 1, et)]

        new_input_ids_list.append(new_input_ids)
        new_attention_mask_list.append(new_attention_mask)

    return torch.tensor(new_input_ids_list), torch.tensor(new_attention_mask_list)


class ModelComparison(lit_model.Model):

    def __init__(self, model_path_cpe, model_path_cme, model_name):

        super().__init__()
        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_name, model_max_length=300, padding=True, truncation=True,
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<NE>', '</NE>', '<CNTXT>', '</CNTXT>']})
        self.model_cpe = BartForConditionalGeneration.from_pretrained(model_path_cpe)
        self.model_cpe.resize_token_embeddings(len(self.tokenizer))
        self.model_cme = BartForConditionalGeneration.from_pretrained(model_path_cme)
        self.model_cme.resize_token_embeddings(len(self.tokenizer))

    def max_minibatch_size(self):
        return 4

    def predict_minibatch(self, inputs):

        global batch_counter
        print("Batch number: ", batch_counter)
        batch_counter += 1

        encoded_cpe_input = self.tokenizer.batch_encode_plus(
            [ex["context"] for ex in inputs],
            return_tensors="pt",
            max_length=300,
            padding="longest",
            truncation=True
        )
        encoded_output = self.tokenizer.batch_encode_plus(
            [ex["description"] for ex in inputs],
            return_tensors="pt",
            max_length=300,
            padding="longest",
            truncation=True
        )

        encoded_cme_input = encoded_cpe_input.copy()

        old_cme_masks = encoded_cme_input['attention_mask']
        encoded_cme_input['input_ids'], encoded_cme_input['attention_mask'] = \
            masking_entities(self.tokenizer, encoded_cme_input['input_ids'], encoded_cme_input['attention_mask'])

        if torch.cuda.is_available():
            self.model_cme.cuda()
            self.model_cpe.cuda()
            for tensor in encoded_cme_input:
                try:
                    encoded_cme_input[tensor].requires_grad = True
                except:
                    pass
                encoded_cme_input[tensor] = encoded_cme_input[tensor].cuda()
            for tensor in encoded_cpe_input:
                try:
                    encoded_cpe_input[tensor].requires_grad = True
                except:
                    pass
                encoded_cpe_input[tensor] = encoded_cpe_input[tensor].cuda()

        # TODO: check if this is the right place to put this
        encoded_cme_input['attention_mask'] = old_cme_masks.cuda()

        with torch.torch.set_grad_enabled(True):
            out_cpe = self.model_cpe(**encoded_cpe_input, output_attentions=True, output_hidden_states=True)
            ids_cpe = self.model_cpe.generate(encoded_cpe_input['input_ids'])

            out_cme = self.model_cme(**encoded_cme_input, output_attentions=True, output_hidden_states=True)
            ids_cme = self.model_cme.generate(encoded_cme_input['input_ids'])

        batched_outputs = {
            "probas_cpe": torch.nn.functional.softmax(out_cpe.logits, dim=-1),
            "input_ids_cpe": encoded_cpe_input["input_ids"],
            "input_ntok_cpe": torch.sum(encoded_cpe_input["attention_mask"], dim=1),
            "target_ids_cpe": encoded_output["input_ids"],
            "target_ntok_cpe": torch.sum(encoded_output["attention_mask"], dim=1),
            # "decoder_layer_1_attention_cpe": out_cpe.decoder_attentions[0],
            # "encoder_layer_1_attention_cpe": out_cpe.encoder_attentions[0],
            "encoder_final_embedding_cpe": masked_token_mean(
                out_cpe.encoder_last_hidden_state, encoded_cpe_input["attention_mask"]),
            'token_embeddings_cpe': out_cpe.encoder_last_hidden_state,
            # 'encoder_attentions_cpe': out_cpe.encoder_attentions,
            # 'decoder_attentions_cpe': out_cpe.decoder_attentions,

            "probas_cme": torch.nn.functional.softmax(out_cme.logits, dim=-1),
            "input_ids_cme": encoded_cpe_input["input_ids"],
            "input_ntok_cme": torch.sum(encoded_cpe_input["attention_mask"], dim=1),
            "target_ids_cme": encoded_output["input_ids"],
            "target_ntok_cme": torch.sum(encoded_output["attention_mask"], dim=1),
            # "decoder_layer_1_attention_cme": out_cme.decoder_attentions[0],
            # "encoder_layer_1_attention_cme": out_cme.encoder_attentions[0],
            "encoder_final_embedding_cme": masked_token_mean(
                out_cme.encoder_last_hidden_state, encoded_cme_input["attention_mask"]),
            'token_embeddings_cme': out_cme.encoder_last_hidden_state,
            # 'encoder_attentions_cme': out_cme.encoder_attentions,
            # 'decoder_attentions_cme': out_cme.decoder_attentions
        }

        # TODO: check if this is the right gradient to take
        scalar_pred_for_gradients_cpe = torch.max(
            batched_outputs["probas_cpe"], dim=1, keepdim=False, out=None)[0]
        batched_outputs["input_emb_grad_cpe"] = torch.autograd.grad(
            scalar_pred_for_gradients_cpe,
            out_cpe.encoder_last_hidden_state,
            grad_outputs=torch.ones_like(scalar_pred_for_gradients_cpe))[0]

        scalar_pred_for_gradients_cme = torch.max(
            batched_outputs["probas_cme"], dim=1, keepdim=False, out=None)[0]
        batched_outputs["input_emb_grad_cme"] = torch.autograd.grad(
            scalar_pred_for_gradients_cme,
            out_cme.encoder_last_hidden_state,
            grad_outputs=torch.ones_like(scalar_pred_for_gradients_cme))[0]

        detached_outputs = {k: v.detach().cpu().numpy() if type(v) != list and type(v) != tuple else torch.tensor(v).cpu().numpy() for k, v
                            in
                            batched_outputs.items()}

        detached_outputs["output_text_cpe"] = self.tokenizer.batch_decode(ids_cpe, skip_special_tokens=False)
        detached_outputs["output_text_cme"] = self.tokenizer.batch_decode(ids_cme, skip_special_tokens=False)

        for output in utils.unbatch_preds(detached_outputs):

            input_ntok_cpe = output.pop("input_ntok_cpe")
            output["input_tokens_cpe"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids_cpe")[1:input_ntok_cpe - 1])
            output_ntok_cpe = output.pop("target_ntok_cpe")
            output["target_tokens_cpe"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("target_ids_cpe")[1:output_ntok_cpe - 1])
            output["token_grad_sentence_cpe"] = output["input_emb_grad_cpe"][1:input_ntok_cpe - 1]
            output['input_token_embedding_cpe'] = output['token_embeddings_cpe'][1:input_ntok_cpe - 1]
            # output['encoder_layer_1_attention_cpe'] = \
            #     output['encoder_layer_1_attention_cpe'][:, 1:input_ntok_cpe - 1, 1:input_ntok_cpe - 1]

            input_ntok_cme = output.pop("input_ntok_cme")
            output["input_tokens_cme"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids_cme")[1:input_ntok_cme - 1])
            output_ntok_cme = output.pop("target_ntok_cme")
            output["target_tokens_cme"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("target_ids_cme")[1:output_ntok_cme - 1])
            output["token_grad_sentence_cme"] = output["input_emb_grad_cme"][1:input_ntok_cme - 1]
            output['input_token_embedding_cme'] = output['token_embeddings_cme'][1:input_ntok_cme - 1]
            # output['encoder_layer_1_attention_cme'] = \
            #     output['encoder_layer_1_attention_cme'][:, 1:input_ntok_cme - 1, 1:input_ntok_cme - 1]

            yield output

    def input_spec(self) -> lit_types.Spec:

        return {
            "title": lit_types.TextSegment(),
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment()
        }

    def output_spec(self) -> lit_types.Spec:

        spec = {
            "target_text_cpe": lit_types.TextSegment(),
            "input_tokens_cpe": lit_types.Tokens(),
            "target_tokens_cpe": lit_types.Tokens(),
            'input_token_embedding_cpe': lit_types.TokenEmbeddings(align='input_tokens_cpe'),
            "token_grad_sentence_cpe": lit_types.TokenGradients(align="input_tokens_cpe",
                                                                grad_for='input_token_embedding_cpe'),
            "output_text_cpe": lit_types.GeneratedText(parent="description"),
            # "encoder_layer_1_attention_cpe": lit_types.AttentionHeads(align_in="input_tokens_cpe",
            #                                                           align_out="input_tokens_cpe"),
            # "decoder_layer_1_attention_cpe": lit_types.AttentionHeads(align_in="target_tokens_cpe",
            #                                                           align_out="target_tokens_cpe"),
            # "encoder_final_embedding_cpe": lit_types.Embeddings(),

            "target_text_cme": lit_types.TextSegment(),
            "input_tokens_cme": lit_types.Tokens(),
            "target_tokens_cme": lit_types.Tokens(),
            'input_token_embedding_cme': lit_types.TokenEmbeddings(align='input_tokens_cme'),
            "token_grad_sentence_cme": lit_types.TokenGradients(align="input_tokens_cme",
                                                                grad_for='input_token_embedding_cme'),
            "output_text_cme": lit_types.GeneratedText(parent="description"),
            # "encoder_layer_1_attention_cme": lit_types.AttentionHeads(align_in="input_tokens_cme",
            #                                                           align_out="input_tokens_cme"),
            # "decoder_layer_1_attention_cme": lit_types.AttentionHeads(align_in="target_tokens_cme",
            #                                                           align_out="target_tokens_cme"),
            # "encoder_final_embedding_cme": lit_types.Embeddings()
        }

        return spec

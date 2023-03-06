from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers.cross_encoder import CrossEncoder


def tokenize_fast(texts: List[str], tokenizer: AutoTokenizer, max_length: int = 256):
    """huggingface code tokenize may be too slow for long text"""
    def get_input_ids(text):
        tokens = tokenizer.tokenize(text)
        return tokenizer.convert_tokens_to_ids(tokens)

    def prepare_for_model(ids, pair_ids):
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        encoded_inputs = {}
        total_len = len_ids + len_pair_ids + tokenizer.num_special_tokens_to_add(pair=pair)
        if total_len > max_length:
            num_tokens_to_remove = total_len - max_length
            i = 0
            cut_ids, cut_pair_ids = 0, 0
            while i < num_tokens_to_remove:
                if len(ids) > len(pair_ids):
                    tokens_to_remove = min(len(ids) - len(pair_ids), num_tokens_to_remove - i)
                    cut_ids += tokens_to_remove
                    i += tokens_to_remove
                elif len(ids) < len(pair_ids):
                    tokens_to_remove = min(len(pair_ids) - len(ids), num_tokens_to_remove - i)
                    cut_pair_ids += tokens_to_remove
                    i += tokens_to_remove
                else:
                    tokens_to_remove = num_tokens_to_remove - i
                    cut_ids += int(tokens_to_remove / 2)
                    cut_pair_ids += int((tokens_to_remove + 1) / 2)
                    i += tokens_to_remove
            ids = ids[:-cut_ids]
            pair_ids = pair_ids[:-cut_pair_ids]

        sequence = tokenizer.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids, pair_ids)
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids
        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=None, prepend_batch_axis=False
        )
        return batch_outputs

    def _batch_prepare_for_model(batch_ids_pairs):
        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = prepare_for_model(
                first_ids,
                second_ids
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = tokenizer.pad(
            batch_outputs,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=None,
            return_attention_mask=None,
        )
        batch_outputs = BatchEncoding(batch_outputs, tensor_type='pt')
        return batch_outputs

    def _batch_encode_plus(batch_text_or_text_pairs):
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            ids, pair_ids = ids_or_pair_ids
            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids)
            input_ids.append((first_ids, second_ids))
        batch_outputs = _batch_prepare_for_model(input_ids)
        return batch_outputs

    queries = texts[0]
    candidates = texts[1]
    text_pairs = list(zip(queries, candidates))

    return _batch_encode_plus(text_pairs)


def cross_model_infer(cross_model: CrossEncoder, model_inputs: List[List[str]]):
    cross_model.model.eval()
    cross_model.model.to(torch.device("cuda"))
    pred_scores = []
    with torch.no_grad():
        texts = list(zip(*model_inputs))
        tokenized = tokenize_fast(texts, tokenizer=cross_model.tokenizer, max_length=256)
        for name in tokenized:
            tokenized[name] = tokenized[name].to(torch.device("cuda"))
        features = tokenized
        model_predictions = cross_model.model(**features, return_dict=True)
        logits = cross_model.default_activation_function(model_predictions.logits)
        pred_scores.extend(logits)

    if cross_model.config.num_labels == 1:
        pred_scores = [score[0] for score in pred_scores]
    pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
    scores = pred_scores

    return scores

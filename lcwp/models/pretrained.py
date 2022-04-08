""" Here we just load the pretrained models from HF and make predictions for each query """
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering, pipeline
from lcwp.data.util import yield_examples, save_batch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering, pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import json
import logging

import torch
from torch import nn


def _predict_with_bart(example_generator, model, tokenizer, device):  # "facebook/bart-large"
    """ Makes predictions with BartForConditionalGeneration's encoder __call__.
     It takes a <masked> query and predicts the infilling.
     """
    for sequences, batch in example_generator():
        sequences = [item.replace('[MASK]', tokenizer.mask_token) for item in sequences]
        inp = tokenizer(sequences,
                        padding=True,
                        truncation=True,
                        return_tensors="pt").to(device)

        mask_token_index = torch.where(inp["input_ids"] == tokenizer.mask_token_id)

        token_logits = model(**inp).logits

        results = []
        for ind, t in enumerate(token_logits):
            mask_token_logits = t[mask_token_index[1], :]
            top_tokens = torch.topk(mask_token_logits, 30, dim=1)
            results.append([(tokenizer.decode(index), value) for index, value in
                            zip(top_tokens.indices[ind].tolist(), top_tokens.values[ind].tolist())])

        yield results, batch


def _predict_with_bart_infilling(example_generator, model, tokenizer, device):
    """ Makes predictions with bart that was fine-tuned for ranked infilling:
     It takes a context query 'A <mask> B' and generates results 'C [3] D [3] ... <eos>'

     we assume that the tokenizer understands the special tokens
     """
    def _parse_infilling_output(decoded_sequence):
        prev = []
        results = []
        for item in decoded_sequence.split():
            if item in {"[1]", "[2]", "[3]"}:
                results.append((" ".join(prev), item))
                prev = []
            else:
                prev.append(item)

        return results

    for sequences, batch in example_generator():
        results = []
        for sequence in sequences:
            prompt = sequence.replace("[MASK]", tokenizer.mask_token)
            inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
            outputs = model.generate(inputs, max_length=250, do_sample=False)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            results.append(_parse_infilling_output(generated))
        yield results, batch


def _predict_with_clm_infilling(example_generator, model, tokenizer, device):
    """ Makes predictions with any clm that was fine-tuned for ranked infilling:
     It takes a context query 'A [MASK] B [SEP]' and generates results 'C [3] D [3] ... [EOS]'

     we assume that the tokenizer understands the special tokens
     """
    def _parse_infilling_output(decoded_sequence):
        r = []
        _ = []
        for s in decoded_sequence.split():
            if s == '[ANSWER]':
                r.append(' '.join(_))
                _ = []
            else:
                _.append(s)
        return [(s, 0) for s in r]

    for sequences, batch in example_generator():
        results = []
        for sequence in sequences:
            prompt = f"{sequence} [SEP]"
            inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
            outputs = model.generate(inputs, max_length=250, do_sample=False)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            results.append(_parse_infilling_output(generated.replace(prompt, '')))
        yield results, batch


def _predict_with_clm(example_generator, model, tokenizer, device):
    """ Makes predictions with any clm.generate by
     - predicting with a left-side context
     - adding the right side context
     - determining the ranking through perplexity
     """
    def forward(sequence):
        parts = sequence.split("[MASK]")
        if not parts[0] or len(parts) == 1:
            return []
        left_side_context = tokenizer(parts[0], return_tensors="pt").to(device)
        right_side_context = tokenizer(f" {parts[1]}", return_tensors='pt').to(device)
        right_side_context = right_side_context["input_ids"]

        left_side_context_ids = left_side_context["input_ids"]
        next_token_logits = model(**left_side_context).logits[:, -1, :]
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=30, top_p=1.0)
        probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
        top_tokens = torch.topk(probs, k=30)

        result = []
        for t in range(0, top_tokens.indices.size()[1]):
            token_at_t = top_tokens.indices[:, t:t + 1]
            full_context_input_id = torch.cat(
                [left_side_context_ids, token_at_t, right_side_context], dim=-1)

            with torch.no_grad():
                output = model(full_context_input_id, labels=full_context_input_id.clone())
                neg_log_likelihood = output[0]

            result.append((
                tokenizer.decode(token_at_t.tolist()[0]),
                neg_log_likelihood.item()
            ))
        result.sort(key=lambda x: x[1])
        return result

    for sequences, batch in example_generator():
        yield [forward(s) for s in sequences], batch


def _predict_with_mlm(example_generator, model, tokenizer, device):
    for sequences, batch in example_generator(mask_token=tokenizer.mask_token):
        inputs = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)

        token_logits = model(**inputs).logits
        results = []
        for ind, t in enumerate(token_logits):
            mask_token_logits = t[mask_token_index[1], :]
            top_tokens = torch.topk(mask_token_logits, 30, dim=1)
            results.append([(tokenizer.decode(index), value) for index, value in zip(top_tokens.indices[ind].tolist(),
                                                                                     top_tokens.values[ind].tolist())])

        yield results, batch


def predict_from_pretrained(output_dir, model_name, model_type, input_dataset, device_id=0, batch_size=30,
                            min_query_length=3, max_query_length=9):
    """ Main class for predicting with a pretrained model

    :param output_dir: where to store the results
    :param model_name: hugginface-style model name or directory of a saved model
    :param model_type: 'mlm' or 'clm'
    :param input_dataset: a Path to the input dataset
    :param device_id: cuda device id, or -1 for CPU
    :param batch_size: how many queries to predict in parallel
    """
    if device_id == -1 or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f"cuda:{0}"
    logging.warning(torch.cuda.is_available())

    input_generator = partial(yield_examples, input_dataset, batch_size, min_query_length, max_query_length)
    if model_type == 'mlm':
        forward_function = _predict_with_mlm
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    elif model_type == 'clm':
        forward_function = _predict_with_clm
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    elif model_type == 'bart':
        forward_function = _predict_with_bart
        model = BartForConditionalGeneration.from_pretrained(model_name, forced_bos_token_id=0).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    elif model_type == 'infill':
        forward_function = _predict_with_clm_infilling
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    elif model_type == 'bart-infill':
        forward_function = _predict_with_bart_infilling
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")

    for results, batch in forward_function(input_generator, model, tokenizer, device):
        save_batch(batch, results, output_dir / f"{input_dataset.stem}-{model_type}.ndjson")


if __name__ == "__main__":
    pass

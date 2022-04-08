""" Finetune a base model given a dataset

For MLM: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import logging


def ft_seq2seq_infilling(save_directory, train_path, validation_path, model_checkpoint="facebook/bart-base"):
    """
    Finetune BART for Conditional Generation
    https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/5
    """
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['query'], pad_to_max_length=True,
                                                      max_length=block_size, truncation=True)
        target_encodings = tokenizer.batch_encode_plus(example_batch['answer'], pad_to_max_length=True,
                                                       max_length=block_size, truncation=True)

        labels = target_encodings['input_ids'].copy()

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
        }

        return encodings

    def tokenize_function(examples):
        examples["query"] = examples["query"].replace('[MASK]', tokenizer.mask_token)
        rel3 = ' '.join([f"{e} [3]" for e in examples['rel3']])
        rel2 = ' '.join([f"{e} [2]" for e in examples['rel2']])
        rel1 = ' '.join([f"{e} [1]" for e in examples['rel1']])
        examples["answer"] = f"{tokenizer.bos_token} {rel3} {rel2} {rel1} {tokenizer.eos_token}"

        return examples

    new_model_name = f"netspeak-{model_checkpoint.split('/')[-1]}-encoder-mlm"
    logging.warning(f"finetuning {model_checkpoint} for s2sinfilling as {new_model_name}")
    new_model_name = save_directory / new_model_name

    logging.warning("loading model")
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = BartTokenizer.from_pretrained(model_checkpoint, use_fast=False, add_prefix_space=True,
                                              additional_special_tokens=["[1], [2], [3]"])

    model.resize_token_embeddings(len(tokenizer))

    logging.warning("tokenizing dataset")
    dataset = load_dataset('json', data_files={"train": train_path, "validation": validation_path})
    tokenized_datasets = dataset.map(tokenize_function, batched=False, num_proc=4,
                                     remove_columns=["id", 'rel1', 'rel2', 'rel3', 'length', 'position', 'source',
                                                     'pos_class'])

    block_size = 256

    logging.warning(f"dataset: {tokenized_datasets}")
    logging.warning(f"prepare data for training")

    lm_datasets = tokenized_datasets.map(
        convert_to_features,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["query", 'answer'],
    )

    training_args = TrainingArguments(
        new_model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"])

    logging.warning(f"start training")
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    tokenizer.save_pretrained(new_model_name)
    model.save_pretrained(new_model_name)


def ft_encoder_mlm(save_directory, train_path, validation_path, model_checkpoint="distilbert-base-uncased"):
    """
    Finetune Distillbert on MLM
    https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    """

    def tokenize_function(examples):
        # examples["query"][0] = examples["query"][0].replace('[MASK]', examples["answer"][0])  # for wikitext finetuning
        top = examples["rel3"][0] + examples["rel2"][0] + examples["rel1"][0]  # for netspeak-training finetuning
        examples["query"][0] = examples["query"][0].replace('[MASK]', top[0])
        return tokenizer(examples["query"])

    new_model_name = f"netspeak-{model_checkpoint.split('/')[-1]}-encoder-mlm"
    logging.warning(f"finetuning {model_checkpoint} with mlm as {new_model_name}")
    new_model_name = save_directory / new_model_name

    logging.warning("loading model")
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    logging.warning("tokenizing dataset")
    dataset = load_dataset('json', data_files={"train": train_path, "validation": validation_path})
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4,
                                     remove_columns=["query", 'rel1', 'rel2', 'rel3', 'length', 'position', 'source',
                                                     'pos_class'])
    logging.warning(f"dataset: {tokenized_datasets}")
    logging.warning(f"prepare data for training")
    training_args = TrainingArguments(
        new_model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    logging.warning(f"start training")
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    tokenizer.save_pretrained(new_model_name)
    model.save_pretrained(new_model_name)


def ft_decoder_infilling(save_directory, train_path, validation_path, model_checkpoint="distilgpt2"):
    """
    Finetune GPT for mask-filling
    https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    """

    def _tokenize_function(examples):
        rel3 = ' '.join([f"{e} [ANSWER]" for e in examples['rel3'][0]])
        rel2 = ' '.join([f"{e} [ANSWER]" for e in examples['rel2'][0]])
        rel1 = ' '.join([f"{e} [ANSWER]" for e in examples['rel1'][0]])
        examples["query"][0] = f"{examples['query'][0]} [SEP] {rel3} {rel2} {rel1} {tokenizer.eos_token}"
        return tokenizer(examples["query"])

    def _group_texts(examples):
        result = {'input_ids': [], 'attention_mask': []}
        pad_token_id = tokenizer(tokenizer.pad_token)['input_ids'][0]
        inp_block = []
        attn_block = []
        for inp, attn in zip(examples['input_ids'], examples['attention_mask']):
            if len(inp_block) + len(inp) <= block_size:
                inp_block += inp
                attn_block += attn
            else:
                inp_block += [pad_token_id] * (block_size - len(inp_block))
                attn_block += [0] * (block_size - len(attn_block))
                result['input_ids'].append(inp_block)
                result['attention_mask'].append(attn_block)
                inp_block = inp[:block_size]
                attn_block = attn[:block_size]

        inp_block += [pad_token_id] * (block_size - len(inp_block))
        attn_block += [0] * (block_size - len(attn_block))
        result['input_ids'].append(inp_block)
        result['attention_mask'].append(attn_block)

        result["labels"] = result["input_ids"].copy()
        return result

    new_model_name = f"netspeak-{model_checkpoint.split('/')[-1]}-decoder-infilling"
    logging.warning(f"finetuning {model_checkpoint} for infilling as {new_model_name}")
    new_model_name = save_directory / new_model_name
    logging.warning("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, add_prefix_space=True,
                                              pad_token="[PAD]",
                                              additional_special_tokens=["[MASK], [SEP], [ANSWER]"])
    model.resize_token_embeddings(len(tokenizer))

    logging.warning("tokenizing dataset")
    gpt_dataset = load_dataset('json', data_files={"train": train_path, "validation": validation_path})
    tokenized_datasets = gpt_dataset.map(_tokenize_function, batched=True, num_proc=4,
                                         remove_columns=["query", "id", 'rel1', 'rel2', 'rel3', 'length', 'position',
                                                         'source', 'pos_class'])

    block_size = 256
    logging.warning(f"dataset: {tokenized_datasets}")
    logging.warning(f"prepare data for training")
    lm_datasets = tokenized_datasets.map(
        _group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
    )

    training_args = TrainingArguments(
        new_model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"])

    logging.warning(f"start training")
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    tokenizer.save_pretrained(new_model_name)
    model.save_pretrained(new_model_name)


def finetune(output_dir, model_name, strategy, training_path, validation_path):
    """ Main class for fine-tuning models to do LCWP

    :param output_dir: where to save the finetuned model
    :param model_name: hugginface-style model name or directory of a saved base model that should be finetuned
    :param strategy: which strategy to follow for finetuning:
        - 'mlm' simple masked language modelling
        - 'clminfill' a decoder model, given a masked query as promt, that predicts the ranking in order.
        - 's2sranked' an seq2seq model, given a masked query, that predicts the ranking in order.
    :param training_path: Path to the .ndjson file used for training
    :param validation_path: Path to the .ndjson file used for validation
    """
    logging.warning(f"Is cuda available? {torch.cuda.is_available()}")

    if strategy == 'mlm':
        ft_function = ft_encoder_mlm
    elif strategy == 'clminfill':
        ft_function = ft_decoder_infilling
    elif strategy == 's2sranked':
        ft_function = ft_seq2seq_infilling
    else:
        raise NotImplementedError(f"Strategy {strategy} is not implemented.")

    ft_function(save_directory=output_dir, train_path=training_path, validation_path=validation_path,
                model_checkpoint=model_name)


if __name__ == "__main__":
    pass

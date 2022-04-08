"""
Utility functions for loading, saving, and some automated extraction.
"""
import json
import logging
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path


def add_to_dataset(output_path, dataset_name, output_gen, add_id=False):
    """ Adds the output of the generator to the given dataset in the output directory
    Generators should return a dict
    """
    dataset_path = output_path
    dataset_path.mkdir(exist_ok=True, parents=True)
    with open(dataset_path / f'{dataset_name}.ndjson', 'a') as of:
        for ind, d in enumerate(output_gen):
            if add_id:
                d['id'] = str(uuid4())
            of.write(f"{json.dumps(d)}\n")
            if ind % 101 == 0:
                of.flush()


def get_pos_class(ngram, ind):
    """ for a given n-gram (spacy.Sequence) and the index to be masked, determine the POS class
        for the evaluation of low-context queries.

     We differentiate:
        - VERB/AUX solo
        - VERB/AUX multi
        - NOUN solo
        - NOUN multi
        - DET, PRON
        - ADJ, ADV
        - ADP, PART, CCONJ, INTJ, SCONJ """
    masked_token = ngram[ind]
    if masked_token.pos_ == "VERB" or masked_token.pos_ == "AUX":
        other_pos = set.union({t.pos_ for t in ngram[:ind]}, ({t.pos_ for t in ngram[ind + 1:]}))
        if 'VERB' in other_pos or 'AUX' in other_pos:
            return 'VERB_AUX_M'
        return 'VERB_AUX_S'
    if masked_token.pos_ == "NOUN":
        other_pos = set.union({t.pos_ for t in ngram[:ind]}, ({t.pos_ for t in ngram[ind + 1:]}))
        if 'NOUN' in other_pos:
            return 'NOUN_M'
        return 'NOUN_S'
    if masked_token.pos_ == "DET" or masked_token.pos_ == "PRON":
        return 'DET_PRON'
    if masked_token.pos_ == "ADJ" or masked_token.pos_ == "ADV":
        return 'ADJ_ADV'
    if masked_token.pos_ == "ADP" or masked_token.pos_ == "PART" or masked_token.pos_ == "CCONJ" or masked_token.pos_ == "INTJ" or masked_token.pos_ == "SCONJ":
        return 'STOP'


def yield_ngrams(doc_like, ngram_range: range):
    """ yield all valid spans of length n from the doc which have
        - no special characters
        - no non-alphanumeric
        - no unk
    """
    good_tokens = [1 if token.is_alpha and not token.is_punct and not token.text == 'unk' else 0 for token in doc_like]
    for n in ngram_range:
        for ind, token in enumerate(doc_like[:-n]):
            if sum(good_tokens[ind:ind+n]) == n:
                yield doc_like[ind:ind+n]


def yield_examples(dataset_path, batch_size=1, min_query_length=3, max_query_length=9, mask_token=None):
    ind = 0
    batch = []
    for line in open(dataset_path, 'r'):
        line = json.loads(line)
        if not (min_query_length <= line['length'] <= max_query_length):
            continue
        batch.append(line)
        ind += 1
        if ind == batch_size:
            if mask_token:
                texts = [item['query'].replace('[MASK]', mask_token) for item in batch]
            else:
                texts = [item['query'] for item in batch]
            yield texts, batch
            batch = []
            ind = 0


def save_batch(batch: list, results: list, location: Path):
    """ Save model prediction
    :param batch: is a list of lcwp query json objects
    :param results: is a list of query results, where each result is a list of typles (word, score)
    :param location: where to save the results
    """
    for b, res in zip(batch, results):
        b['rel1'] = [r[0] for r in res]
        b['rel2'] = []
        b['rel3'] = []
        b['scores'] = [r[1] for r in res]
    open(location, 'a').writelines([f"{json.dumps(b)}\n" for b in batch])


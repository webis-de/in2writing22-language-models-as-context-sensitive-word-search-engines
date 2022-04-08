""" Functions to extract queries from teachers cloze test dataset (CLOTH):
 - https://arxiv.org/abs/1711.03225
 - github.com/qizhex/Large-scale-Cloze-Test-Dataset-Created-by-Teachers
"""
import xml.etree.ElementTree as et
import spacy
from spacy.tokens import Token
from lcwp.data.util import add_to_dataset, get_pos_class
from tqdm import tqdm
import logging
import json
import re

DISALLOWED_TAGS = {'PROPN', 'NUM'}
re_punct = re.compile(r"[,;:\"{}\[\]]")
re_double_quotes = re.compile(r"''")
re_numbers_before_masks = re.compile(r"(?<!\d)(\d|\d\d)\s*_")
Token.set_extension("is_mask", default=False)
Token.set_extension("options", default=None)
Token.set_extension("answer", default=None)
answer_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
nlp = spacy.load('en_core_web_trf')


def resolve_apostrophes(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"Won\'t", "Will not", phrase)
    phrase = re.sub(r"Can\'t", "Can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"\'\'", "\"", phrase)
    return phrase


def _yield_cloth_docs(cloth_path):
    """ a generator that loads documents from a cloth corpus file and yields the contained document(s)
    @param cloth_path: the path to the input file
    """
    for cloth_file in (cloth_path / 'CLOTH').rglob('*.json'):
        print(cloth_file)
        source_school = cloth_file.parent.stem
        cloth_file = open(cloth_file, 'r').read()
        cloth_file = json.loads(cloth_file)
        article = cloth_file['article']
        options = cloth_file['options']
        answers = cloth_file['answers']

        article = re.sub("[.]", ". ", article)
        article = re.sub(re_double_quotes, " ", article)
        article = re.sub(re_punct, " ", article)
        article = re.sub(re_numbers_before_masks, " _", article)
        article = re.sub(r"[\s]+", " ", article)
        article = resolve_apostrophes(article)

        yield article, options, answers, source_school


def extend_doc(cloth_test, cloth_options, cloth_answers):
    """ Processing of the cloth format. We insert the answer in the _ position to calculate the POS Tags,
        then we add answer and options to the spacy.Token and change the answer in the text back to [MASK] """
    filled_text = []
    option_index = 0
    for t in cloth_test.split(" "):
        if t == "_":
            cloth_answer = cloth_answers.pop(0)
            answer_ind = answer_to_index[cloth_answer]
            options = cloth_options[option_index]
            option_index += 1
            filled_text.append(options[answer_ind])
        else:
            filled_text.append(t)

    doc = nlp(cloth_test)
    filled_doc = nlp(" ".join(filled_text))
    for d1, d2 in zip(doc, filled_doc):
        if d1.text == "_":
            options = cloth_options.pop(0)
            d2._.options = [o for o in options if o != d2.text]
            d2._.answer = d2.text
            d2._.is_mask = True
    return filled_doc


def yield_ngrams(doc_like, ngram_range: range):
    """ yield all valid spans of length n from the doc which have
        - no special characters
        - no non-alphanumeric
        - no unk
    """
    def score_token(token):
        if token._.is_mask:
            return 11
        if token.is_alpha and not token.is_punct and not token.text == 'unk':
            return 1
        return 0

    good_tokens = [score_token(t) for t in doc_like]

    for n in ngram_range:
        for ind, token in enumerate(doc_like[:-n]):
            # print(doc_like[ind:ind+n], good_tokens[ind:ind+n])
            if sum(good_tokens[ind:ind+n]) == n+10:
                yield doc_like[ind:ind+n]


def make_masked_queries(doc_like, ngram_range: range, source_school) -> dict:
    """ Creates each possible low-context queries for each n-gram and each mask position in the n-gram in the doc_like
    Ignores mask-position where the masked token has a :DISALLOWED_TAGS: or is otherwise part of a named entity.

    @param doc_like: a spacy.doc or spacy.span or anything that behaves like it
    @param ngram_range: a range of n for which n-grams should be generated

    :yield: a dict for each query with the query text
    """
    for ngram in yield_ngrams(doc_like, ngram_range):
        for ind, token in enumerate(ngram):  # make queries with tokens in different positions:
            if token._.is_mask:
                query = f"{str.lower(ngram[:ind].text)} [MASK] {str.lower(ngram[ind+1:].text)}".strip()
                yield {"query": query, "rel2": [token._.answer], "rel1": token._.options, "length": len(ngram), "position": ind,
                       "source": f"cloth_{source_school}school", "pos_class": get_pos_class(ngram, ind)}


def compile_cloth(output_path, cloth_path):
    # training queries: No further requirements other than the mask can't be a PROPN
    for cloth_test, cloth_options, cloth_answers, source_school in _yield_cloth_docs(cloth_path):
        extended_doc = extend_doc(cloth_test, cloth_options, cloth_answers)
        add_to_dataset(output_path, 'cloth-test', make_masked_queries(extended_doc, range(3, 10, 1), source_school), add_id=True)


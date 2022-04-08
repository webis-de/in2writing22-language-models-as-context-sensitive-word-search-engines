""" Functions to extract single-choice low-context MLM queries from four datasets to fine-tune models with.
 - europarl - V7 as downloadable from https://www.statmt.org/europarl/, using only the english section
 - wikitext - cite https://arxiv.org/pdf/1609.07843.pdf version 103, only training tokens
 @misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
 - nyt - New York Times Corpus as disctibuted by the Linguistic Data Consortium (LDC). Catalog number LDC2008T19, ISBN 1-58563-486-5
"""
import xml.etree.ElementTree as et
import spacy
from lcwp.data.util import add_to_dataset, get_pos_class, yield_ngrams
from tqdm import tqdm
import logging

DISALLOWED_TAGS = {'PROPN', 'NUM'}


def _yield_europarl_docs(europarl_path):
    """ a generator that loads documents from a nyt corpus file and yields the contained document(s)
    @param inp: tha path to the input file
    """
    doc = ""
    for document in (europarl_path / "txt/en").glob("*.txt"):
        for line in open(document, 'r'):
            line = line.strip()
            if not line or line.startswith("<P"):
                continue
            if line.startswith("<SPEAKER") and doc:
                yield doc
                doc = ""
                continue
            doc += f" {line}"


def _yield_wikitext_docs(wikitext_filepath, test=False):
    """ a generator that loads documents from a nyt corpus file and yields the contained document(s)
    @param inp: tha path to the input file
    """
    p = wikitext_filepath
    doc = ""
    for line in open(p, 'r'):
        line = line.strip()
        if not line:
            continue
        if (line.startswith("=") or line.startswith(" =")) and doc:
            yield doc
            doc = ""
            continue
        doc += f" {line}"


def _yield_nyt_docs(nyt_path):
    """ a generator that loads documents from a nyt corpus file and yields the contained document(s)
    @param inp: tha path to the input file
    """
    for article in (nyt_path / 'data').rglob('*.xml'):
        tree = et.parse(article)
        root = tree.getroot()

        # Extract the full_text from the xml dom
        body_content = list(root.iter("body.content"))[0]
        sentences = []
        for block in body_content:
            if block.attrib["class"] == "full_text":
                for p in block.iter("p"):
                    sentences.append(p.text.replace("'", ""))
        yield " ".join(sentences)


def make_masked_queries(doc_like, ngram_range: range) -> dict:
    """ Creates each possible low-context queries for each n-gram and each mask position in the n-gram in the doc_like
    Ignores mask-position where the masked token has a :DISALLOWED_TAGS: or is otherwise part of a named entity.

    @param doc_like: a spacy.doc or spacy.span or anything that behaves like it
    @param ngram_range: a range of n for which n-grams should be generated

    :yield: a dict for each query with the query text
    """
    for ngram in yield_ngrams(doc_like, ngram_range):
        for ind, token in enumerate(ngram):  # make queries with tokens in different positions:
            if not token.pos_ in DISALLOWED_TAGS and token.ent_type == 0:
                query = f"{ngram[:ind].text} [MASK] {ngram[ind+1:].text}".strip()
                yield {"query": query, "answer": token.text, "options": [], "length": len(ngram), "position": ind,
                       "source": 'wikitext-103', "pos_class": get_pos_class(ngram, ind)}


def compile_singlechoice(output_path, nyt_path, wikitext_path, europarl_path, training=False):
    if europarl_path:
        logging.warning("Europarl parsing is not implemented yet")
        # europarl_docs = _yield_europarl_docs(europarl_path)
    if nyt_path:
        logging.warning("NYT parsing is not implemented yet")
        # nyt_docs = _yield_nyt_docs(nyt_path)

    nlp = spacy.load('en_core_web_trf')
    for wikitext_train in tqdm(_yield_wikitext_docs(wikitext_path), desc="Docs in wikitext"):
        doc = nlp(wikitext_train)
        if training:
            add_to_dataset(output_path, 'wikitext-train', make_masked_queries(doc, range(3, 10, 1)))
        else:
            add_to_dataset(output_path, 'wikitext-test', make_masked_queries(doc, range(3, 10, 1)), add_id=True)

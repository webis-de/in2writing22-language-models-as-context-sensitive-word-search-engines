""" Functions to get the silver-standard answers to the teachers' queries from Netspeak """
from sonora import client
import lcwp.proto.NetspeakService_pb2 as nspb
import lcwp.proto.NetspeakService_pb2_grpc as nspbg
from lcwp.data.util import add_to_dataset
import re
import json
import logging
from tqdm import tqdm


def _get_rel(response, relevance, threshold1=10000, threshold2=100000):
    """ Return the WORD_FOR_QMARK in the response in the order given by Netspeak.

    @param response: netspeaks response in protobuf.
    @param relevance: if 1, return the WORD_FOR_QMARK if phrase frequency is below threshold1, 2 if between, 3 if above thr2
    @param threshold: threshold for the relevance """
    return [word.text
            for phrase in response.result.phrases
            for word in phrase.words
            if word.tag == 1 and (
                (relevance == 1 and phrase.frequency < threshold1) or
                (relevance == 2 and threshold1 <= phrase.frequency < threshold2) or
                (relevance == 3 and phrase.frequency >= threshold2)
            )]


def _get_response(query_loader: iter, plain=False):
    """ Get the search result for each query yielded by query loader and return a dict with the results
        in the input format used in lcwp. """
    with client.insecure_web_channel("https://ngram.api.netspeak.org") as channel:
        stub = nspbg.NetspeakServiceStub(channel)
        for query in query_loader:
            q = re.sub(r'\[MASK]', "?", query['query'])
            try:
                response = stub.Search(nspb.SearchRequest(query=q, corpus="web-en"))
                if plain:
                    yield response
            except Exception as e:
                logging.exception(query, e)

            rel1 = _get_rel(response, 1)
            rel2 = _get_rel(response, 2)
            rel3 = _get_rel(response, 3)
            if (len(rel1) + len(rel2) + len(rel3)) == 0:
                continue

            yield {'id': query.get('id', None), 'query': query['query'], "rel1": rel1, "rel2": rel2, "rel3": rel3,
                   'length': query['length'], "position": query['position'],
                   "source": f"netspeak{query['length']}-{query['source']}",
                   "pos_class": query['pos_class']}


def _select_queries(input_file, query_length):
    """ load queries from a file and yield the lines that match the criteria (query length)"""
    qlen = {query_length}
    if query_length == 0:
        qlen = {3, 4, 5}
    for line in open(input_file, 'r'):
        line = json.loads(line)
        if line['length'] not in qlen:
            continue

        yield line


def compile_netspeak(output_path, output_name, input_file, query_length):
    query_loader = _select_queries(input_file, query_length)
    add_to_dataset(output_path, output_name, _get_response(query_loader), add_id=False)


def time_netspeak_queries(input_file):
    query_loader = _select_queries(input_file, 0)
    res = list(_get_response(query_loader, plain=True))
    return res


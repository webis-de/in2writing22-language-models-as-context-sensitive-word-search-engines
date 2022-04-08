"""
These functions create the evaluation results from the prediction files. You should call it from the cli.

The main method (compile_evaluation_results) refers fixed file names that are static for the files generated in the paper.
If you adapt the code to different models, you need to change these names.
"""

import json
from statistics import mean
from math import log2
from pathlib import Path


def _clean_ranking(ranking):
    ranking_set = set()
    ranking_duplicate = []
    for r in ranking:
        r = r.strip()
        r = r.lower()
        if not r.isalpha():
            continue
        if r not in ranking_set:
            ranking_duplicate.append(r)
            ranking_set.add(r)
    return ranking_duplicate


def _get_mean(l):
    if len(l) > 0:
        return mean(l)
    else:
        return 0


def _mrr(predictions, truth):
    """
    calculate the MRR of the true label:
        mean of (1/rank of first relevant document) over all queries
    :param predictions: a dict with the predicted labels
        {id: lcwp_example{"query", "rel1", "rel2", "rel3", "length", "position", "source", "pos_class"}}
    :param predictions: a dict with the true label in rel2
    """

    def _get_rr(qid, tr):
        try:
            ranking = predictions[qid].get('rel3', []) + predictions[qid].get('rel2', []) + predictions[qid].get('rel1',
                                                                                                                 [])
            ranking = _clean_ranking(ranking)
            return next((1 / (ind + 1)
                         for ind, pred in enumerate(ranking)
                         if pred == tr['rel2'][0]), 0)
        except Exception as e:

            print(e, qid, predictions[qid], tr)
            return 0

    return _get_mean([_get_rr(k, v) for k, v in truth.items()])


def _mean_ndcg(predictions, truth, top_n=30):
    def _get_ndcg(qid, tr):
        ranking = predictions[qid].get('rel3', []) + predictions[qid].get('rel2', []) + predictions[qid].get('rel1',
                                                                                                             [])[:top_n]
        # deduplicate
        ranking = _clean_ranking(ranking)

        r3 = {q: 3 for q in tr.get('rel3', [])}
        qrel = r3
        r2 = {q: 2 for q in tr.get('rel2', [])}
        qrel.update(r2)
        r1 = {q: 1 for q in tr.get('rel1', [])}
        qrel.update(r1)
        dcg = sum([qrel.get(r, 0) / log2(ind + 2) for ind, r in enumerate(ranking)])
        ideal_ranking = [3] * len(r3) + [2] * len(r1) + [1] * len(r1) + [0] * len(ranking)
        idcg = sum([rel / log2(ind + 2) for ind, rel in enumerate(ideal_ranking[:top_n])])
        return dcg / idcg

    return _get_mean([_get_ndcg(k, v) for k, v in truth.items()])


def _query_stats(query_dict):
    """
    :param query_dict: a dict
        {id: lcwp_example{"query", "rel1", "rel2", "rel3", "length", "position", "source", "pos_class"}}
    """
    return {
        'mean_all': _get_mean([len(v.get('rel1', [])) + len(v.get('rel2', [])) + len(v.get('rel3', [])) for v in query_dict.values()]),
        'mean_rel1': _get_mean([len(v.get('rel1', [])) for v in query_dict.values()]),
        'mean_rel2': _get_mean([len(v.get('rel2', [])) for v in query_dict.values()]),
        'mean_rel3': _get_mean([len(v.get('rel3', [])) for v in query_dict.values()])
    }


def evaluate(output_path: Path, test_dataset_path: Path, netspeak_results_path: Path,
             model_result_paths: list, qlen):
    # for each dataset
    predictions_netspeak = {item['id']: item for item in
                            [json.loads(line) for line in open(netspeak_results_path, 'r')]}
    truth_with_qlen = {item['id']: item for item in [json.loads(line) for line in open(test_dataset_path, 'r')] if
                       item['length'] == qlen}
    truth_only_netspeak = {k: v for k, v in truth_with_qlen.items() if k in predictions_netspeak}
    answer_ratio = len(truth_only_netspeak) / len(truth_with_qlen)  # count discards and output ratio

    mr = {}
    for mrp in model_result_paths:
        name = mrp.stem
        print(name)
        predictions_model_qlen_queries = {item['id']: item for item in [json.loads(line) for line in open(mrp, 'r')] if
                                          item['length'] == qlen}
        predictions_model_only_netspeak = {k: v for k, v in predictions_model_qlen_queries.items()
                                           if k in predictions_netspeak}

        mr[name] = {
            'mrr': {'all': _mrr(predictions_model_qlen_queries, truth_with_qlen),
                    'shared_query_mrr': _mrr(predictions_model_only_netspeak, truth_only_netspeak),
                    'position': {},
                    'pos_class': {}
                    },
            'ndcg': [_mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 1),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 2),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 3),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 4),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 5),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 10),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 15),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 20),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 25),
                     _mean_ndcg(predictions_model_only_netspeak, truth_only_netspeak, 30),
                     ]

        }
        # mask position
        for i in range(0, qlen):
            mr[name]['mrr']['position'][i] = _mrr(
                {k: v for k, v in predictions_model_only_netspeak.items() if v['position'] == i},
                {k: v for k, v in truth_only_netspeak.items() if v['position'] == i})
        # pos
        for i in ['VERB_AUX_M', 'VERB_AUX_S', 'NOUN_M', 'NOUN_S', 'DET_PRON', 'ADJ_ADV', 'STOP']:
            mr[name]['mrr']['pos_class'][i] = _mrr(
                {k: v for k, v in predictions_model_only_netspeak.items() if v['pos_class'] == i},
                {k: v for k, v in truth_only_netspeak.items() if v['pos_class'] == i})

    results = {
        'qlen': qlen,
        'total_query_count': len(truth_with_qlen),
        'shared_query_count': len(truth_only_netspeak),
        'shared_query_ratio': answer_ratio,
        'netspeak_query_stats': _query_stats(predictions_netspeak),  # results statistics for netspeak
        'model_results': mr,
        'netspeak_results': {
            'mrr': {'all': 0,
                    'shared_query_mrr': _mrr(predictions_netspeak, truth_only_netspeak),
                    'position': {},
                    'pos_class': {}
                    },
        }
    }

    for i in range(0, qlen):
        results['netspeak_results']['mrr']['position'][i] = _mrr(
            {k: v for k, v in predictions_netspeak.items() if v['position'] == i},
            {k: v for k, v in truth_only_netspeak.items() if v['position'] == i})
    # pos
    for i in ['VERB_AUX_M', 'VERB_AUX_S', 'NOUN_M', 'NOUN_S', 'DET_PRON', 'ADJ_ADV', 'STOP']:
        results['netspeak_results']['mrr']['pos_class'][i] = _mrr(
            {k: v for k, v in predictions_netspeak.items() if v['pos_class'] == i},
            {k: v for k, v in truth_only_netspeak.items() if v['pos_class'] == i})

    open(output_path / f'{test_dataset_path.stem}-qlen{qlen}.json', 'w').write(json.dumps(results))


def compile_evaluation_results(test_datasets_path: Path, predictions_path: Path, output_path: Path):
    evaluate(output_path, test_datasets_path / "cloth-test.ndjson", test_datasets_path / 'netspeak3-cloth-test.ndjson',
             [predictions_path / 'cloth-test-mlm.ndjson', predictions_path / 'cloth-test-mlm-finetuned.ndjson',
              predictions_path / 'cloth-test-bart.ndjson', predictions_path / 'cloth-test-bart-finetuned.ndjson',
              predictions_path / 'cloth-test-mlm-ft-netspeak.ndjson'], 3)
    evaluate(output_path, test_datasets_path / "cloth-test.ndjson", test_datasets_path / 'netspeak5-cloth-test.ndjson',
             [predictions_path / 'cloth-test-mlm.ndjson', predictions_path / 'cloth-test-mlm-finetuned.ndjson',
              predictions_path / 'cloth-test-bart.ndjson', predictions_path / 'cloth-test-bart-finetuned.ndjson',
              predictions_path / 'cloth-test-mlm-ft-netspeak.ndjson'], 5)
    evaluate(output_path, test_datasets_path / "wikitext-test.ndjson", test_datasets_path / 'netspeak3-wikitext-test.ndjson',
             [predictions_path / 'wikitext-test-mlm.ndjson', predictions_path / 'wikitext-test-mlm-finetuned.ndjson',
              predictions_path / 'wikitext-test-bart.ndjson', predictions_path / 'wikitext-test-bart-finetuned.ndjson',
              predictions_path / 'wikitext-test-mlm-ft-netspeak.ndjson'], 3)
    evaluate(output_path, test_datasets_path / "wikitext-test.ndjson", test_datasets_path / 'netspeak5-wikitext-test.ndjson',
             [predictions_path / 'wikitext-test-mlm.ndjson', predictions_path / 'wikitext-test-mlm-finetuned.ndjson',
              predictions_path / 'wikitext-test-bart.ndjson', predictions_path / 'wikitext-test-bart-finetuned.ndjson',
              predictions_path / 'wikitext-test-mlm-ft-netspeak.ndjson'], 5)

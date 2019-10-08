from __future__ import division
import json
import configargparse
from collections import Counter


def main():
    parser = configargparse.ArgumentParser(
        'Parameters for Graph Quality Measure')
    parser.add_argument('-file',
                        default='outputs/re_webnlg_f1_e100/valid_output.json',
                        type=str, help='path to files containing graphs')
    args = parser.parse_args()

    file = args.file
    with open(file) as f:
        data = json.load(f)

    corr_edge_rates = []
    for item in data:
        sent = item['sent']
        graph = item['graph']
        corr_edges = [edge['truth'] == edge['pred'] for edge in graph]
        corr_edge_rate = sum(corr_edges) / len(corr_edges)
        corr_edge_rates.append(corr_edge_rate)

        for edge in graph:
            if (edge['truth'] != edge['pred']) and (edge['truth'] == '<unk>'):
                continue
                import pdb;
                pdb.set_trace()
    cnt = Counter(corr_edge_rates)
    sorted_cnt = sorted(list(cnt.items()))

    corr_graph = cnt[1.0] / len(data)
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()

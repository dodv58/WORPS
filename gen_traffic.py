import networkx as nx
from common import *
import numpy as np
import argparse
import json
import time

np.random.seed(11)
random.seed(11)


def new_session(net, id, args):
    group_size = random.randint(args.group_size_min, args.group_size_max)
    nodes = random.sample(list(net.nodes), group_size)
    random.shuffle(nodes)
    return Session(id, nodes[0], nodes[1:], random.uniform(args.bw_min, args.bw_max), random.randint(args.delta_min, args.delta_max))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='run index/dataset id', required=True)
    parser.add_argument('--topo', required=True, help='nsfnet / geant2 / bics')
    parser.add_argument('--min_capacity', help='min link capacity', type=float, default=10)
    parser.add_argument('--max_capacity', help='max link capacity', type=float, default=15)
    parser.add_argument('--demand_size', help='number of session in a demand', type=int, default=100)
    parser.add_argument('--n_demands', help='number of demand', type=int, default=1)
    parser.add_argument('--bw_min', type=float, default=0.04)
    parser.add_argument('--bw_max', type=float, default=0.1)
    parser.add_argument('--delta_min', type=int, default=0)
    parser.add_argument('--delta_max', type=int, default=3)
    parser.add_argument('--group_size_min', type=int, default=5)
    parser.add_argument('--group_size_max', type=int, default=8)

    args = parser.parse_args()
    dir = f'datasets/{args.run}'
    Path(dir).mkdir(parents=True, exist_ok=True)

    with open(dir + '/meta.txt', 'w') as f:
        meta = dict()
        for k, v in vars(args).items():
            meta[k] = v
        meta['w_min'] = w_min
        meta['w_max'] = w_max
        json.dump(meta, f, indent=4)

    net = nx.read_gml(f'topos/{args.topo}.gml')
    net = nx.convert_node_labels_to_integers(net)

    for m,n in net.edges:
        net[m][n]['capacity'] = random.uniform(args.min_capacity, args.max_capacity)

    nx.write_gml(net, dir + '/topo.gml')
    # gen traffic demands
    demands = [[new_session(net, i, args) for i in range(args.demand_size)] for _ in range(args.n_demands)]

    with open(dir + '/traffic.txt', 'w') as f:
        json.dump(demands, f, indent=2)

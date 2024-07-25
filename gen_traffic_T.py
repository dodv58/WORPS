import random

import networkx as nx
from common import *
import numpy as np
import argparse
import json
import time

np.random.seed(11)
random.seed(11)


def new_session(net, id, source = None, type = 'normal', args = None):
    nodes = list(net.nodes)
    group_size = random.randint(args.group_size_min, args.group_size_max)
    if type == 'normal':
        delta = random.randint(args.normal_delta_min, args.normal_delta_max)
    else:
        delta = random.randint(args.critical_delta_min, args.critical_delta_max)

    if source in nodes:
        nodes = [m for m in nodes if m != source]
        dests = random.sample(nodes, group_size - 1)
        session = Session(id, source, dests, random.uniform(args.bw_min, args.bw_max), delta)
    else:
        nodes = random.sample(nodes, group_size)
        random.shuffle(nodes)
        session = Session(id, nodes[0], nodes[1:], random.uniform(args.bw_min, args.bw_max), delta)

    return session

def new_demand(net, args):
    demand = []
    nodes = [m for m in net.nodes]
    n_new_sessions = [3] + [2] * (args.T - 1)
    for t in range(args.T):
        old_sessions = demand[t-1] if t > 0 else []

        sources = nodes * (n_new_sessions[t]//len(net.nodes))
        sources += random.sample(nodes, n_new_sessions[t] - len(sources))
        random.shuffle(sources)

        critical_session_count = int(n_new_sessions[t]*args.critical)
        types = ['critical'] * critical_session_count + ['normal'] * (n_new_sessions[t]-critical_session_count)

        demand.append(old_sessions + [new_session(net, i + len(old_sessions), sources[i], type=types[i], args=args) for i in range(n_new_sessions[t])])
    return demand


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='run index/dataset id', required=True)
    parser.add_argument('--topo', required=True, help='nsfnet / geant2 / bics')
    parser.add_argument('--min_capacity', help='min link capacity', type=float, default=10)
    parser.add_argument('--max_capacity', help='max link capacity', type=float, default=15)
    parser.add_argument('--demand_size_0', help='number of session in a demand', type=int, default=3)
    parser.add_argument('--n_demands', help='number of demand', type=int, default=1)
    parser.add_argument('--bw_min', type=float, default=0.1)
    parser.add_argument('--bw_max', type=float, default=0.5)
    parser.add_argument('--critical_delta_min', type=int, default=0)
    parser.add_argument('--critical_delta_max', type=int, default=1)
    parser.add_argument('--normal_delta_min', type=int, default=2)
    parser.add_argument('--normal_delta_max', type=int, default=3)
    parser.add_argument('--critical', type=float, default=0.1, help='proportion of critical traffic in a demand')
    parser.add_argument('--group_size_min', type=int, default=5)
    parser.add_argument('--group_size_max', type=int, default=6)
    parser.add_argument('-T', type=int, default=5)

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
    demands = [new_demand(net, args) for _ in range(args.n_demands)]

    with open(dir + '/traffic.txt', 'w') as f:
        json.dump(demands, f, indent=2)

import networkx as nx
from common import *
import numpy as np
import argparse
import json
import time
from wodba import wodba
from worps_solver import worps_solver

np.random.seed(1)
random.seed(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', help='sum_cost/max_load', default='sum_cost')
    parser.add_argument('--dataset', help='run index', required=True)
    parser.add_argument('--log', default=0, type=int)

    args = parser.parse_args()
    wandb_run = wandb.init(project='wodba', mode='disabled' if not args.log else 'online')
    wandb.config['dataset'] = args.dataset

    dir = f'experiments/{args.dataset}'
    Path(dir).mkdir(parents=True, exist_ok=True)
    net = nx.read_gml(f'datasets/{args.dataset}/topo.gml')
    net = nx.convert_node_labels_to_integers(net)

    with open(f'datasets/{args.dataset}/traffic.txt') as f:
        demands = json.load(f)
        demands = [[Session(*s) for s in traffic] for traffic in demands]


    traffic = demands[0]
    obj, weight, RP, route = worps_solver(net, traffic, dir)
    for s in traffic:
        print("========", s)
        print("route: ", set(route[s.id]))
        print("RP: ", [m for m,j in RP if j == s.id])


    apply_network_setting(net, traffic, weight, RP)


    wodba(net, demands[0])


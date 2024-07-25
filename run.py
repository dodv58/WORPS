import networkx as nx
from common import *
import numpy as np
import argparse
import json
import time
from worps import worps_ga, worps_pimsm, worps_hopcount
from worps_solver import worps_solver
from multiprocessing import Pool

np.random.seed(1)
random.seed(1)

def run(net, demands, solver, dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

    if solver in ['optimal']:
        traffic = demands[0]
        obj, weight, RP, route = worps_solver(net, traffic, dir)
        for s in traffic:
            print("========", s)
            print("route: ", set(route[s.id]))
            print("RP: ", [m for m, j in RP if j == s.id])

        apply_network_setting(net, traffic, weight, RP)

    if solver in ['all', 'ga']:
        f1 = open(f"{dir}/ga50.txt", "w+")
        f2 = open(f"{dir}/ga100.txt", "w+")
        ga50_results = []
        ga100_results = []
        i = 0
        for traffic in demands:
            print("traffic index: ", i)
            results = worps_ga(net, traffic, max_iters=[50, 100])
            result50, result100 = list(results)
            cost, failed_rate = apply_network_setting(net, traffic, result50.chromosome, result50.rps)
            ga50_results.append({
                "cost": cost,
                "failed_sessions": failed_rate,
                "weights": result50.chromosome,
                "selected_rps": result50.rps
            })

            cost, failed_rate = apply_network_setting(net, traffic, result100.chromosome, result100.rps)
            ga100_results.append({
                "cost": cost,
                "failed_sessions": failed_rate,
                "weights": result100.chromosome,
                "selected_rps": result100.rps
            })
            i += 1

        json.dump(ga50_results, f1)
        json.dump(ga100_results, f2)

    if solver in ['all', 'pim']:
        with open(f"{dir}/pimsm.txt", "w+") as f:
            results = []
            for traffic in demands:
                weights, selected_rps, cost, failed_rate = worps_pimsm(net, traffic)
                results.append({
                    "weights": weights,
                    "cost": cost,
                    "selected_rps": selected_rps,
                    "failed_sessions": failed_rate
                })
            json.dump(results, f)
    if solver in ['all', 'hopcount']:
        with open(f"{dir}/hopcount.txt", "w+") as f:
            results = []
            for traffic in demands:
                weights, selected_rps, cost, failed_rate = worps_hopcount(net, traffic)
                results.append({
                    "weights": weights,
                    "cost": cost,
                    "selected_rps": selected_rps,
                    "failed_sessions": failed_rate
                })
            json.dump(results, f)
    return f"{dir} {solver}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', help='sum_cost/max_load', default='sum_cost')
    parser.add_argument('--dataset', help='dataset', required=True, nargs='*')
    parser.add_argument('--log', default=0, type=int)
    parser.add_argument('--solver', help= "optimal/ga/all", default="all")
    parser.add_argument('--run', default="01")

    args = parser.parse_args()
    # wandb_run = wandb.init(project='wodba', mode='disabled' if not args.log else 'online')
    # wandb.config['dataset'] = args.dataset
    with Pool(len(args.dataset)) as p:
        params = []
        for dataset in args.dataset:
            dir = f'experiments/{args.run}/{dataset}'
            net = nx.read_gml(f'datasets/{dataset}/topo.gml')
            net = nx.convert_node_labels_to_integers(net)
            with open(f'datasets/{dataset}/traffic.txt') as f:
                demands = json.load(f)
                demands = [[Session(*s) for s in traffic] for traffic in demands]
            params.append((net, demands, args.solver, dir))
        print(p.starmap(run, params))









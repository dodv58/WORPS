import networkx as nx
import numpy as np
import rustworkx as rx
from common import *
import json

def route_from_path(p):
    return [edge_index_map[p[i]-1, p[i]] for i in range(1, len(p))]

dataset = "025"
net = nx.read_gml(f'datasets/{dataset}/topo.gml')
net = nx.convert_node_labels_to_integers(net)
net = rx.networkx_converter(net, keep_attributes=True)
capacities = np.array([e['capacity'] for e in net.edges()])

with open(f'datasets/{dataset}/traffic.txt') as f:
    demands = json.load(f)
    demands = [[Session(*s) for s in traffic] for traffic in demands]
    traffic = demands[0]

n_edges = len(net.edges())
n_nodes = len(net.nodes())
n_traffic = len(traffic)

edge_index_map = np.full((n_nodes, n_nodes), -1, dtype=int)
for index, (m,n) in enumerate(net.edge_list()):
    edge_index_map[m, n] = index

D = np.array([s.bw for s in traffic])


W = np.ones(n_edges)
for e in net.edges():
    e['w'] = 1

R = np.zeros((n_edges, n_traffic))
shortest_paths = rx.all_pairs_dijkstra_shortest_paths(net, edge_cost_fn=lambda e: e['w'])
shortest_routes = {m: {n: route_from_path(shortest_paths[m][n]) for n in shortest_paths[m].keys()} for m in net.node_indices()}

for j, s in enumerate(traffic):
    for d in s.ds:
        R[shortest_routes[s.s][d], j] = 1

allocated_bw = R @ D
print(allocated_bw/capacities)









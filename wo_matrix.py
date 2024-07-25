import numpy as np
from common import *
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


dataset = "025"
net = nx.read_gml(f'datasets/{dataset}/topo.gml')
net = nx.convert_node_labels_to_integers(net)
n_nodes = len(net.nodes)
n_edges = len(net.edges)
node_indices = [i for i in range(n_nodes)]

capacities = np.zeros(n_edges)
weight_matrix = np.zeros((n_nodes, n_nodes))
edge_index_map = np.full((n_nodes, n_nodes), -1, dtype=int)
for i, (m,n) in enumerate(net.edges):
    edge_index_map[m,n] = i
    capacities[i] = net[m][n]['capacity']
    weight_matrix[m,n] = 1/net[m][n]['capacity']


with open(f'datasets/{dataset}/traffic.txt') as f:
    demands = json.load(f)
    demands = [[Session(*s) for s in traffic] for traffic in demands]
    traffic = demands[0]
n_traffic = len(traffic)
traffic_bw = np.array([s.bw for s in traffic])

traffic_sources = np.zeros((n_traffic, n_nodes), dtype=bool)
traffic_sources[range(n_traffic), [traffic[j].s for j in range(n_traffic)]] = True
traffic_dests = np.zeros((n_traffic, n_nodes), dtype=bool)
for j in range(n_traffic):
    traffic_dests[j, traffic[j].ds] = True


def network_cost(weight_matrix, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw):
    nodes = range(len(weight_matrix))
    n_traffic = len(traffic_bw)
    n_edges = len(e_cap)

    shortest_paths = np.zeros((n_nodes, n_nodes, n_edges), dtype=int)
    shortest_path_trees = np.zeros((n_edges, n_traffic), dtype=int)
    graph = csr_matrix(weight_matrix)
    _, predecessors = dijkstra(csgraph=graph, directed=True, return_predecessors=True)

    for m in nodes:
        for n in nodes:
            if m == n:
                continue
            u, v = n, n
            while v != m:
                u = predecessors[m][v]
                shortest_paths[m,n,e_index_map[u,v]] = 1
                v = u

    for j in range(n_traffic):
        shortest_path_trees[:, j] = shortest_paths[traffic_sources[j], traffic_dests[j]].sum(axis=0)

    shortest_path_trees[shortest_path_trees > 1] = 1

    loads = shortest_path_trees @ traffic_bw / e_cap
    return loads

def get_mdt(weight_matrix, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw):
    # weight_matrix: n_nodes x n_nodes
    # e_index_map: n_nodes x n_nodes -> e_index
    # e_cap: n_edges
    # traffic_sources: n_traffic x n_nodes
    # traffic_dests: n_traffic x n_nodes
    # traffic_bw: n_traffic

    nodes = range(len(weight_matrix))
    n_traffic = len(traffic_bw)
    n_edges = len(e_cap)

    shortest_paths = np.zeros((n_nodes, n_nodes, n_edges), dtype=int)
    shortest_path_trees = np.zeros((n_edges, n_traffic), dtype=int)
    graph = csr_matrix(weight_matrix)
    _, predecessors = dijkstra(csgraph=graph, directed=True, return_predecessors=True)

    for m in nodes:
        for n in nodes:
            if m == n:
                continue
            u, v = n, n
            while v != m:
                u = predecessors[m][v]
                shortest_paths[m,n,e_index_map[u,v]] = 1
                v = u

    for j in range(n_traffic):
        shortest_path_trees[:, j] = shortest_paths[traffic_sources[j], traffic_dests[j]].sum(axis=0)

    shortest_path_trees[shortest_path_trees > 1] = 1

    return shortest_path_trees

cost = network_cost(weight_matrix=weight_matrix,
                    e_index_map=edge_index_map,
                    e_cap=capacities,
                    traffic_sources=traffic_sources,
                    traffic_dests=traffic_dests,
                    traffic_bw=traffic_bw)
print(max(cost))
        

def calculate_fitness(net, traffic, weights):
    net = copy.deepcopy(net)
    for m, n in net.edges:
        net[m][n]['w'] = weights[m,n]

    nx.set_edge_attributes(net, 0, 'allocated_bw')
    all_shortest_path = dict(nx.all_pairs_dijkstra_path(net, weight='w'))

    for s in traffic:
        tree = set()
        routes = [links_in_path(all_shortest_path[s.s][d]) for d in s.ds]

        for route in routes:
            tree.update(route)
        for m,n in tree:
            net[m][n]['allocated_bw'] += s.bw


    return [net[m][n]['allocated_bw']/net[m][n]['capacity'] for m,n in net.edges]
res = calculate_fitness(net, traffic, weight_matrix)
print(max(res))

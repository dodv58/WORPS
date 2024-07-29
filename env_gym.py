from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from common import *
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from gymnasium.wrappers import FlattenObservation
from itertools import product

def get_edge_adjacency(e_index_map, n_edges):
    n_nodes = e_index_map.shape[0]
    edge_adjacency = np.zeros((n_edges, n_edges), dtype=int)
    for m in range(n_nodes):
        in_edges = e_index_map[e_index_map[:, m] > -1, m]
        out_edges = e_index_map[m, e_index_map[m, :] > -1]
        arc_list = product(in_edges, out_edges)
        arc_sources, arc_dests = zip(*arc_list)
        edge_adjacency[arc_sources, arc_dests] = 1
    return edge_adjacency
    # return np.array(edge_adjacency.nonzero())

class Network(gym.Env):
    def __init__(self, net, demands, max_step=200, early_stop=True):
        self.n_nodes = len(net.nodes)
        self.n_edges = len(net.edges)
        self.nodes = list(range(self.n_nodes))

        self.e_cap = np.array([net[m][n]['capacity'] for m, n in net.edges])
        self.max_e_cap = self.e_cap.max()
        self.e_sources = np.array([m for m, n in net.edges], dtype=int)
        self.e_dests = np.array([n for m, n in net.edges], dtype=int)
        self.edges = [(m,n) for m, n in net.edges]
        self.e_index_map = np.full((self.n_nodes, self.n_nodes), -1, dtype=int)
        self.e_index_map[self.e_sources, self.e_dests] = np.arange(self.n_edges)
        self.edge_adjacency = get_edge_adjacency(e_index_map=self.e_index_map, n_edges=self.n_edges)
        

        self.demands = [sorted(traffic, key=lambda x: x.bw * len(x.ds), reverse=True) for traffic in demands]
        self.traffic_index = 0

        self.max_step = max_step
        self.max_weight = 100
        self.early_stop = early_stop

        self.n_features = 6
        self.extra_features = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_edges * self.n_features + self.extra_features,), dtype=np.float32)

        self.action_space = spaces.Discrete(self.n_edges*2 + 1)

    def _get_info(self):
        return {"initial_network_cost": self.initial_network_cost,
                "increasing_steps": self.increasing_steps/(self.step_count + 1)}

    def _get_obs(self):
        max_link_load = self.loads >= self.loads.max()
        overload_link = self.loads >= 1.
        return np.concatenate((
            self.weights/self.max_weight,
            self.loads,
            self.edge_betweenness,
            max_link_load.astype(np.float32),
            overload_link.astype(np.float32),
            self.e_cap / self.max_e_cap,
            np.array([self.step_count/self.max_step], dtype=np.float32)
        ))

    def reset(
        self,
        seed: int = None,
        options = None,
    ):
        super().reset(seed=seed)
        self.weights = np.full(self.n_edges, 20, dtype=np.float32)

        self.traffic_index += 1
        if self.traffic_index >= len(self.demands):
            self.traffic_index = 0
        self.traffic = self.demands[self.traffic_index]
        self.n_traffic = len(self.traffic)
        self.traffic_bw = np.array([s.bw for s in self.traffic])
        self.traffic_sources = np.zeros((self.n_traffic, self.n_nodes), dtype=bool)
        self.traffic_sources[range(self.n_traffic), [self.traffic[j].s for j in range(self.n_traffic)]] = True
        self.traffic_dests = np.zeros((self.n_traffic, self.n_nodes), dtype=bool)
        for j in range(self.n_traffic):
            self.traffic_dests[j, self.traffic[j].ds] = True

        mdt, all_shortest_paths = get_mdt(weights=self.weights,
                                          e_sources=self.e_sources,
                                          e_dests=self.e_dests,
                                          nodes=self.nodes,
                                            e_index_map=self.e_index_map,
                                            e_cap=self.e_cap,
                                            traffic_sources=self.traffic_sources,
                                            traffic_dests=self.traffic_dests,
                                            traffic_bw=self.traffic_bw)

        self.loads = mdt @ self.traffic_bw / self.e_cap
        self.initial_network_cost = self.loads.max()
        self.increasing_steps = 0
        self.edge_betweenness = all_shortest_paths.sum(axis=(0, 1)) / (self.n_nodes ** 2)

        self.step_count = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def get_weights(self):
        return self.weights
    
    def get_network_cost(self):
        return self.loads.max()
    
    def step(self, action):
        self.step_count += 1
        if action < self.n_edges:
            self.weights[action] += 1
        elif action < self.n_edges*2:
            e_index = action - self.n_edges
            self.weights[e_index] = max(1, self.weights[e_index] - 1)
        else:
            # keep the weights unchanged
            if self.early_stop:
                return self._get_obs(), 0, True, False, self._get_info()
            else:
                return self._get_obs(), 0, self.step_count >= self.max_step, False, self._get_info()
                

        mdt, all_shortest_paths = get_mdt(weights=self.weights,
                                          e_sources=self.e_sources,
                                          e_dests=self.e_dests,
                                          nodes=self.nodes,
                                          e_index_map=self.e_index_map,
                                          e_cap=self.e_cap,
                                          traffic_sources=self.traffic_sources,
                                          traffic_dests=self.traffic_dests,
                                          traffic_bw=self.traffic_bw)
        current_loads = self.loads
        self.loads = mdt @ self.traffic_bw / self.e_cap
        self.edge_betweenness = all_shortest_paths.sum(axis=(0,1)) / (self.n_nodes**2)

        reward = current_loads.max() - self.loads.max()
        if reward > 0:
            self.increasing_steps += 1

        terminated = self.step_count >= self.max_step
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info


def get_all_pair_shortest_paths(weight_matrix, e_index_map, e_cap):
    # weight_matrix: n_nodes x n_nodes
    # e_index_map: n_nodes x n_nodes -> e_index
    # e_cap: n_edges
    # traffic_sources: n_traffic x n_nodes
    # traffic_dests: n_traffic x n_nodes

    n_nodes = len(weight_matrix)
    nodes = list(range(n_nodes))
    n_edges = len(e_cap)

    shortest_paths = np.zeros((n_nodes, n_nodes, n_edges), dtype=int)
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

    return shortest_paths

def get_mdt(weights, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw):
    # e_index_map: n_nodes x n_nodes -> e_index
    # e_cap: n_edges
    # traffic_sources: n_traffic x n_nodes
    # traffic_dests: n_traffic x n_nodes
    # traffic_bw: n_traffic

    n_nodes = len(nodes)
    n_traffic = len(traffic_bw)
    n_edges = len(e_cap)

    shortest_paths = np.zeros((n_nodes, n_nodes, n_edges), dtype=int)
    shortest_path_trees = np.zeros((n_edges, n_traffic), dtype=int)
    graph = csr_matrix((weights, (e_sources, e_dests)), shape=(n_nodes, n_nodes))
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

    return shortest_path_trees, shortest_paths

def rps(all_pair_shortest_paths, e_cap, sources, dests, traffic_bw, deltas):
    n_traffic = len(traffic_bw)
    nodes = len(range(all_pair_shortest_paths.shape[0]))
    selected_rps = np.full(n_traffic, -1, dtype=int)
    mdt = np.zeros((all_pair_shortest_paths.shape[-1], n_traffic), dtype=int)
    for j in range(n_traffic):
        current_loads = mdt @ traffic_bw / e_cap
        for m in nodes:
            cost, tree = attempt_rp(all_pair_shortest_paths, current_loads, sources[j], dests[j], traffic_bw[j], m, deltas[j])
            tmp_rps.append((m, cost, tree))
        selected_rp, cost, tree = min(tmp_rps, key=lambda x: x[1])
        selected_rps[j] = selected_rp
        mdt[tree, j] = 1
    return mdt, selected_rps

def attempt_rp(all_pair_shortest_paths, current_loads, s, dests, bw, m, delta):
    n_edges = current_loads.shape[0]
    routes_from_rp = all_pair_shortest_paths[m, dests]
    edge_list = all_pair_shortest_paths[s, m].squeeze() + routes_from_rp.sum(axis=0)
    edge_list[edge_list > 1] = 1
    tree_size = edge_list.sum()
    route_lengths = routes_from_rp.sum(axis=1)

    is_delay_variation_satisfied = (route_lengths.max() - route_lengths.min() <= delta)
    is_using_max_load_link = (edge_list[current_loads.argmax()] == 1)
    
    # return bw_consumption, is_using_max_load_link, is_delay_variation_satisfied
    return tree_size/n_edges + is_using_max_load_link*2/n_edges + is_delay_variation_satisfied * 10, edge_list


class CatObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n_edges = env.observation_space["weight"].shape[0]
        self.observation_space = spaces.Box(shape=(n_edges, 6), low=-np.inf, high=np.inf)

    def observation(self, obs):
        return np.stack(
            (
                obs["weight"],
                obs["loads"],
                obs["edge_betweenness"],
                obs["max_link_load"],
                obs["overload_link"],
                obs["e_cap"]
            ),
            axis=-1)


def make_env(dataset, env_max_step, early_stop):
    def inner():
        net = nx.read_gml(f'datasets/{dataset}/topo.gml')
        net = nx.convert_node_labels_to_integers(net)

        with open(f'datasets/{dataset}/traffic.txt') as f:
            demands = json.load(f)
            demands = [[Session(*s) for s in traffic] for traffic in demands]
            # traffic = demands[0]

        env = Network(net=net, demands = demands, max_step=env_max_step, early_stop=early_stop)
        # env = CatObservation(env)
        # env = FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return inner

if __name__ == "__main__":
    dataset = "025"
    net = nx.read_gml(f'datasets/{dataset}/topo.gml')
    net = nx.convert_node_labels_to_integers(net)

    with open(f'datasets/{dataset}/traffic.txt') as f:
        demands = json.load(f)
        demands = [[Session(*s) for s in traffic] for traffic in demands]

    env =  Network(net=net, demands=demands)
    # env = CatObservation(env)
    # env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    print(env.edges[0])
    es, ed = env.edges[0]
    e_in =  [(m,n) for m,n in env.edges if n == es]
    print("e_in: ", e_in)
    e_in_index = [env.e_index_map[m,n] for m,n in e_in] 
    print("e in index: ", e_in_index)
    e_out = [(m,n) for m,n in env.edges if m == ed]
    print("e_out: ", e_out)
    e_out_index = [env.e_index_map[m,n] for m,n in e_out] 
    print("e_out_index: ", e_out_index)
    print("out: ", env.edge_adjacency[0].nonzero())
    print("in: ", env.edge_adjacency[:,0].nonzero())
    print(env.edge_adjacency.shape)
    
    obs, info = env.reset()
    for _ in range(4):
        obs, r, terminated, _, info = env.step(1)
    # print(obs.shape)
    # print(obs)
        print(terminated)
        print(info)

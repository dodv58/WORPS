from collections import namedtuple
import copy
import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import random
import wandb

w_min, w_max = 1, 100


Individual = namedtuple('Individual', 'chromosome fitness bsa')
Session = namedtuple('Session', 'id s ds bw delta') # multicast

def downlinks(route, m):
    i = 0
    while i < len(route) and route[i][0] != m:
        i += 1
    if i == len(route):
        raise ValueError(f"{m} is not in {route}")
    return route[i:]

class SPT:
    def __init__(self, net, session, all_pair_shortest_paths, buffer_allocation):
        self.net = net
        self.root = session.s
        self.leaves = session.ds
        self.session = session
        self.edges = np.zeros((net.number_of_nodes(), net.number_of_nodes()))
        self.tree_nodes = set()
        self.cflows = {} # route for each destination
        self.links = {}  # conceptual flows pass through link mn
        for d in self.leaves:
            path = all_pair_shortest_paths[session.s][d]
            route = links_in_path(path)
            self.tree_nodes.update(path)
            self.cflows[d] = route
            for m, n in route:
                self.edges[m, n] = 1
                if (m,n) in self.links:
                    self.links[(m,n)].append(d)
                else:
                    self.links[(m,n)] = [d]

        for k, v in buffer_allocation.items():
            links = downlinks(self.cflows[k], v)
            for m,n in links:
                self.edges[m,n] = min(self.edges[m,n] + 1, len(self.links[(m,n)]))


    def bw_consumption(self):
        return self.edges.sum() * self.session.bw

    def bfs(self, m=None):
        if m is None: m = self.root
        nodes = [m]
        i = 0
        while True:
            nodes += self.childs(nodes[i])
            i += 1
            if i >= len(nodes):
                return nodes

    def edges_list(self):
        return np.transpose(self.edges.nonzero())

    def redundant_ratio(self):
        return self.edges.sum()/np.count_nonzero(self.edges)

    def draw(self, ax):
        layout = nx.kamada_kawai_layout(self.net)
        nx.draw(self.net, pos=layout, ax=ax, with_labels=True, node_color='white', edgecolors='k')
        nx.draw_networkx_nodes(self.net, pos=layout, ax=ax, nodelist=[self.root], node_shape='s', linewidths=5, node_color='r')
        nx.draw_networkx_nodes(self.net, pos=layout, ax=ax, nodelist=self.leaves, node_shape='d', linewidths=5, node_color='g')
        nx.draw_networkx_nodes(self.net, pos=layout, ax=ax, nodelist=self.branch_nodes, node_color='blue')

        nx.draw_networkx_edges(self.net, pos=layout, ax=ax, edgelist=np.transpose(self.edges.nonzero()), edge_color='r', width=2)
        edge_labels = {(m, n): int(self.edges[m, n]) for m, n in self.net.edges if self.edges[m, n] > 0}
        nx.draw_networkx_edge_labels(self.net, ax=ax, pos=layout, edge_labels=edge_labels)
        # nx.draw_networkx_labels(self.net, ax=ax, pos=layout, verticalalignment='bottom', horizontalalignment='left',
        #                         labels={m: (len(self.net.nodes[m]['stored_session']), self.net.nodes[m]['capacity']) for m in self.net.nodes})
        nx.draw_networkx_labels(self.net, ax=ax, pos=layout, verticalalignment='bottom', horizontalalignment='left', labels=self.branch_count)

def links_in_path(p):
    return [(p[i-1], p[i]) for i in range(1, len(p))]

def link_cost(c, l):
    # c: capacity, l: load
    u = l/c
    cost = max(u, 3*u-2/3, 10*u-16/3, 70*u-178/3, 500*u-1468/3, 5000*u-19468/3)
    return cost

def link_load_level(load):
    if load < 1/3:
        return 0
    elif load < 2/3:
        return 1
    elif load < 9/10:
        return 2
    elif load < 1:
        return 3
    elif load < 4/3:
        return 4
    else:
        return 5

available_node = lambda net, m: len(net.nodes[m]['stored_session']) < net.nodes[m]['capacity']

def get_network_cost(net, traffic, weight=None, metric='sum_cost'):
    net_copy = copy.deepcopy(net)
    if isinstance(weight, list):
        for i, (m,n) in enumerate(net_copy.edges):
            net_copy[m][n]['w'] = weight[i]
    elif type(weight) == np.ndarray:
        for m, n in net_copy.edges:
            net_copy[m][n]['w'] = weight[m][n]
    elif weight == 'uniform':
        for m, n in net_copy.edges:
            net_copy[m][n]['w'] = np.random.uniform(w_min, w_max)
    elif weight == 'equal':
        nx.set_edge_attributes(net_copy, 1, 'w')
    elif weight == 'inverse_capacity':
        for m, n in net_copy.edges:
            net_copy[m][n]['w'] = 1/net_copy[m][n]['capacity']
    else:
        pass     # use current network's weights if weight = None

    nx.set_edge_attributes(net_copy, 0, 'allocated_bw')
    all_shortest_path = dict(nx.all_pairs_dijkstra_path(net_copy, weight='w'))

    for m in net_copy.nodes:
        net_copy.nodes[m]['route'] = np.zeros((len(traffic)))
        net_copy.nodes[m]['stored_session'] = []
    for m,n in net_copy.edges:
        net_copy[m][n]['route'] = np.zeros((len(traffic)))

    MDTs = []
    for i, x in enumerate(traffic):
        paths = [all_shortest_path[x.s][d] for d in x.ds]
        MDTs.append(SPT(net_copy, x, paths))
        for p in paths:
            for m,n in links_in_path(p):
                net_copy.nodes[m]['route'][i] += 1
                net_copy[m][n]['route'][i] += 1

    # assign_branch_state_nodes(net_copy, MDTs)

    for m,n in net_copy.edges:
        net_copy[m][n]['allocated_bw'] = sum([traffic[session_index].bw * net_copy[m][n]['route'][session_index] for session_index in net_copy[m][n]['route'].nonzero()[0]])


    if metric == 'max_load':
        res = [net_copy[m][n]['allocated_bw']/net_copy[m][n]['capacity'] for m,n in net_copy.edges]
    elif metric == 'sum_cost':
        res = [link_cost(net_copy[m][n]['capacity'], net_copy[m][n]['allocated_bw']) for m, n in net_copy.edges]
    else:
        raise Exception()
    return res


def bsa(net, traffic, weights):
    net_copy = copy.deepcopy(net)
    for i, (m,n) in enumerate(net_copy.edges):
        net_copy[m][n]['w'] = weights[i]
        net_copy[m][n]['transmission'] = np.zeros((len(traffic)))

    nx.set_edge_attributes(net_copy, 0, 'allocated_bw')
    all_shortest_path = dict(nx.all_pairs_dijkstra_path(net_copy, weight='w'))

    for m in net_copy.nodes:
        net_copy.nodes[m]['stored_session'] = []

    # branch state node assignment
    assigned_MDTs = [0 for _ in traffic]
    MDTs = []
    transmission = {(m,n): np.zeros((len(traffic))) for m,n in net_copy.edges}
    for s in traffic:
        MDTs.append(SPT(net_copy, s, [all_shortest_path[s.s][d] for d in s.ds]))
        for m, n in MDTs[-1].edges_list():
            transmission[(m,n)][s.id] = MDTs[-1].edges[m, n]

    # while sum(assigned_MDTs) < len(traffic):
    #     max_load = 0
    #     session = None
    #     for m,n in net_copy.edges:
    #         sessions_on_link = [session_index for session_index in transmission[(m,n)].nonzero()[0]]
    #         allocated_bw = sum([traffic[j].bw * transmission[(m,n)][j] for j in sessions_on_link])
    #         if allocated_bw/net_copy[m][n]['capacity'] > max_load:
    #             max_possible_bw_reduction = 0
    #             for j in sessions_on_link:
    #                 if assigned_MDTs[j] == 0 and MDTs[j].possible_bw_reduction(m,n) > max_possible_bw_reduction:
    #                     session = j
    #                     max_possible_bw_reduction = max_possible_bw_reduction
    #
    #     if session is None:
    #         for s in traffic:
    #             if assigned_MDTs[s.id] == 0:
    #                 session = s.id
    #                 break
    #     tree = MDTs[session]
    #     if len(net_copy.nodes[tree.root]['stored_session']) < net_copy.nodes[tree.root]['capacity']:
    #         avail_branch_nodes = [u for u in tree.branch_nodes if available_node(net_copy, u)]
    #         for u in avail_branch_nodes:
    #             net_copy.nodes[u]['stored_session'].append(tree.session.id)
    #     assigned_MDTs[session] = 1
    #     for m, n in tree.edges_list():
    #         transmission[(m,n)][tree.session.id] = tree.edges[m, n]

    #################
    MDTs = sorted(MDTs, key=lambda t: t.sorting_value(), reverse=True)
    for tree in MDTs:
        if len(net_copy.nodes[tree.root]['stored_session']) < net_copy.nodes[tree.root]['capacity']:
            avail_branch_nodes = [u for u in tree.branch_nodes if available_node(net_copy, u)]
            for u in avail_branch_nodes:
                net_copy.nodes[u]['stored_session'].append(tree.session.id)
    #################

    MDTs = []
    for s in traffic:
        t = SPT(net_copy, s, [all_shortest_path[s.s][d] for d in s.ds])
        MDTs.append(t)
        for m,n in t.edges_list():
            net_copy[m][n]['transmission'][s.id] = t.edges[m,n]

    for m,n in net_copy.edges:
        net_copy[m][n]['allocated_bw'] = sum([traffic[session_index].bw * net_copy[m][n]['transmission'][session_index] for session_index in net_copy[m][n]['transmission'].nonzero()[0]])

    cost = sum([link_cost(net_copy[m][n]['capacity'], net_copy[m][n]['allocated_bw']) for m,n in net_copy.edges])
    branch_states = [(m,j) for m in net_copy.nodes for j in net_copy.nodes[m]['stored_session']]
    return cost, branch_states


def apply_network_setting(net, traffic, weights, buffer_allocation):
    net_copy = copy.deepcopy(net)

    for m in net_copy.nodes:
        net_copy.nodes[m]['buffer'] = []
    nx.set_edge_attributes(net_copy, 0, 'allocated_bw')
    for m, n in net_copy.edges:
        net_copy[m][n]['transmission'] = np.zeros((len(traffic)))

    if isinstance(weights, list):
        for i, (m,n) in enumerate(net_copy.edges):
            net_copy[m][n]['w'] = weights[i]
    else:
        for m, n in net_copy.edges:
            net_copy[m][n]['w'] = weights[(m, n)]

    session_buffer_allocation = {s.id: {} for s in traffic}
    for m, j, k in buffer_allocation:
        net_copy.nodes[m]['buffer'].append((j,k))
        session_buffer_allocation[j][k] = m

    all_pair_shortest_path = dict(nx.all_pairs_dijkstra_path(net_copy, weight='w'))

    MDTs = []
    for s in traffic:
        t = SPT(net_copy, s, all_pair_shortest_path, session_buffer_allocation[s.id])
        MDTs.append(t)

        for m,n in t.edges_list():
            net_copy[m][n]['transmission'][s.id] = t.edges[m,n]

    for m, n in net_copy.edges:
        net_copy[m][n]['allocated_bw'] = sum([traffic[i].bw * net_copy[m][n]['transmission'][i] for i in net_copy[m][n]['transmission'].nonzero()[0]])

    cost = sum([link_cost(net_copy[m][n]['capacity'], net_copy[m][n]['allocated_bw']) for m, n in net_copy.edges])

    return cost, MDTs, {m: net_copy.nodes[m]['stored_session'] for m in net_copy.nodes}, \
           {(m,n): net_copy[m][n]['allocated_bw'] for m, n in net_copy.edges}


def create_gnome(genes, size):
    return random.choices(genes, k=size)

def create_individual(net, traffic, chromosome):
    fitness, branch_states = bsa(net, traffic, chromosome)
    return Individual(chromosome, fitness, branch_states)


def mate(parent1, parent2, genes):
    child_chromosome = []
    for i in range(len(parent1.chromosome)):
        # random probability
        prob = random.random()

        # if prob is less than 0.45, insert gene
        # from parent 1
        if prob < 0.45:
            child_chromosome.append(parent1.chromosome[i])

        # if prob is between 0.45 and 0.90, insert
        # gene from parent 2
        elif prob < 0.90:
            child_chromosome.append(parent2.chromosome[i])

        # otherwise insert random gene(mutate),
        # for maintaining diversity
        else:
            child_chromosome.append(random.choice(genes))

    return child_chromosome

# Driver code
def WOBSA(net, traffic, max_iters=[50], log=False):
    print('Start Genetic algorithm')
    # Number of individuals in each generation
    max_iters = sorted(max_iters)
    POPULATION_SIZE = 120
    # current generation
    generation = 1
    GENES = [i for i in range(w_min, w_max+1)]
    n_edges = len(net.edges)

    best_individual = None
    population = []


    # create initial population
    for _ in range(POPULATION_SIZE):
        gnome = create_gnome(GENES, n_edges)
        population.append(create_individual(net, traffic, gnome))

    while generation <= max_iters[-1]:
        # sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)

        # if the individual having lowest fitness score ie.
        # 0 then we know that we have reached to the target
        # and break the loop
        if not best_individual:
            best_individual = copy.deepcopy(population[0])
        elif population[0].fitness <= best_individual.fitness:
            best_individual = copy.deepcopy(population[0])

        # Otherwise generate new offsprings for new generation
        new_generation = []

        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        # From 50% of fittest population, Individuals
        # will mate to produce offspring
        s = int((90 * POPULATION_SIZE) / 100)
        half_population = int(POPULATION_SIZE/2)
        for _ in range(s):
            parent1 = random.choice(population[:half_population])
            parent2 = random.choice(population[:half_population])
            child_chromosome = mate(parent1, parent2, GENES)
            new_generation.append(create_individual(net, traffic, child_chromosome))

        population = new_generation
        if log is not None:
            wobsa_cost, MDTs, stored_sessions, allocated_bw = apply_network_setting(net, traffic, best_individual.chromosome, best_individual.bsa)
            loads = [allocated_bw[(m, n)] / net[m][n]['capacity'] for m, n in net.edges]
            log.append({
                'cost': {
                    'GA': best_individual.fitness,
                },
                'max_link_load': {
                    'GA': max(loads)
                },
                'n_overloaded_links': {
                    'GA': len([l for l in loads if l > 1])
                },
                'n_full_load_nodes': {
                    'GA': len([m for m in net.nodes if len(stored_sessions[m]) / net.nodes[m]['capacity'] >= 1])
                },
            })
        print(f"Generation: {generation}\tbest solution cost: {best_individual.fitness}")

        if generation in max_iters:
            yield best_individual
        generation += 1



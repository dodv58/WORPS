from common import *
from itertools import product
from env_gym import get_mdt
import json

Individual = namedtuple('Individual', 'chromosome fitness')

def calculate_fitness(weights, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw):
    weights = np.array(weights)
    mdt, _ = get_mdt(weights, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw)
    loads = mdt @ traffic_bw / e_cap
    return loads.max()


def create_gnome(genes, size):
    return random.choices(genes, k=size)


def create_individual(chromosome, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw):
    fitness = calculate_fitness(chromosome, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw)
    return Individual(chromosome, fitness)


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
def wo_ga(net, traffic, max_iters=[50], log=False):
    print('Start Genetic algorithm')
    # Number of individuals in each generation
    max_iters = sorted(max_iters)
    POPULATION_SIZE = 150
    # current generation
    generation = 1
    GENES = [i for i in range(w_min, w_max + 1)]
    n_edges = len(net.edges)
    nodes = list(net.nodes)
    n_nodes = len(nodes)
    n_traffic = len(traffic)
    e_cap = np.array([net[m][n]['capacity'] for m, n in net.edges])
    e_sources = np.array([m for m, n in net.edges], dtype=int)
    e_dests = np.array([n for m, n in net.edges], dtype=int)
    e_index_map = np.full((n_nodes, n_nodes), -1, dtype=int)
    e_index_map[e_sources, e_dests] = np.arange(n_edges)
    traffic_bw = np.array([s.bw for s in traffic], dtype=float)
    traffic_sources = np.zeros((n_traffic, n_nodes), dtype=bool)
    traffic_sources[range(n_traffic), [traffic[j].s for j in range(n_traffic)]] = True
    traffic_dests = np.zeros((n_traffic, n_nodes), dtype=bool)
    for j in range(n_traffic):
        traffic_dests[j, traffic[j].ds] = True

    best_individual = None
    population = []

    # create initial population
    population.append(create_individual([1]*n_edges, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw))
    for _ in range(POPULATION_SIZE - 1):
        gnome = create_gnome(GENES, n_edges)
        population.append(create_individual(gnome, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw))

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
        half_population = int(POPULATION_SIZE / 2)
        for _ in range(s):
            parent1 = random.choice(population[:half_population])
            parent2 = random.choice(population[:half_population])
            child_chromosome = mate(parent1, parent2, GENES)
            new_generation.append(create_individual(child_chromosome, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests, traffic_bw))

        population = new_generation

        if generation in max_iters:
            yield best_individual
        generation += 1

if __name__ == "__main__":
    dataset = '025'
    net = nx.read_gml(f'datasets/{dataset}/topo.gml')
    net = nx.convert_node_labels_to_integers(net)

    with open(f'datasets/{dataset}/traffic.txt') as f:
        demands = json.load(f)
        demands = [[Session(*s) for s in traffic] for traffic in demands]

    final_network_costs = []
    for i, traffic in enumerate(demands):
        print(">>>>> Traffic ", i)
        results = wo_ga(net, traffic, max_iters=[200])
        for result in results:
            print(result.fitness)
            final_network_costs.append(result.fitness)

        # n_edges = len(net.edges)
        # nodes = list(net.nodes)
        # n_nodes = len(nodes)
        # n_traffic = len(traffic)
        # e_cap = np.array([net[m][n]['capacity'] for m, n in net.edges])
        # e_sources = np.array([m for m, n in net.edges], dtype=int)
        # e_dests = np.array([n for m, n in net.edges], dtype=int)
        # e_index_map = np.full((n_nodes, n_nodes), -1, dtype=int)
        # e_index_map[e_sources, e_dests] = np.arange(n_edges)
        # traffic_bw = np.array([s.bw for s in traffic], dtype=float)
        # traffic_sources = np.zeros((n_traffic, n_nodes), dtype=bool)
        # traffic_sources[range(n_traffic), [traffic[j].s for j in range(n_traffic)]] = True
        # traffic_dests = np.zeros((n_traffic, n_nodes), dtype=bool)
        # for j in range(n_traffic):
        #     traffic_dests[j, traffic[j].ds] = True
        # weights = np.array([1/net[m][n]["capacity"] for m,n in net.edges])
        # mdt, _ = get_mdt(weights, e_sources, e_dests, nodes, e_index_map, e_cap, traffic_sources, traffic_dests,
        #                  traffic_bw)
        # loads = mdt @ traffic_bw / e_cap
        # final_network_costs.append(loads.max())

    print(final_network_costs)
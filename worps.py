from common import *
from itertools import product

def rps(net, traffic, weights):
    net = copy.deepcopy(net)
    for i, (m, n) in enumerate(net.edges):
        net[m][n]['w'] = weights[i]

    all_shortest_path = dict(nx.all_pairs_dijkstra_path(net, weight='w'))

    traffic = sorted(traffic, key=lambda x: x.bw*len(x.ds), reverse=True)
    selected_rps = {}
    for s in traffic:
        tmp_rps = [(m, attempt_rp(all_shortest_path, s, m)) for m in net.nodes]
        selected_rp, bw_consumption = min(tmp_rps, key=lambda x: x[1])
        selected_rps[s.id] = selected_rp
        # if bw_consumption >= 1e6:
        #     print(s, selected_rp)
        #     raise "RP not found"
        
    return selected_rps

def calculate_fitness(net, traffic, weights, selected_rps):
    net = copy.deepcopy(net)
    for i, (m, n) in enumerate(net.edges):
        net[m][n]['w'] = weights[i]

    nx.set_edge_attributes(net, 0, 'allocated_bw')
    all_shortest_path = dict(nx.all_pairs_dijkstra_path(net, weight='w'))
    penalties = 0

    failed_sessions = set()
    for s in traffic:
        rp = selected_rps[s.id]
        if rp != s.s:
            for m,n in links_in_path(all_shortest_path[s.s][rp]):
                net[m][n]['allocated_bw'] += s.bw
        tree = set()
        routes = [links_in_path(all_shortest_path[rp][d]) for d in s.ds]
        delays = [len(route) for route in routes]

        for route in routes:
            tree.update(route)
        for m,n in tree:
            net[m][n]['allocated_bw'] += s.bw

        for d1, d2 in product(delays, delays):
            if d1 - d2 > s.delta:
                failed_sessions.add(s.id)
                break

    # penalties = len(failed_sessions)/len(traffic)*10
    penalties = len(failed_sessions) / 15
    return max([net[m][n]['allocated_bw']/net[m][n]['capacity'] for m,n in net.edges]) + penalties

def attempt_rp(all_shortest_path, s, m):
    # return bw consumption by s if m is selected as rp for s
    tree = set(links_in_path(all_shortest_path[s.s][m])) if m != s.s else set()
    delays = []
    penalty = 0
    for d in s.ds:
        route = links_in_path(all_shortest_path[m][d])
        tree.update(route)
        delays.append(len(route))
    for d1, d2 in product(delays, delays):
        if d1 - d2 > s.delta:
            penalty = 10
            break

    return len(tree)*s.bw + penalty

def create_gnome(genes, size):
    return random.choices(genes, k=size)

def create_individual(net, traffic, chromosome):
    selected_rps = rps(net, traffic, chromosome)
    fitness = calculate_fitness(net, traffic, chromosome, selected_rps)
    return Individual(chromosome, fitness, selected_rps)


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
def worps_ga(net, traffic, max_iters=[50], log=False):
    print('Start Genetic algorithm')
    # Number of individuals in each generation
    max_iters = sorted(max_iters)
    POPULATION_SIZE = 100
    # current generation
    generation = 1
    GENES = [i for i in range(w_min, w_max+1)]
    n_edges = len(net.edges)

    best_individual = None
    population = []


    # create initial population
    population.append(create_individual(net, traffic, [1]*n_edges))
    for _ in range(POPULATION_SIZE - 1):
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
        if log:
            wobsa_cost, MDTs, stored_sessions, allocated_bw = apply_network_setting(net, traffic, best_individual.chromosome, best_individual.rps)
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

def local_search(net, traffic, weight):
    pass

def worps_pimsm(net, traffic):
    weights = inverse_capacity_weights(net, traffic)
    selected_rps = rps(net, traffic, weights)
    cost, failed_rate = apply_network_setting(net, traffic, weights, selected_rps)
    return weights, selected_rps, cost, failed_rate

def worps_hopcount(net, traffic):
    weights = unit_weights(net, traffic)
    selected_rps = rps(net, traffic, weights)
    cost, failed_rate = apply_network_setting(net, traffic, weights, selected_rps)
    return weights, selected_rps, cost, failed_rate
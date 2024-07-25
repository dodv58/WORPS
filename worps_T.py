import gurobipy as gp
from gurobipy import GRB
from common import *
import networkx as nx
from pathlib import Path
import argparse
from multiprocessing import Pool
import json

np.random.seed(1)
random.seed(1)

def worps_T_solver(net, demands, dir=None):
    T = list(range(len(demands)))
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            print('feasible solution', model.cbGetSolution(model.getVars()))

    model = gp.Model()
    model.setParam('LogToConsole', 1)
    model.setParam('MIPGap', 0.5)
    model.setParam('TimeLimit', 2*60*60)

    if dir is not None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        model.setParam('SolFiles', f'{dir}/feasible')

    all_node_pair = [(m, n) for m in net.nodes for n in net.nodes if m != n]
    all_pair_simple_paths = {}  # path contains nodes
    all_pair_simple_routes = {}  # route contains links
    for m, n in all_node_pair:
        paths = list(nx.all_simple_paths(net, m, n))
        all_pair_simple_paths[(m,n)] = paths
        all_pair_simple_routes[(m,n)] = [links_in_path(p) for p in paths]

    demands = {i: demand for i, demand in enumerate(demands)}
    demand_traffic_ids = {k: [s.id for s in v] for k,v in demands.items()}
    traffic = []
    traffic_ids = []
    for demand in demands.values():
        traffic.extend(demand)
        traffic_ids.extend([s.id for s in demand])

    X = model.addVars([(m,n,j,k,t) for m,n in net.edges for t in T for j in demand_traffic_ids[t] for k in traffic[j].ds], vtype=GRB.BINARY, name='x')
    Y = model.addVars([(m,n,j,t) for m,n in net.edges for t in T for j in demand_traffic_ids[t]], vtype=GRB.BINARY, name='y')
    W = model.addVars([(m, n, t) for m, n in net.edges for t in T], vtype=GRB.INTEGER, name='w', lb=w_min, ub=w_max)
    B = model.addVars([(m, j, t) for m in net.nodes for t in T for j in demand_traffic_ids[t]], vtype=GRB.BINARY, name='b')
    # C = model.addVars([(m, n) for m, n in net.edges], vtype=GRB.CONTINUOUS, name='c')

    max_utilization = model.addVars(T, vtype=GRB.CONTINUOUS, name='max_utilization')
    update_cost = model.addVars([(m,j,t) for m in net.nodes for t in T for j in demand_traffic_ids[t]], vtype=GRB.CONTINUOUS, name='update_cost')

    # flow conservation
    for m in net.nodes:
        for t in T:
            for j in demand_traffic_ids[t]:
                for k in traffic[j].ds:
                    if m == k:
                        model.addConstr(X.sum('*', m, j, k, t) - X.sum(m, '*', j, k, t) == 1)
                        model.addConstr(X.sum(m, '*', j, k, t) <= 0)
                        model.addConstr(X.sum('*', m, j, k, t) <= 1)
                    elif m == traffic[j].s:
                        model.addConstr(X.sum('*', m, j, k, t) - X.sum(m, '*', j, k, t) == -1)
                        model.addConstr(X.sum('*', m, j, k, t) <= 0)
                    else:
                        model.addConstr(X.sum('*', m, j, k, t) - X.sum(m, '*', j, k, t) == 0)
                        model.addConstr(X.sum('*', m, j, k, t) <= 1)
    model.addConstrs((Y[m,n,j,t] >= X[m,n,j,k,t]) for m,n,j,k,t in X)
    model.addConstrs((Y[m,n,j,t] <= X.sum(m,n,j,'*',t)) for m, n, j, t in Y)


    model.addConstrs((B.sum('*', j, t) == 1 for t in T for j in demand_traffic_ids[t]))

    model.addConstrs((X.sum(m,'*',j,k,t) >= B[m,j,t] for m in net.nodes for t in T for j in demand_traffic_ids[t] for k in traffic[j].ds))

    # delay variation
    model.addConstrs((X.sum('*', '*', j, k, t) - X.sum('*', '*', j, h, t) <= traffic[j].delta
        for t in T for j in demand_traffic_ids[t] for k in traffic[j].ds for h in traffic[j].ds if h != k))


    ### link load
    link_loads = [[gp.quicksum(Y[m, n, j, t] * traffic[j].bw for j in demand_traffic_ids[t]) / net[m][n]['capacity'] for m, n in net.edges] for t in T]

    # network cost
    for t in T:
        model.addConstrs((max_utilization[t] >= link_loads[t][i] for i in range(len(net.edges))))

    # # update cost
    for t in T[:-1]:
        model.addConstrs((update_cost[m, j, t] >= B[m, j, t] - B[m, j, t+1] for m in net.nodes for j in demand_traffic_ids[t]))
        model.addConstrs((update_cost[m, j, t] >= B[m, j, t+1] - B[m, j, t] for m in net.nodes for j in demand_traffic_ids[t]))


    # shortest path
    for t in T:
        for j in demand_traffic_ids[t]:
            for k in traffic[j].ds:
                routes = all_pair_simple_routes[(traffic[j].s, k)]
                for t1, p in enumerate(routes):
                    for t2, q in enumerate(routes):
                        if t1 != t2:
                            model.addConstr(gp.quicksum(W[m,n,t]*X[m,n,j,k,t] for m,n in p) + 0.1 <= gp.quicksum(W[m,n,t] for m,n in q))


    model.setObjective(max_utilization.sum() + update_cost.sum()*0.1, GRB.MINIMIZE)
    model.optimize()
    model.write('model.lp')
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        f = open('vars.txt', 'w')
        for v in model.getVars():
            if v.X > 0:
                f.write(f"{v.VarName} = {v.X}\n")
        route = [{j: [] for j in traffic_ids} for t in T]
        for m,n,j,k,t in X:
            if X[m,n,j,k,t].X > 0:
                route[t][j].append((m,n))

        # for m,j,t in update_cost:
        #     print((m,j,t), update_cost[m,j,t].X)

        return model.getObjective().getValue(), {(m,n,t): W[m,n,t].X for m,n,t in W} , \
               [(m,j,t) for m,j,t in B if B[m,j,t].X > 0], route
    else:
        return None

def run(net, demands, T, dir=None):
    results = []
    worps_T_solver(net, demands)
    #
    # for traffic in demands:
    #     weights, selected_rps, cost, failed_rate = worps_T_solver(net, traffic, T)
    #     results.append({
    #         "weights": weights,
    #         "cost": cost,
    #         "selected_rps": selected_rps,
    #         "failed_sessions": failed_rate
    #     })
    # print(results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', help='sum_cost/max_load', default='max_load')
    parser.add_argument('--dataset', help='dataset', required=True)
    parser.add_argument('--log', default=0, type=int)
    parser.add_argument('--run', default="t01")

    args = parser.parse_args()

    dir = f'experiments/{args.run}/{args.dataset}'
    net = nx.read_gml(f'datasets/{args.dataset}/topo.gml')
    net = nx.convert_node_labels_to_integers(net)
    with open(f'datasets/{args.dataset}/meta.txt') as f:
        meta = json.load(f)

    with open(f'datasets/{args.dataset}/traffic.txt') as f:
        data = json.load(f)[0]
        demands = [[Session(*s) for s in traffic] for traffic in data]

    run(net, demands, meta["T"])
import gurobipy as gp
from gurobipy import GRB
from common import *
import networkx as nx
from pathlib import Path

def worps_solver(net, traffic, dir=None):

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

    traffic_ids = [s.id for s in traffic]
    traffic = {s.id: s for s in traffic}

    X = model.addVars([(m,n,j,k) for m,n in net.edges for j in traffic_ids for k in traffic[j].ds], vtype=GRB.BINARY, name='x')
    Y = model.addVars([(m,n,j) for m,n in net.edges for j in traffic_ids], vtype=GRB.BINARY, name='y')
    W = model.addVars([(m, n) for m, n in net.edges], vtype=GRB.INTEGER, name='w', lb=w_min, ub=w_max)
    B = model.addVars([(m, j) for m in net.nodes for j in traffic_ids], vtype=GRB.BINARY, name='b')
    C = model.addVars([(m, n) for m, n in net.edges], vtype=GRB.CONTINUOUS, name='c')
    max_utilization = model.addVar(vtype=GRB.CONTINUOUS, name='max_utilization')

    # flow conservation
    for m in net.nodes:
        for j in traffic_ids:
            for k in traffic[j].ds:
                if m == k:
                    model.addConstr(X.sum('*', m, j, k) - X.sum(m, '*', j, k) == 1)
                    model.addConstr(X.sum(m, '*', j, k) <= 0)
                    model.addConstr(X.sum('*', m, j, k) <= 1)
                elif m == traffic[j].s:
                    model.addConstr(X.sum('*', m, j, k) - X.sum(m, '*', j, k) == -1)
                    model.addConstr(X.sum('*', m, j, k) <= 0)
                else:
                    model.addConstr(X.sum('*', m, j, k) - X.sum(m, '*', j, k) == 0)
                    model.addConstr(X.sum('*', m, j, k) <= 1)
    model.addConstrs((Y[m,n,j] >= X[m,n,j,k]) for m,n,j,k in X)
    model.addConstrs((Y[m,n,j] <= X.sum(m,n,j,'*')) for m, n, j in Y)


    model.addConstrs((B.sum('*', j) == 1 for j in traffic_ids))

    model.addConstrs((X.sum(m,'*',j,k) >= B[m,j] for m in net.nodes for j in traffic_ids for k in traffic[j].ds))

    # link transmission capacity
    # model.addConstrs((L[m,n] <= net[m][n]['capacity'] for m,n in net.edges))

    # delay variation
    model.addConstrs((X.sum('*', '*', j, k) - X.sum('*', '*', j, h) <= traffic[j].delta
        for j in traffic_ids for k in traffic[j].ds for h in traffic[j].ds if h != k))


    ### link load
    link_loads = [gp.quicksum(Y[m, n, j] * traffic[j].bw for j in traffic_ids) / net[m][n]['capacity'] for m, n in
                 net.edges]

    #
    model.addConstrs((max_utilization >= link_loads[i] for i in range(len(net.edges))))


    # shortest path
    for j in traffic_ids:
        for k in traffic[j].ds:
            routes = all_pair_simple_routes[(traffic[j].s, k)]
            for t1, p in enumerate(routes):
                for t2, q in enumerate(routes):
                    if t1 != t2:
                        model.addConstr(gp.quicksum(W[m,n]*X[m,n,j,k] for m,n in p) + 0.1 <= gp.quicksum(W[m,n] for m,n in q))


    model.setObjective(max_utilization, GRB.MINIMIZE)
    model.optimize()
    model.write('model.lp')
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        f = open('vars.txt', 'w')
        for v in model.getVars():
            if v.X > 0:
                f.write(f"{v.VarName} = {v.X}\n")
        route = {j: [] for j in traffic_ids}
        for m,n,j,k in X:
            if X[m,n,j,k].X > 0:
                route[j].append((m,n))

        return model.getObjective().getValue(), {(m, n): W[m, n].X for m, n in W} , \
               [(m, j) for m,j in B if B[m,j].X > 0], route
    else:
        return None
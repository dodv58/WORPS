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

    X1 = model.addVars([(m, n, j) for m,n in net.edges for j in traffic_ids] , vtype=GRB.BINARY, name='x1')
    X2 = model.addVars([(m,n,j,k) for m,n in net.edges for j in traffic_ids for k in traffic[j].ds], vtype=GRB.BINARY, name='x2')
    Y = model.addVars([(m, n, j) for m,n in net.edges for j in traffic_ids], vtype=GRB.BINARY, name='y')
    W = model.addVars([(m, n) for m, n in net.edges], vtype=GRB.INTEGER, name='w', lb=w_min, ub=w_max)
    B = model.addVars([(m, j) for m in net.nodes for j in traffic_ids], vtype=GRB.BINARY, name='b')
    C = model.addVars([(m, n) for m, n in net.edges], vtype=GRB.CONTINUOUS, name='c')
    # delta = model.addVars([(m, j, k) for m in net.nodes for j in traffic_ids for k in traffic[j].ds], vtype=GRB.BINARY, name='delta')
    # gamma = model.addVars([(m, j, k) for m in net.nodes for j in traffic_ids for k in traffic[j].ds], vtype=GRB.BINARY, name='gamma')

    # flow conservation
    for m in net.nodes:
        for j in traffic_ids:
            if m == traffic[j].s:
                model.addConstr(X1.sum('*', m, j) - X1.sum(m, '*', j) == B[m,j] - 1)
                model.addConstr(X1.sum('*', m, j) <= 0)
            else:
                model.addConstr(X1.sum('*', m, j) - X1.sum(m, '*', j) == B[m,j])

    for m in net.nodes:
        for j in traffic_ids:
            for k in traffic[j].ds:
                if m == k:
                    model.addConstr(X2.sum('*', m, j, k) - X2.sum(m, '*', j, k) == 1 - B[m,j])
                    model.addConstr(X2.sum(m, '*', j, k) <= 0)
                else:
                    model.addConstr(X2.sum('*', m, j, k) - X2.sum(m, '*', j, k) == -B[m,j])
                    model.addConstr(X1.sum('*', m, j, k) <= 1 - B[m, j])

    model.addConstrs((X.sum('*', m, j, k) <= 1 for m in net.nodes for j in traffic_ids for k in traffic[j].ds))
    model.addConstrs((Y[m,n,j] >= X[m,n,j,k] for m,n,j,k in X))
    model.addConstrs((Y[m, n, j] <= X.sum(m, n, j, '*') for m, n, j in Y))

    # model.addConstrs((delta[m,j,k] <= X[m,n,j,k] for m,n,j,k in X))
    # model.addConstrs((delta[m,j,k] >= B[m,j] for m,j,k in delta))
    # model.addConstrs((delta[m,j,k] >= delta[n,j,k]*X[m,n,j,k] for m,n,j,k in X))
    #
    # model.addConstrs((gamma[m, j, k] <= X[m, n, j, k] for m, n, j, k in X))
    # model.addConstrs((gamma[m, j, k] >= B[m, j] for m, j, k in gamma))
    # model.addConstrs((gamma[n, j, k] >= delta[m, j, k] * X[m, n, j, k] for m, n, j, k in X))
    #
    # model.addConstrs((X.sum('*', n, j, k)*delta[n,j,k] <= 1 for n in net.nodes for j in traffic_ids for k in traffic[j].ds))
    # model.addConstrs(
    #     (X.sum('*', n, j, k) * gamma[n, j, k] <= 1 for n in net.nodes for j in traffic_ids for k in traffic[j].ds))


    model.addConstrs((B.sum('*', j) == 1 for j in traffic_ids))

    model.addConstrs((X.sum(m,'*',j,k) >= B[m,j] for m in net.nodes for j in traffic_ids for k in traffic[j].ds))

    # link transmission capacity
    # model.addConstrs((L[m,n] <= net[m][n]['capacity'] for m,n in net.edges))

    # delay variation
    model.addConstrs((X.sum('*', '*', j, k) - X.sum('*', '*', j, h) <= traffic[j].delta
        for j in traffic_ids for k in traffic[j].ds for h in traffic[j].ds if h != k))


    ### link load
    link_load = [gp.quicksum(Y[m, n, j] * traffic[j].bw for j in traffic_ids) / net[m][n]['capacity'] for m, n in
                 net.edges]

    # link cost
    model.addConstrs((C[m, n] >= link_load[index] for index, (m, n) in enumerate(net.edges)))
    model.addConstrs((C[m, n] >= 3 * link_load[index] - 2 / 3 for index, (m, n) in enumerate(net.edges)))
    model.addConstrs((C[m, n] >= 10 * link_load[index] - 16 / 3 for index, (m, n) in enumerate(net.edges)))
    model.addConstrs((C[m, n] >= 70 * link_load[index] - 178 / 3 for index, (m, n) in enumerate(net.edges)))
    model.addConstrs((C[m, n] >= 500 * link_load[index] - 1468 / 3 for index, (m, n) in enumerate(net.edges)))
    model.addConstrs((C[m, n] >= 5000 * link_load[index] - 19468 / 3 for index, (m, n) in enumerate(net.edges)))


    for j in traffic_ids:
        for k in traffic[j].ds:
            routes = all_pair_simple_routes[(traffic[j].s, k)]
            for t1, p in enumerate(routes):
                for t2, q in enumerate(routes):
                    if t1 != t2:
                        model.addConstr(gp.quicksum(W[m,n]*X[m,n,j,k] for m,n in p) + 0.1 <= gp.quicksum(W[m,n] for m,n in q))


    model.setObjective(C.sum(), GRB.MINIMIZE)
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
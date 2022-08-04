import networkx as nx
import numpy as np
from Environment import *
from pyomo.environ import Var, ConstraintList, ConcreteModel, Binary, NonNegativeReals, Objective, SolverFactory, \
    quicksum, minimize
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
from copt_pyomo import *


def _generate_network(case: GridCase):
    local_graph = nx.relabel_nodes(case.graph, case.bus_dict)
    edge = list(local_graph.edges)
    prob = case.linesafe[:, -1].flatten()
    G = nx.Graph()
    G.add_edges_from(edge)
    double_ind = {}
    single_ind = {}
    for i, (u, v) in enumerate(edge):
        double_ind[(u, v)] = 2 * i
        double_ind[(v, u)] = 2 * i + 1
        single_ind[(u, v)] = i
        single_ind[(v, u)] = i
    node_index = range(len(G.nodes))
    v = np.array(case.bus[:, VALUE] * case.bus[:, PLOAD]).flatten()
    return G, edge, double_ind, single_ind, node_index, prob, v


def _optimize_model_gurobipy(case):
    import gurobipy as grb
    model = grb.Model(name='Minimal-Risk Spanning Tree')
    graph, edge, d_index, s_index, n_index, prob, value = _generate_network(case)
    adj = graph.adj
    N = len(graph.nodes)
    L = len(graph.edges)
    # 线路选择参数
    x = model.addVars(range(L), vtype=grb.GRB.CONTINUOUS, lb=0, ub=1)
    b = model.addVars(range(2 * L), vtype=grb.GRB.BINARY)
    # 节点停运概率
    P = model.addVars(range(N), vtype=grb.GRB.CONTINUOUS, lb=0.0, ub=1.0)
    z = model.addVars(range(2 * L), vtype=grb.GRB.CONTINUOUS, lb=0.0, ub=1.0)

    # 线路选择约束
    for i in range(L):
        model.addConstr(x[i] == b[2 * i] + b[2 * i + 1], name=f'线路选择约束{edge[i]}')

    # 辐射状结构约束
    for i in range(1, N):
        model.addConstr(grb.quicksum([b[d_index[(j, i)]] for j in adj[i].keys()]) == 1,
                        name=f'辐射状约束，节点{i}')
    # 根节点约束
    for j in adj[0].keys():
        model.addConstr(b[d_index[j, 0]] == 0, name=f'根节点约束b')
    # 节点概率约束
    for i in range(1, N):
        for j in adj[i].keys():
            model.addConstr(prob[s_index[(i, j)]] * P[j] - z[d_index[(j, i)]] <= 1 - b[d_index[(j, i)]], name='辅助变量约束a')
            model.addConstr(prob[s_index[(i, j)]] * P[j] - z[d_index[(j, i)]] >= 0.0, name=f'辅助变量约束b{i}{j}')
            model.addConstr(z[d_index[(j, i)]] <= b[d_index[(j, i)]], name='辅助变量约束c')
        model.addConstr(P[i] == grb.quicksum([z[d_index[(j, i)]] for j in adj[i].keys()]), name=f'节点概率对辅助变量z约束{i}')

    model.addConstr(P[0] == 1.0, name='电源约束')
    model.update()
    model.setObjective(grb.quicksum([(1 - P[i]) * value[i] for i in range(N)]), sense=grb.GRB.MINIMIZE)
    model.optimize()
    if model.status == grb.GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        # do IIS, find infeasible constraints
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
    elif model.status == grb.GRB.OPTIMAL:
        empty_graph = nx.DiGraph()
        index = np.array([x[k].X for k in range(L)])
        index = list(np.nonzero(index > 0.5)[0])
        edge_list = []
        empty_graph.add_nodes_from(range(N))
        for i, e in enumerate(edge):
            if i in index:
                if b[d_index[e]].X > 0:
                    edge_list.append(e)
                else:
                    edge_list.append((e[1], e[0]))
        empty_graph.add_edges_from(edge_list)
        new_pos = graphviz_layout(empty_graph, prog="dot")
        nx.draw_networkx(empty_graph, pos=new_pos)
        print(f'系统最小风险为{model.getObjective().getValue()}')
    else:
        print(model.status)


def _optimize_model_pyomo(case, solver='glpk'):
    m = ConcreteModel()
    graph, edge, d_index, s_index, n_index, prob, value = _generate_network(case)
    adj = graph.adj
    N = len(graph.nodes)
    L = len(graph.edges)
    m.x = Var(range(L), domain=NonNegativeReals, bounds=(0, 1))
    m.b = Var(range(2 * L), domain=Binary)
    # 节点停运概率
    m.P = Var(range(N), domain=NonNegativeReals, bounds=(0, 1))
    m.z = Var(range(2 * L), domain=NonNegativeReals, bounds=(0, 1))

    # 线路选择约束
    m.line_select = ConstraintList()
    for i in range(L):
        m.line_select.add(expr=m.x[i] == m.b[2 * i] + m.b[2 * i + 1])

    # 辐射状结构约束
    m.radial = ConstraintList()
    for i in range(1, N):
        m.radial.add(expr=quicksum([m.b[d_index[(j, i)]] for j in adj[i].keys()]) == 1)
    # 根节点约束
    m.root = ConstraintList()
    for j in adj[0].keys():
        m.root.add(expr=m.b[d_index[j, 0]] == 0)
    # 节点概率约束
    m.prob = ConstraintList()
    for i in range(1, N):
        for j in adj[i].keys():
            m.prob.add(prob[s_index[(i, j)]] * m.P[j] - m.z[d_index[(j, i)]] <= 1 - m.b[d_index[(j, i)]])
            m.prob.add(prob[s_index[(i, j)]] * m.P[j] - m.z[d_index[(j, i)]] >= 0.0)
            m.prob.add(m.z[d_index[(j, i)]] <= m.b[d_index[(j, i)]])
        m.prob.add(m.P[i] == quicksum([m.z[d_index[(j, i)]] for j in adj[i].keys()]))
    m.prob.add(m.P[0] == 1.0)
    m.obj = Objective(expr=quicksum([(1 - m.P[i]) * value[i] for i in range(N)]), sense=minimize)
    opt = SolverFactory(solver)
    opt.solve(m, tee=True)
    empty_graph = nx.DiGraph()
    index = np.array([m.x[k]() for k in range(L)])
    index = list(np.nonzero(index > 0.5)[0])
    edge_list = []
    empty_graph.add_nodes_from(range(N))
    for i, e in enumerate(edge):
        if i in index:
            if m.b[d_index[e]]() > 0:
                edge_list.append(e)
            else:
                edge_list.append((e[1], e[0]))
    empty_graph.add_edges_from(edge_list)
    # new_pos = graphviz_layout(empty_graph, prog="dot")
    # nx.draw_networkx(empty_graph, pos=new_pos)
    from netgraph import Graph
    node_color_dict = dict.fromkeys([n for n in empty_graph.nodes])
    import matplotlib.cm as cm
    for k in empty_graph.nodes:
        node_color_dict[k] = cm.hsv(m.P[k]())
    Graph(empty_graph, node_layout='dot', arrows=True, node_labels=True, node_label_fontdict=dict(size=14),
          edge_cmap=cm)
    import matplotlib.pyplot as plt
    plt.show()
    print(f'系统最小风险为{m.obj()}')


def optimize_minimum_spanning_tree(case, model_lang='pyomo', solver='glpk'):
    if model_lang == 'gurobipy':
        _optimize_model_gurobipy(case)
    else:
        _optimize_model_pyomo(case, solver=solver)


if __name__ == "__main__":
    # station_name = 'IEEE33Loop'
    # filepath = '../../data/'
    # case = Case().from_file(filepath=filepath, station_name=station_name)
    # case.bus[:, VALUE] = 10 * np.random.rand(len(case.bus))
    # case.linesafe = np.random.rand(len(case.linesafe)).reshape(-1, 1)
    from GridResilience.GridCase import *

    case = case_32_modified()
    optimize_minimum_spanning_tree(case, model_lang='pyomo', solver='copt_direct')

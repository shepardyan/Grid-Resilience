import gurobipy as grb
from pyomo.environ import Var, ConstraintList, ConcreteModel, Binary, NonNegativeReals, Reals, Objective, SolverFactory, \
    quicksum, maximize, NonPositiveReals
from pyomo.contrib import appsi
from pandapower.networks import *
import networkx as nx
from math import pi
from pandapower.topology.create_graph import create_nxgraph

ext_net = case_ieee30()


def prof(net):
    lines = len(net.line)
    rel = [0.9] * len(net.line)
    G = nx.Graph()
    G.add_nodes_from(net.bus['name'])
    for i in range(lines):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        G.add_edge(u, v)

    # 生成索引
    i_to_j = {}
    ij = {}
    q_ind = {}
    for i in range(lines):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        i_to_j[(u, v)] = i * 2
        i_to_j[(v, u)] = i * 2 + 1
        ij[(u, v)] = i
        ij[(v, u)] = i
        q_ind[(u, v, u)] = i * 2
        q_ind[(v, u, u)] = i * 2
        q_ind[(u, v, v)] = i * 2 + 1
        q_ind[(v, u, v)] = i * 2 + 1

    m = ConcreteModel()  # 确定模型
    adj = G.adj  # 邻接表
    m.z = Var(range(lines), domain=Binary)
    m.theta = Var(range(len(net.bus)), domain=Reals, bounds=(-2 * pi, 2 * pi))
    m.c = Var(range(len(net.bus)), domain=Binary)
    m.p = Var(range(lines * 2), domain=Reals, bounds=(-10, 10))
    m.q = Var(range(2 * lines), domain=Reals, bounds=(-10, 10))

    # 支路潮流平衡
    m.brh_pf = ConstraintList()
    for i in range(lines):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        m.brh_pf.add(expr=m.p[i_to_j[(u, v)]] == 1 / (net.line['x_ohm_per_km'].iloc[i] * 100 / (230 ** 2)) * (
                m.q[q_ind[(u, v, u)]] - m.q[q_ind[(u, v, v)]]))
        m.brh_pf.add(expr=m.p[i_to_j[(u, v)]] == -m.p[i_to_j[(v, u)]])

    # 节点功率平衡
    m.bus_pf = ConstraintList()
    load_info = net.load.copy()
    load_info = load_info.set_index('bus')
    for i in range(len(net.bus)):
        if i != net.gen['bus'].iloc[0]:
            try:
                m.bus_pf.add(expr=quicksum([m.p[i_to_j[(i, j)]]
                                            for j in adj[i]]) == -m.c[i] * load_info['p_mw'].loc[i] / 100)
            except KeyError:
                pass
    # θ相关约束
    m.q_aux = ConstraintList()
    m.q_aux.add(expr=m.theta[net.gen['bus'].iloc[0]] == 0.0)
    for i in range(lines):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        m.q_aux.add(expr=m.q[q_ind[(u, v, u)]] <= 2 * pi * m.z[ij[(u, v)]])
        m.q_aux.add(expr=m.q[q_ind[(u, v, u)]] >= -2 * pi * m.z[ij[(u, v)]])
        m.q_aux.add(expr=m.theta[u] - m.q[q_ind[(u, v, u)]] <= 2 * pi * (1 - m.z[ij[(u, v)]]))
        m.q_aux.add(expr=m.theta[u] - m.q[q_ind[(u, v, u)]] >= -2 * pi * (1 - m.z[ij[(u, v)]]))
        m.q_aux.add(expr=m.q[q_ind[(u, v, v)]] <= 2 * pi * m.z[ij[(u, v)]])
        m.q_aux.add(expr=m.q[q_ind[(u, v, v)]] >= -2 * pi * m.z[ij[(u, v)]])
        m.q_aux.add(expr=m.theta[v] - m.q[q_ind[(u, v, v)]] <= 2 * pi * (1 - m.z[ij[(u, v)]]))
        m.q_aux.add(expr=m.theta[v] - m.q[q_ind[(u, v, v)]] >= -2 * pi * (1 - m.z[ij[(u, v)]]))

    # c相关约束
    m.c_aux = ConstraintList()
    m.c_aux.add(expr=m.c[0] == 1)
    for i in range(len(net.bus)):
        if i != net.gen['bus'].iloc[0]:
            m.c_aux.add(expr=m.c[i] <= quicksum([m.z[ij[(i, j)]] for j in adj[i]]))
            for j in adj[i]:
                m.c_aux.add(expr=m.c[i] >= m.z[ij[(i, j)]])

    # 线路越界约束
    m.y = Var(domain=Binary)
    M = 10
    m.overflow = ConstraintList()
    p_tolerance = 5
    m.overflow.add(expr=m.p[i_to_j[(0, 1)]] + M * m.y >= p_tolerance)
    m.overflow.add(expr=-m.p[i_to_j[(0, 1)]] + M * (1 - m.y) >= p_tolerance)
    # 上下界

    m.obj = Objective(expr=quicksum(
        [(1 - m.z[i]) * np.log(1 - rel[i]) for i in range(lines)]) + quicksum(
        [m.z[i] * np.log(rel[i]) for i in range(lines)]), sense=maximize)

    opt = SolverFactory('gurobi')
    opt.options['PoolSearchMode'] = 2  # Find N feasible MIP solutions
    opt.options['PoolSolutions'] = 5
    opt.solve(m, tee=True)
    m.display()
    return m


prof(ext_net)
net = ext_net

import pyomo.environ as pe
import networkx as nx
import numpy as np
from GridResilience.Environment import *


def bus_types(gc: GridCase):
    s, pv, pq = [], [], []
    return s, pv, pq


def overflow_search(gc: GridCase, rel, solver='copt_direct'):
    assert len(rel) == np.size(gc.branch, axis=0)
    # 定义相关常量
    BaseMVA = gc.baseMVA
    BaseZ = (gc.bus[0, BASE_KV] ** 2) / BaseMVA  # 阻抗基值
    slack, PV, PQ = bus_types(gc.bus, gc.extend_attr['gen'])  # 三类节点对应索引
    bus_list = gc.bus[:, 0].tolist()  # type:list
    branch_list = gc.branch[:, 0].tolist()  # type:list

    BRH_NUM = len(pp_net.line)
    BUS_NUM = len(pp_net.bus)
    type_dict = {}
    ij_idx = range(BRH_NUM)
    bus = pp_net.bus.index
    gen = [pp_net.ext_grid['bus'].iloc[0]]
    type_dict[pp_net.ext_grid['bus'].iloc[0]] = "Vθ"
    other_gen = [k for k in pp_net.gen['bus']]
    for k in other_gen:
        type_dict[k] = "PV"
    gen.extend(other_gen)
    load_node = [k for k in pp_net.load['bus']]
    load = []
    for i in load_node:
        if i not in gen:
            load.append(i)
            type_dict[i] = "PQ"
    G = nx.Graph()
    G.add_nodes_from(bus)
    for i in ij_idx:
        u, v = pp_net.line['from_bus'].iloc[i], pp_net.line['to_bus'].iloc[i]
        G.add_edge(u, v)
    adj = G.adj
    # 生成索引
    i_to_j = {}
    ij = {}
    q_ind = {}
    num_to_branch = {}
    for i in ij_idx:
        u, v = pp_net.line['from_bus'].iloc[i], pp_net.line['to_bus'].iloc[i]
        num_to_branch[i] = (u, v)
        i_to_j[(u, v)] = i * 2
        i_to_j[(v, u)] = i * 2 + 1
        ij[tuple(sorted((u, v)))] = i
        q_ind[(u, v, u)] = i * 2
        q_ind[(v, u, u)] = i * 2
        q_ind[(u, v, v)] = i * 2 + 1
        q_ind[(v, u, v)] = i * 2 + 1

    # 建立优化模型
    m = pe.ConcreteModel()
    m.ang = pe.Var(bus_list, within=pe.Reals, bounds=(-2 * np.pi, 2 * np.pi))
    m.vol = pe.Var(bus_list, within=pe.NonNegativeReals, bounds=(0.0, 1.5))
    m.Pij = pe.Var()
    m.Qij = pe.Var()
    m.Pi = pe.Var(bus_list, within=pe.NonNegativeReals)
    m.Qi = pe.Var(bus_list, within=pe.Reals)

    m.obj = pe.Objective()
    opt = pe.SolverFactory(solver)
    opt.solve(m, tee=True)


if __name__ == "__main__":
    from pandapower.networks import case33bw
    from GridResilience.Environment.Converter import pandapower_to_gc
    from pandapower.converter import to_ppc

    pp_net = case33bw()
    ppc = pc.to_ppc(pp_net, init='flat')
    net = pandapower_to_gc(pp_net)
    slack, PV, PQ = bus_types(net)

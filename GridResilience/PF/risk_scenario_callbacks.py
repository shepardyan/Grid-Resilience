import numpy as np
import gurobipy as grb
import pandapower as pp
import pandas as pd
from pandapower.networks import *
import pandapower.converter as pc
import networkx as nx
from math import pi
from gurobipy import *
import prettytable as pt
import matplotlib.pyplot as plt


def basic_model(pp_net: pp.pandapowerNet, rel: list, line_from: int, line_to: int, break_line_index: int,
                pool_search_method=2,
                pool_num=5, warm_init=None, verbose=1, formulation="0"):
    """
    应用直流潮流模型，选择灾害最有可能发生的场景集
    """
    assert len(rel) == len(pp_net.line)
    # Network definition
    BRH_NUM = len(pp_net.line)
    BUS_NUM = len(pp_net.bus)
    type_dict = {}
    ij_idx = range(BRH_NUM)
    bus = range(BUS_NUM)
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
    for i in ij_idx:
        u, v = pp_net.line['from_bus'].iloc[i], pp_net.line['to_bus'].iloc[i]
        i_to_j[(u, v)] = i * 2
        i_to_j[(v, u)] = i * 2 + 1
        ij[tuple(sorted((u, v)))] = i
        q_ind[(u, v, u)] = i * 2
        q_ind[(v, u, u)] = i * 2
        q_ind[(u, v, v)] = i * 2 + 1
        q_ind[(v, u, v)] = i * 2 + 1

    # 模型初始化
    model = Model(name="Risk scenario selection")
    model.setParam('PoolSearchMode', pool_search_method)
    model.setParam('PoolSolutions', pool_num)
    model.setParam('MIPFocus', 2)
    model.setParam('Heuristics', 0)
    model.setParam('GomoryPasses', 0)
    model.setParam('Aggregate', 0)

    # 变量定义区
    brh_p = model.addVars(i_to_j, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'p')
    ang = model.addVars(bus, vtype=GRB.CONTINUOUS, lb=-2 * pi, ub=2 * pi, name='ang')
    z = model.addVars(ij, vtype=GRB.BINARY, name='z')
    c = model.addVars(bus, vtype=GRB.BINARY, name='c')
    q = model.addVars(i_to_j, vtype=GRB.CONTINUOUS, lb=-2 * pi, ub=2 * pi, name='q')
    gen_p = model.addVars(gen, vtype=GRB.CONTINUOUS, lb=0.0, name='gen_p')

    # 支路功率约束
    for (u, v), i in ij.items():
        model.addConstr(
            brh_p[(u, v)] == 1 / (pp_net.line['x_ohm_per_km'].iloc[i] * pp_net.sn_mva / (
                    pp_net.bus['vn_kv'].iloc[0] ** 2)) * (
                    q[(u, v)] - q[(v, u)]), name=f"支路{u}->{v}功率约束")
        model.addConstr(brh_p[(u, v)] == - brh_p[(v, u)], name=f"支路{v}->{u}功率约束")

    # 节点功率约束
    load_info = pp_net.load.copy()
    load_info = load_info.set_index('bus')
    gen_info_ext = pp_net.ext_grid[['bus', 'max_p_mw', 'min_p_mw']]
    gen_info_gen = pp_net.gen[['bus', 'max_p_mw', 'min_p_mw', 'p_mw']]
    gen_info = pd.concat([gen_info_ext, gen_info_gen])
    gen_info = gen_info.set_index('bus')
    for i in bus:
        if i in load:  # 负荷节点，功率输入和节点负荷平衡
            model.addConstr(
                quicksum([brh_p[(i, j)] for j in adj[i]]) == - c[i] * load_info['p_mw'].loc[i] / pp_net.sn_mva,
                name=f"负荷节点{i}功率平衡")
        elif i in gen:  # 发电机节点，功率输入+负荷和输出平衡
            gen_load = 0.0
            try:
                gen_load = load_info['p_mw'].loc[i] / pp_net.sn_mva
            except KeyError:
                ...
            model.addConstr(
                quicksum([brh_p[(i, j)] for j in adj[i]]) == gen_p[i] - c[i] * gen_load, name=f'发电机节点{i}功率平衡')

            if not np.isnan(gen_info['p_mw'].loc[i]):
                model.addConstr(gen_p[i] == c[i] * gen_info['p_mw'].loc[i] / pp_net.sn_mva, name=f'PV节点{i}功率约束')
            else:
                model.addConstr(gen_p[i] <= gen_info['max_p_mw'].loc[i] * c[i] / pp_net.sn_mva, name=f'发电机{i}上限')
                model.addConstr(gen_p[i] >= gen_info['min_p_mw'].loc[i] * c[i] / pp_net.sn_mva, name=f'发电机{i}下限')
        else:
            model.addConstr(quicksum([brh_p[(i, j)] for j in adj[i]]) == 0.0, name=f'无负荷发电节点{i}约束')

    # 相角约束辅助变量
    for (u, v) in ij:
        model.addConstr(q[(u, v)] <= 2 * pi * z[tuple(sorted((u, v)))], name=f'辅助变量{u}-{v}')
        model.addConstr(q[(u, v)] >= -2 * pi * z[tuple(sorted((u, v)))], name=f'辅助变量{u}-{v}')
        model.addConstr(ang[u] - q[(u, v)] <= 2 * pi * (1 - z[tuple(sorted((u, v)))]), name=f'辅助变量{u}-{v}')
        model.addConstr(ang[u] - q[(u, v)] >= -2 * pi * (1 - z[tuple(sorted((u, v)))]), name=f'辅助变量{u}-{v}')
        model.addConstr(q[(v, u)] <= 2 * pi * z[tuple(sorted((u, v)))], name=f'辅助变量{v}-{u}')
        model.addConstr(q[(v, u)] >= -2 * pi * z[tuple(sorted((u, v)))], name=f'辅助变量{v}-{u}')
        model.addConstr(ang[v] - q[(v, u)] <= 2 * pi * (1 - z[tuple(sorted((u, v)))]), name=f'辅助变量{v}-{u}')
        model.addConstr(ang[v] - q[(v, u)] >= -2 * pi * (1 - z[tuple(sorted((u, v)))]), name=f'辅助变量{v}-{u}')

    # 节点连通性约束
    model.addConstrs((c[i] == 1 for i in gen if adj[i]), name='电源连通')

    for i in bus:
        model.addConstr(c[i] <= quicksum([z[tuple(sorted((i, j)))] for j in adj[i]]), name=f'节点{i}连通上界')
        model.addConstrs((c[i] >= z[tuple(sorted((i, j)))] for j in adj[i]), name=f'节点{i}连通下界')

    # Vθ节点相角约束
    model.addConstr(ang[gen_info_ext['bus'].iloc[0]] == 0.0, name='slack_angle')

    # 支路功率越限
    model.addConstr(z[tuple(sorted((line_from, line_to)))] == 1.0)
    line_tol = ext_net.line['tolerance'].iloc[break_line_index]
    FORMULATION = formulation  # 0:广义约束  1:大M法  2 最大值形式  3a from -> to overload 3b to -> from overload
    if FORMULATION == "0":  # 广义约束
        y = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='GenAbs')
        model.addGenConstrAbs(y, brh_p[(line_from, line_to)], name='Overflow Aux Var')
        model.addConstr(y >= line_tol / pp_net.sn_mva, name='Overflow Tolerance')
    elif FORMULATION == "1":  # 大M法
        m_aux = model.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, name="Big-M Formulation")
        M = 2 * line_tol / pp_net.sn_mva
        model.addConstr(brh_p[(line_from, line_to)] + M * m_aux >= line_tol / pp_net.sn_mva)
        model.addConstr(brh_p[(line_from, line_to)] + M * (1 - m_aux) >= line_tol / pp_net.sn_mva)
    elif FORMULATION == "2":
        M = 3 * line_tol / pp_net.sn_mva
        u = model.addVars([1, 2], vtype=GRB.BINARY)
        max_aux = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="max of branch flow")
        model.addConstr(brh_p[(line_from, line_to)] <= max_aux)
        model.addConstr(brh_p[(line_to, line_from)] <= max_aux)
        model.addConstr(brh_p[(line_from, line_to)] >= max_aux - M * (1 - u[1]))
        model.addConstr(brh_p[(line_to, line_from)] >= max_aux - M * (1 - u[2]))
        model.addConstr(quicksum(u) >= 1)
        model.addConstr(max_aux >= line_tol / pp_net.sn_mva)
    elif FORMULATION == "3a":
        model.addConstr(brh_p[(line_from, line_to)] >= line_tol / pp_net.sn_mva, name='Overflow Tolerance')
    elif FORMULATION == "3b":
        model.addConstr(brh_p[(line_from, line_to)] <= -line_tol / pp_net.sn_mva, name='Overflow Tolerance')
    else:
        ...
    # 目标函数
    model.setObjective(
        quicksum([(1 - z[tuple(sorted((u, v)))]) * np.log(1 - rel[i]) for (u, v), i in ij.items()]) + quicksum(
            [z[tuple(sorted((u, v)))] * np.log(rel[i]) for (u, v), i in ij.items()]),
        sense=GRB.MINIMIZE)

    try:
        for cursor, zz in enumerate(ij):
            z[zz].Start = np.round(warm_init[break_line_index][cursor])
        model.NumStart = 1
    except KeyError:
        model.NumStart = 0

    model.update()
    model.optimize()
    # model.tune()
    if model.status == grb.GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        # do IIS, find infeasible constraints
        # model.computeIIS()
        # for c in model.getConstrs():
        #     if c.IISConstr:
        #         print('%s' % c.constrName)
        return None, None, 0.0
    else:
        break_lines = []
        if verbose == 1:  # 输出表格
            brh_tb = pt.PrettyTable()
            brh_tb.field_names = ["BRH#", "From Bus", "To Bus", "Status", "P(MW)"]
            for (u, v), i in ij.items():
                brh_tb.add_row(
                    [i, u, v, "DSC" if z[(u, v)].X < 0.5 else "CON", f'{np.abs(brh_p[(u, v)].X) * pp_net.sn_mva:.3f}'])
                if z[(u, v)].X < 0.5:
                    break_lines.append(i)
            print(brh_tb)

            bus_tb = pt.PrettyTable()
            bus_tb.field_names = ["BUS#", "Type", "Ang(deg.)", "CONNECTION", "Power Generation", "Power Demand",
                                  'Power Insert']
            for i in bus:
                this_type = type_dict.get(i, "PQ")
                if i in load:
                    this_type = "PQ"
                    bus_tb.add_row(
                        [i, this_type, f'{ang[i].X * 180 / pi: .3f}', c[i].X == 1.0, "-", load_info['p_mw'].loc[i],
                         (sum([brh_p[(i, j)].X for j in adj[i]]) - sum([brh_p[(j, i)].X for j in adj[i]])) *
                         pp_net.sn_mva])
                elif i in gen:
                    temp = "-"
                    try:
                        temp = str(load_info['p_mw'].loc[i])
                    except KeyError:
                        ...
                    bus_tb.add_row(
                        [i, this_type, f'{ang[i].X * 180 / pi: .3f}', c[i].X == 1.0, gen_p[i].X * pp_net.sn_mva, temp,
                         (sum([brh_p[(i, j)].X for j in adj[i]]) - sum([brh_p[(j, i)].X for j in adj[i]])) *
                         pp_net.sn_mva])
                else:
                    bus_tb.add_row([i, this_type, f'{ang[i].X * 180 / pi: .3f}', c[i].X == 1.0, "-", "-",
                                    (sum([brh_p[(i, j)].X for j in adj[i]]) - sum([brh_p[(j, i)].X for j in adj[i]])) *
                                    pp_net.sn_mva])
            print(bus_tb)

            gen_tb = pt.PrettyTable()
            gen_tb.field_names = ["Gen At", "Min", "Max", "Power"]

            for i in sorted(gen):
                gen_tb.add_row(
                    [i, gen_info['min_p_mw'].loc[i], gen_info['max_p_mw'].loc[i],
                     f'{gen_p[i].X * pp_net.sn_mva:.3f}'])
            print(gen_tb)
        print(f'number of solution stored:{model.SolCount}')
        final_prob = []
        for e in range(model.SolCount):
            for line in range(len(pp_net.line)):
                u, v = pp_net.line['from_bus'].iloc[line], pp_net.line['to_bus'].iloc[line]
                if np.abs(brh_p[(u, v)].Xn) >= pp_net.line['tolerance'].iloc[line] / pp_net.sn_mva:
                    warm_init[line] = [z[zind].Xn for zind in ij]
            model.setParam(GRB.Param.SolutionNumber, e)
            # print('%g' % model.PoolObjVal, end=' ')
            # print(f'{[z[i].Xn for i in ij]}')
            final_prob.append(np.exp(model.PoolObjVal))
        # plt.plot(range(len(final_prob)), np.log(final_prob))
        # plt.plot(range(len(final_prob)), fp)
        # plt.show()
        # plt.xlabel("故障场景编号", fontsize=14)
        # plt.ylabel("故障阶数", fontsize=14)
        # print(f'\nFinal Prob = {np.sum(final_prob)}')
        return model, break_lines, np.sum(final_prob)


# 回调函数
def other_line_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        if model.cbGet(GRB.Callback.MIPSOL_OBJ) < -20.73:
            model.terminate()


if __name__ == "__main__":
    import pandapower.converter as pc
    from pypower.api import rundcpf
    import time

    warm_init = {}
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ext_net = case30()
    net_vn = ext_net.bus['vn_kv'].iloc[0]
    # rel = list(np.random.rand(len(ext_net.line)))
    rel = [0.9] * len(ext_net.line)
    # m, brk_lines, _ = basic_model(ext_net, rel, 0, 1, 10, pool_search_method=2, pool_num=10)
    # rundcpf(pc.to_ppc(ext_net, init='flat'))
    # pp.rundcpp(ext_net)
    prob = np.zeros(len(ext_net.line))
    st = time.time()
    # for i in range(len(ext_net.line)):
    ext_net.line['tolerance'] = ext_net.line['max_i_ka'] * net_vn * np.sqrt(3)
    for i in range(len(ext_net.line)):
        # for i in [31]:
        f_bus, t_bus = ext_net.line['from_bus'].iloc[i], ext_net.line['to_bus'].iloc[i]
        _, _, prob[i] = basic_model(ext_net, rel, f_bus, t_bus, i, pool_num=10, pool_search_method=2,
                                    warm_init=warm_init, verbose=0, formulation="0")
    total_time = time.time() - st
    print(f'计算用时{total_time}')
    # exp_prob = np.expm1(prob)
    # total_prob = 1.0 - np.array(rel) + prob
    # m, brk_lines = basic_model(ext_net, list(np.random.rand(len(ext_net.line))), 5, 8, 55)
    # if brk_lines is not None:
    #     ext_net.line.drop(index=brk_lines, inplace=True)
    #     ppc = pc.to_ppc(ext_net, init='flat', check_connectivity=True)
    #     pp_res = rundcpf(ppc)
    # pp.rundcpp(ext_net)

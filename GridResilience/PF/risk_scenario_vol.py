import gurobipy as grp
import pandapower as pp
from pypower.api import makeYbus, makeB, makeSbus, makeBdc
import networkx as nx
import numpy as np
from gurobipy import quicksum
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib


def linearized_risk_scenario_search(pp_net: pp.pandapowerNet, rel, vol_fact, vol_quad, o=10, if_plot=False):
    assert len(rel) == len(pp_net.line)
    plt.style.use(['science', 'no-latex', 'std-colors'])
    plt.rcParams['font.family'] = 'STsong'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.unicode_minus'] = False
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

    # 标幺制基值
    BaseMVA = pp_net.sn_mva
    BaseZ = (pp_net.bus['vn_kv'].iloc[0] ** 2) / BaseMVA

    # 模型建立
    model = grp.Model(name="Risk scenario selection")
    ang = model.addVars(pp_net.bus.index, vtype=grp.GRB.CONTINUOUS, ub=2 * np.pi, lb=-2 * np.pi)  # 节点相角
    vol = model.addVars(pp_net.bus.index, vtype=grp.GRB.CONTINUOUS, ub=1.5, lb=0.0)  # 节点电压
    Pij = model.addVars(i_to_j, vtype=grp.GRB.CONTINUOUS, lb=-grp.GRB.INFINITY, ub=grp.GRB.INFINITY)
    Qij = model.addVars(i_to_j, vtype=grp.GRB.CONTINUOUS, lb=-grp.GRB.INFINITY, ub=grp.GRB.INFINITY)
    Pg = model.addVars(pp_net.bus.index, vtype=grp.GRB.CONTINUOUS, ub=grp.GRB.INFINITY, lb=0.0)
    Qg = model.addVars(pp_net.bus.index, vtype=grp.GRB.CONTINUOUS, ub=grp.GRB.INFINITY, lb=-grp.GRB.INFINITY)
    z = model.addVars(ij, vtype=grp.GRB.BINARY, name='z')  # 支路开断状态变量
    c = model.addVars(pp_net.bus.index, vtype=grp.GRB.BINARY, name='c')  # 节点状态变量
    z_ang = model.addVars(i_to_j, vtype=grp.GRB.CONTINUOUS, lb=-2 * np.pi, ub=2 * np.pi)
    z_vol = model.addVars(i_to_j, vtype=grp.GRB.CONTINUOUS, lb=0.0, ub=1.5)

    # model.setParam('PoolSearchMode', 2)
    model.setParam('PoolSolutions', 1)
    model.setParam('IntegralityFocus', 1)
    model.setParam('IntFeasTol', 1e-9)

    # 支路有/无功功率  Branch traverse
    for u, v in ij:
        s_uv = tuple(sorted((u, v)))
        this_line = ij[s_uv]
        r = pp_net.line['r_ohm_per_km'].iloc[this_line] * pp_net.line['length_km'].iloc[this_line] / BaseZ
        x = pp_net.line['x_ohm_per_km'].iloc[this_line] * pp_net.line['length_km'].iloc[this_line] / BaseZ
        y = 1 / (r + x * 1j)
        g = y.real
        b = y.imag
        vol_factor = vol_fact[0]
        vol_quadric = vol_factor ** 2
        model.addConstr(
            Pij[(u, v)] == vol_factor * g * (z_vol[(u, v)] - z_vol[(v, u)]) - vol_quadric * b * (
                    z_ang[(u, v)] - z_ang[(v, u)]))
        model.addConstr(
            Qij[(u, v)] == - vol_quadric * g * (z_ang[(u, v)] - z_ang[(v, u)]) - vol_factor * b * (
                    z_vol[(u, v)] - z_vol[(v, u)]))
        model.addConstr(Pij[(u, v)] == - Pij[(v, u)])
        model.addConstr(Qij[(u, v)] == - Qij[(v, u)])


        # 辅助变量约束
        model.addConstr(z_ang[(u, v)] <= 2 * np.pi * z[s_uv])
        model.addConstr(z_ang[(u, v)] >= -2 * np.pi * z[s_uv])
        model.addConstr(ang[u] - z_ang[(u, v)] <= 2 * np.pi * (1 - z[s_uv]))
        model.addConstr(ang[u] - z_ang[(u, v)] >= -2 * np.pi * (1 - z[s_uv]))
        model.addConstr(z_ang[(v, u)] <= 2 * np.pi * z[s_uv])
        model.addConstr(z_ang[(v, u)] >= -2 * np.pi * z[s_uv])
        model.addConstr(ang[v] - z_ang[(v, u)] <= 2 * np.pi * (1 - z[s_uv]))
        model.addConstr(ang[v] - z_ang[(v, u)] >= -2 * np.pi * (1 - z[s_uv]))

        model.addConstr(z_vol[(u, v)] <= 1.5 * z[s_uv])
        model.addConstr(z_vol[(u, v)] >= 0.0)
        model.addConstr(vol[u] - z_vol[(u, v)] <= 1.5 * (1 - z[s_uv]))
        model.addConstr(vol[u] - z_vol[(u, v)] >= 0.0)
        model.addConstr(z_vol[(v, u)] <= 1.5 * z[s_uv])
        model.addConstr(z_vol[(v, u)] >= 0.0)
        model.addConstr(vol[v] - z_vol[(v, u)] <= 1.5 * (1 - z[s_uv]))
        model.addConstr(vol[v] - z_vol[(v, u)] >= 0.0)

    # 节点有功/无功

    load_info = pp_net.load.copy()
    load_info = load_info.set_index('bus')

    gen_info_ext = pp_net.ext_grid[['bus', 'max_p_mw', 'min_p_mw', 'max_q_mvar', 'min_q_mvar']]
    try:
        gen_info_gen = pp_net.gen[['bus', 'max_p_mw', 'min_p_mw', 'p_mw', 'max_q_mvar', 'min_q_mvar']]
    except KeyError:
        gen_info_gen = pp_net.gen[['bus', 'p_mw']]
    gen_info = pd.concat([gen_info_ext, gen_info_gen])
    gen_info = gen_info.set_index('bus')
    for i in bus:
        if i in gen:  # 发电机节点，功率输入+负荷和输出平衡
            try:
                gen_p = load_info['p_mw'].loc[i] / BaseMVA
                gen_q = load_info['q_mvar'].loc[i] / BaseMVA
            except KeyError:
                gen_p = 0.0
                gen_q = 0.0
            model.addConstr(
                quicksum([Pij[(i, j)] for j in adj[i]]) == Pg[i] - c[i] * gen_p, name=f'发电机节点{i}有功功率平衡')
            model.addConstr(
                quicksum([Qij[(i, j)] for j in adj[i]]) == Qg[i] - c[i] * gen_q, name=f'发电机节点{i}无功功率平衡')

            if not np.isnan(gen_info['p_mw'].loc[i]):
                model.addConstr(Pg[i] == c[i] * gen_info['p_mw'].loc[i] / BaseMVA, name=f'PV节点{i}有功功率约束')
                model.addConstr(Qg[i] <= c[i] * gen_info['max_q_mvar'].loc[i] / BaseMVA, name=f'PV节点{i}无功上限')
                model.addConstr(Qg[i] >= c[i] * gen_info['min_q_mvar'].loc[i] / BaseMVA, name=f'PV节点{i}无功下限')
            else:
                try:
                    model.addConstr(Pg[i] <= gen_info['max_p_mw'].loc[i] * c[i] / BaseMVA, name=f'发电机{i}有功上限')
                    model.addConstr(Pg[i] >= gen_info['min_p_mw'].loc[i] * c[i] / BaseMVA, name=f'发电机{i}有功下限')
                    model.addConstr(Qg[i] <= gen_info['max_q_mvar'].loc[i] * c[i] / BaseMVA, name=f'发电机{i}有功上限')
                    model.addConstr(Qg[i] >= gen_info['min_q_mvar'].loc[i] * c[i] / BaseMVA, name=f'发电机{i}有功下限')
                except KeyError:
                    ...
        elif i in load:  # 负荷节点，功率输入和节点负荷平衡
            model.addConstr(
                quicksum([Pij[(i, j)] for j in adj[i]]) == - c[i] * load_info['p_mw'].loc[i] / BaseMVA,
                name=f"负荷节点{i}有功功率平衡")
            model.addConstr(
                quicksum([Qij[(i, j)] for j in adj[i]]) == - c[i] * load_info['q_mvar'].loc[i] / BaseMVA,
                name=f"负荷节点{i}无功功率平衡")
            model.addConstr(Pg[i] == 0.0)
            model.addConstr(Qg[i] == 0.0)
        else:
            model.addConstr(quicksum([Pij[(i, j)] for j in adj[i]]) == 0.0, name=f'无负荷发电节点{i}有功约束')
            model.addConstr(quicksum([Qij[(i, j)] for j in adj[i]]) == 0.0, name=f'无负荷发电节点{i}无功约束')
            model.addConstr(Pg[i] == 0.0)
            model.addConstr(Qg[i] == 0.0)

    # 节点连通性约束
    model.addConstrs((gen_info['max_p_mw'].loc[i] * c[i] >= Pg[i] for i in gen if adj[i]), name='电源连通')

    for i in pp_net.bus.index:
        if i not in gen:
            model.addConstr(c[i] <= quicksum([z[tuple(sorted((i, j)))] for j in adj[i]]), name=f'节点{i}连通上界')
            model.addConstrs((c[i] >= z[tuple(sorted((i, j)))] for j in adj[i]), name=f'节点{i}连通下界')
            model.addConstr(10 * c[i] >= vol[i])
            model.addConstr(-2 * np.pi * c[i] <= ang[i])
            model.addConstr(2 * np.pi * c[i] >= ang[i])

    # Vθ节点相角约束
    for i in range(len(gen_info_ext)):
        node = gen_info_ext['bus'].iloc[i]
        model.addConstr(ang[node] == 0.0, name=f'松弛节点{node}相角约束')
        model.addConstr(vol[node] == pp_net.ext_grid['vm_pu'].iloc[i], f'松弛节点{node}电压约束')
    for i in range(len(gen_info_gen)):
        node = gen_info_gen['bus'].iloc[i]
        model.addConstr(vol[node] == pp_net.gen['vm_pu'].iloc[i])

    # 电压越限约束
    model.addConstr(vol[o] <= 0.90)
    model.addConstr(vol[o] >= 0.88)

    # 配电网辐射状约束
    dist = True
    if dist:
        bd = model.addVars(i_to_j, vtype=grp.GRB.BINARY)
        for i in gen:
            model.addConstrs(bd[(j, i)] == 0 for j in adj[i])
        for i in load:
            model.addConstr(grp.quicksum([bd[j, i] for j in adj[i]]) == 1)
        for (u, v) in ij:
            model.addConstr(z[(u, v)] == bd[(u, v)] + bd[(v, u)])

    # 功率越限约束
    # model.addConstr(Pij[(1, 2)] ** 2 + Qij[(1, 2)] ** 2 >= 4.10 ** 2 / BaseMVA, name='Overflow Aux Var')

    model.setObjective(
        quicksum([(1 - z[tuple(sorted((u, v)))]) * np.log(1 - rel[i]) for (u, v), i in ij.items()]) + quicksum(
            [z[tuple(sorted((u, v)))] * np.log(rel[i]) for (u, v), i in ij.items()]), sense=grp.GRB.MAXIMIZE)

    model.update()
    model.optimize()
    if model.status == grp.GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
    else:
        final_prob = []
        for e in range(model.SolCount):
            model.setParam(grp.GRB.Param.SolutionNumber, e)
            print(np.exp(model.PoolObjVal))
            final_prob.append(np.exp(model.PoolObjVal))
            print(f'拓扑组合{[i for i in z if z[i].Xn < 0.5]}')
        print(f'总概率{np.sum(final_prob)}')
        pp_net.line['in_service'] = pd.Series([bool(z[i].Xn) for i in z])
        dc_net = deepcopy(pp_net)
        if if_plot:
            fig = plt.figure(figsize=(20, 13))
            pp.rundcpp(dc_net)
            pp.runpp(pp_net)
            plt.subplot(321)
            plt.scatter(bus, [vol[i].Xn for i in bus], label="DLPF")
            plt.plot(bus, pp_net.res_bus['vm_pu'], label='ACPF')
            plt.plot(bus, [1.0 for _ in bus], label='DCPF')
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            plt.title("节点电压幅值 (p.u.)")
            plt.subplot(322)
            error_series = 100 * np.subtract([vol[i].Xn for i in bus], pp_net.res_bus['vm_pu']) / pp_net.res_bus[
                'vm_pu']
            error_series.plot.bar(label="DLPF", rot=0)
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            plt.title("节点电压幅值相对误差 (%)")
            plt.subplot(323)
            branch_p = np.array([np.abs(Pij[(u, v)].Xn * BaseMVA) for (u, v) in ij])
            branch_q = np.array([np.abs(Qij[(u, v)].Xn * BaseMVA) for (u, v) in ij])
            branch_s = np.sqrt(branch_p ** 2 + branch_q ** 2)
            pp_p = np.abs(pp_net.res_line['p_from_mw'])
            pp_q = np.abs(pp_net.res_line['q_from_mvar'])
            pp_s = np.sqrt(pp_p ** 2 + pp_q ** 2)
            plt.scatter(range(len(pp_net.line)), branch_s, label="DLPF")

            plt.plot(range(len(pp_net.line)), pp_s, label="ACPF")
            plt.plot(range(len(pp_net.line)), np.abs(dc_net.res_line['p_from_mw']), label="DCPF")
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            plt.title("支路复功率 (MVA)")
            plt.subplot(324)
            error_dlpf = np.subtract(branch_s, pp_s)
            error_dlpf.plot.bar(label="DLPF", rot=0)
            plt.title("支路复功率误差 (MVA)")
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            plt.subplot(325)
            plt.scatter(bus, [ang[i].Xn * 180 / np.pi for i in bus], label="DLPF")
            plt.plot(bus, pp_net.res_bus['va_degree'], label='ACPF')
            plt.plot(bus, dc_net.res_bus['va_degree'], label='DCPF')
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            plt.title("节点电压相角 (度)")
            plt.subplot(326)
            ang_error = np.subtract([ang[i].Xn * 180 / np.pi for i in bus], pp_net.res_bus['va_degree'])
            ang_error.plot.bar(label="DLPF", rot=0)
            plt.title("节点电压相角误差 (度)")
            plt.xticks(fontproperties='Times New Roman', size=14)
            plt.yticks(fontproperties='Times New Roman', size=14)
            plt.legend()
            # plt.savefig('D:\\DocumentFiles\\FINALTHINGS\\毕设文档\\FinalReport\\figures\\IEEE33OutagePF.png', dpi=300)
            plt.show()
            pp_net.line = pp_net.line.drop([ij[i] for i in z if z[i].Xn < 0.5])
            pp.runpp(pp_net)
        vol_fact[0] = np.average(np.array([vol[i].X for i in vol]))
        for u, v in ij:
            s_uv = tuple(sorted((u, v)))
            this_line = ij[s_uv]
            vol_quad[this_line] = vol[u].X * vol[v].X


if __name__ == "__main__":
    from pandapower.networks import case33bw, case30

    net = case33bw()
    net.load['p_mw'] *= 1.3
    net.load['q_mvar'] *= 1.3

    rel = 0.5 + 0.5 * np.random.rand(len(net.line))

    vol_fac = [1.0]
    vol_qua = [1.0 for _ in range(len(net.line))]
    vol_fac_last_step = [0.0]
    iter_max = 20
    iter_count = 0
    while iter_count < iter_max:
        if np.abs(vol_fac[0] - vol_fac_last_step[0]) <= 1e-4:
            linearized_risk_scenario_search(net, rel, vol_fac, vol_qua, o=10, if_plot=True)
            break
        else:
            plot_option = iter_count == iter_max - 1
            vol_fac_last_step[0] = vol_fac[0]
            linearized_risk_scenario_search(net, rel, vol_fac, vol_qua, o=10, if_plot=plot_option)
            iter_count += 1

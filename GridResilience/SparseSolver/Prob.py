import random
from copy import deepcopy
import networkx as nx
import pandas as pd
import ray
from tqdm import tqdm
import numpy as np
import ray.util.multiprocessing as mp
from scipy.sparse import coo_matrix, spmatrix, identity
from scipy.sparse.linalg import inv, spsolve, gmres
from numpy.linalg import norm
from math import prod
from GridResilience.Environment import *
import jax.numpy as jnp

def _create_fun_c(uv, x_vec, fun_vec, nbr):
    fun_vec[uv] = prod([x_vec[nb] for nb in nbr])


def _create_function_with_ind(fun_vec, x_vec, adj, a, b, c, n):
    local_x = 1 - x_vec
    for k in range(len(c)):
        _create_fun_c(c[k], local_x, fun_vec, n[k])
    # numpy element-wise operations to accelerate vector construction
    if len(a) != 0:
        fun_vec[a] = -x_vec[a]
    if len(b) != 0:
        fun_vec[b] = adj[b] - x_vec[b]
    if len(c) != 0:
        fun_vec[c] = adj[c] * (1 - fun_vec[c]) - x_vec[c]


def _create_function_index(source_list, G_edge_dict, adj_dict):
    a_list, b_list, c_list, nbr_list = [], [], [], []
    a_append = a_list.append
    b_append = b_list.append
    c_append = c_list.append
    nbr_append = nbr_list.append
    for i, (u, v) in enumerate(G_edge_dict):
        if u in source_list:
            a_append(i)
        elif v in source_list:
            b_append(i)
        else:
            c_append(i)
            nbr_append([G_edge_dict[(v, nbr)] for nbr in adj_dict[v] if nbr != u])
    return np.array(a_list), np.array(b_list), np.array(c_list), nbr_list


def prob_multi_station(filePath: str, station_list: list):
    ...


def prob_multi_period(case: GridCase, funtol=1e-5, itermax=100, t_list=None, need_id=True):
    """
    多时段求解概率
    :param case: 算例对象
    :param funtol: 方程求解精度，默认为1e-5
    :param itermax: 方程求解最大迭代次数，默认为100
    :param t_list: 方程求解时段范围，默认为None，设置时采用list对象
    :return:
    """
    _local_graph = nx.relabel_nodes(case.graph, case.bus_dict, copy=True)
    adj_data = _adj_data(case, _local_graph)

    def remote_prob(t):
        return _solve_with_data(case, funtol, itermax, t, adj_data)

    pool = mp.Pool()
    if t_list is None:
        prob_list = pool.map_async(remote_prob, range(np.size(case.linesafe, axis=1))).get()
        prob_all = np.concatenate([p[0] for p in prob_list], axis=1)
        jac_all = [p[1] for p in prob_list]
        if need_id:
            df = pd.DataFrame(data=prob_all, index=case.bus[:, BUS_I],
                              columns=[f"{ts}" for ts in range(np.size(case.linesafe, axis=1))])
            df.index.name = "id"
            df.columns.name = "Time"
            return df, jac_all
        else:
            return prob_all, jac_all
    else:
        try:
            prob_list = pool.map_async(remote_prob, list(t_list)).get()
            prob_all = np.concatenate([p[0] for p in prob_list], axis=1)
            jac_all = [p[1] for p in prob_list]
            return prob_all, jac_all
        except ValueError:
            print('时间段设置有误')


def _adj_data(case: GridCase, local_graph):
    source_list = set(case.source_idx[0])  # 电源节点列表
    G = local_graph  # 读取算例的networkx网络图
    adj = G.adj
    adj_dict = [None for _ in range(len(adj))]  # 算例的邻接字典
    G_edge_list = [e for e in G.edges]  # 算例的边列表
    G_edge_dict = {}  # 算例的边编号字典映射
    bus_num = np.size(case.bus, axis=0)
    for k in adj:
        adj_dict[k] = set([nbr for nbr in adj[k]])
    x_row = []
    x_col = []
    for ind, edge in enumerate(G_edge_list):
        x, y = edge[0], edge[1]
        x_row.append(x)
        x_row.append(y)
        x_col.append(y)
        x_col.append(x)
        G_edge_dict[(x, y)] = ind * 2
        G_edge_dict[(y, x)] = ind * 2 + 1
    a_l, b_l, c_l, n_l = _create_function_index(source_list, G_edge_dict, adj_dict)
    return source_list, adj_dict, G_edge_list, G_edge_dict, bus_num, G, a_l, b_l, c_l, n_l


def _solve_with_data(case, funtol, itermax, t, data_tuple=None, need_x=False, need_f=False):
    sparse_jac = None

    if data_tuple is None:
        _local_graph = nx.relabel_nodes(case.graph, case.bus_dict, copy=True)
        data_tuple = _adj_data(case, _local_graph)
    source_list, adj_dict, G_edge_list, G_edge_dict, bus_num, G, a_type, b_type, c_type, nbr_l = data_tuple
    ls = 1 - case.linesafe[:, t]
    ea = case.edge_array
    # 第一部分：生成邻接矩阵数据
    full_num = len(G_edge_dict)
    data = np.ones(full_num)
    for ind, (u, v) in enumerate(ea):
        data[G_edge_dict[(u, v)]] *= ls[ind]
        data[G_edge_dict[(v, u)]] *= ls[ind]
    data = 1.0 - data
    # 第二部分：生成函数矢量
    x = np.ones(full_num)
    fun = np.zeros(full_num)
    _create_function_with_ind(fun, x, data, a_type, b_type, c_type, nbr_l)
    # 第三部分：迭代求解部分
    iter_count = 0
    # Jacobian 矩阵索引确定
    jac_rows = []
    jac_cols = []
    row_append = jac_rows.append
    col_append = jac_cols.append
    for ind in range(full_num // 2):
        i, j = G_edge_list[ind]
        if j not in source_list:
            for k in adj_dict[j]:
                if k != i:
                    row_append(G_edge_dict[(i, j)])
                    col_append(G_edge_dict[(j, k)])
        if i not in source_list:
            for k in adj_dict[i]:
                if k != j:
                    row_append(G_edge_dict[(j, i)])
                    col_append(G_edge_dict[(i, k)])
    full_ind = range(full_num)
    jac_cols.extend(full_ind)
    jac_rows.extend(full_ind)
    jac_data = np.zeros(len(jac_cols))
    jac_index = []
    jac_R = []
    R_append = jac_R.append
    cursor = 0
    while norm(fun) > funtol and iter_count < itermax:
        x = 1 - x
        if sparse_jac is None:
            for ind, (i, j) in enumerate(G_edge_list):
                ij = 2 * ind
                local_data = data[ij]
                if j not in source_list:
                    for k in adj_dict[j]:
                        if k != i:
                            jac_index.append([G_edge_dict[(j, tt)] for tt in adj_dict[j] if tt != k and tt != i])
                            jac_data[cursor] = local_data * prod(
                                [x[jac_ind] for jac_ind in jac_index[cursor]])
                            R_append(local_data)
                            cursor += 1
                if i not in source_list:
                    for k in adj_dict[i]:
                        if k != j:
                            jac_index.append([G_edge_dict[(i, tt)] for tt in adj_dict[i] if tt != k and tt != j])
                            jac_data[cursor] = local_data * prod(
                                [x[jac_ind] for jac_ind in jac_index[cursor]])
                            R_append(local_data)
                            cursor += 1
            jac_data[cursor:] = -1.0
            sparse_jac = coo_matrix((jac_data, (jac_rows, jac_cols)), shape=(full_num, full_num))
        else:
            for c, j_inds in enumerate(jac_index):
                jac_data[c] = prod([x[j_i] for j_i in j_inds])
            jac_data[:cursor] *= jac_R
            sparse_jac.data = jac_data
        dx = spsolve(sparse_jac.tocsc(), fun)  # 稀疏直接求解
        x = 1 - x - dx
        _create_function_with_ind(fun, x, data, a_type, b_type, c_type, nbr_l)
        iter_count += 1
    # 第四部分：节点概率计算部分
    p = np.zeros((bus_num, 1))
    x = 1 - x
    for i in range(bus_num):
        if i not in source_list:
            p[i] = prod([x[G_edge_dict[(i, nbr)]] for nbr in adj_dict[i]])
    p = 1 - p
    if need_x and not need_f:
        return p, sparse_jac, 1 - x
    elif need_f and not need_x:
        return p, sparse_jac, fun
    elif need_f and need_x:
        return p, sparse_jac, 1 - x, fun
    else:
        return p, sparse_jac


def prob_solver(case, funtol=1e-5, itermax=100, t=0, adj_data=None, need_id=True):
    """
    单一时刻求解边缘概率
    :param case: CPSCase 算例对象
    :param funtol: 方程求解精度，默认为1e-5
    :param itermax: 方程最大迭代次数，默认为100
    :param t: 方程求解时段，默认为0时刻
    :return: p: 节点有电概率
             jac: 最后一次计算的Jacobian矩阵
    """
    # Localize the graph and case data
    grp = nx.relabel_nodes(G=case.graph, mapping=case.bus_dict, copy=True)

    # Solving procedure
    if adj_data is None:
        p, jac = _solve_with_data(case, funtol, itermax, t, _adj_data(case, local_graph=grp))
    else:
        p, jac = _solve_with_data(case, funtol, itermax, t, adj_data)
    if need_id:
        df = pd.DataFrame(data=p, index=case.bus[:, BUS_I].astype(np.int64), columns=[f"{t}"])

        def _source_mapping():
            append_df = pd.DataFrame(data=[df.loc[case.contracted_mapping[cls]] for cls in case.contracted_mapping],
                                     index=case.contracted_mapping.keys(), columns=[f"{t}"])
            return pd.concat([df, append_df])

        if case.contracted_mapping:
            df = _source_mapping()
        df.index.name = "id"
        df.columns.name = "Time"
        return df, jac
    else:
        return p, jac


def risk_assessment(case: GridCase, res_array=None, funtol=1e-5, itermax=100, t=0):
    grp = nx.relabel_nodes(G=case.graph, mapping=case.bus_dict, copy=True)
    if res_array is None:
        res_array, _ = _solve_with_data(case, funtol, itermax, t, _adj_data(case, grp))
    return np.sum(case.bus[:, PLOAD] * case.bus[:, VALUE] * (1 - res_array.flatten()))


def _perturbation_for_edges(case: GridCase, funtol=1e-5, itermax=100, t=0):
    _local_graph = nx.relabel_nodes(case.graph, case.bus_dict, copy=True)
    adj = _adj_data(case, _local_graph)
    ref_p, _ = _solve_with_data(case, funtol, itermax, t, adj)
    ref_risk = risk_assessment(case, res_array=ref_p)
    disturbance = 1e-8
    pool = mp.Pool()

    def _perturbation(i):
        test_case = deepcopy(case)
        test_case.linesafe[i, t] -= disturbance
        p, _ = _solve_with_data(test_case, funtol, itermax, t, adj)
        return np.abs(
            ref_risk - risk_assessment(case, res_array=p)) / disturbance

    partial_risk = pool.map_async(_perturbation, tqdm(range(len(case.branch)))).get()

    return np.array(partial_risk)


def _analytical_sensitivity(case: GridCase, funtol=1e-5, itermax=100, t=0, adj_data=None, solver_result=None,
                            node_wise=False, risk=False):
    bus_rev = dict(zip(range(case.bus.shape[0]), case.bus[:, BUS_I].tolist()))
    if adj_data is None:
        _local_graph = nx.relabel_nodes(case.graph, case.bus_dict, copy=True)
        adj = _adj_data(case, _local_graph)
    else:
        adj = adj_data
    if solver_result is None:
        ref_p, ref_jac, x = _solve_with_data(case, funtol, itermax, t, adj, need_x=True)
    else:
        ref_p, ref_jac, x = solver_result
    # 获取dx_i /d x_j -> dx_mat
    if isinstance(ref_jac, spmatrix):
        ref_jac = ref_jac.tocsc()
        dx_mat = -np.linalg.inv(ref_jac.toarray())
    else:
        raise ValueError('输入Jacobian矩阵非稀疏矩阵')
    source_list, adj_dict, G_edge_list, G_edge_dict, bus_num, G, a_type, b_type, c_type, nbr_l = adj
    full_num = len(G_edge_dict)
    data = np.ones(full_num)
    ls = 1 - case.linesafe[:, t]
    for ind, (u, v) in enumerate(case.edge_array):
        data[G_edge_dict[(u, v)]] *= ls[ind]
        data[G_edge_dict[(v, u)]] *= ls[ind]
    data = 1.0 - data
    sense = np.zeros((len(case.bus), len(case.branch)))
    if risk:
        value_mat = case.bus[:, PLOAD] * case.bus[:, VALUE]
    else:
        value_mat = np.ones((len(case.bus), 1))
    nk_dict = {}
    for n in range(len(case.bus)):
        nk_dict[n] = np.array([G_edge_dict[(n, k)] for k in adj_dict[n]])
    for i, j in G.edges:
        fbus, tbus = bus_rev[i], bus_rev[j]
        index_1 = np.where((case.branch[:, FBUS] == fbus) & (case.branch[:, TBUS] == tbus))[0]
        index_2 = np.where((case.branch[:, FBUS] == tbus) & (case.branch[:, TBUS] == fbus))[0]
        edge_ind = np.concatenate((index_1, index_2)).tolist()  # A set with single/multiple edges
        ij = G_edge_dict[(i, j)]
        ji = G_edge_dict[(j, i)]
        data_ind = G_edge_dict[(i, j)]
        this_data = data[data_ind]
        ji_data = x[ji] / this_data
        ij_data = x[ij] / this_data
        this = dx_mat[:, [ij, ji]]
        # 1. 计算单边灵敏度
        if i not in source_list and j not in source_list:
            for n in range(len(case.bus)):
                nk = nk_dict[n]
                j_temp, i_temp = np.sum(this[nk, :] / (1 - x[nk]).reshape(-1, 1), axis=0)
                sense[n, edge_ind] = ji_data * i_temp + ij_data * j_temp
        elif i in source_list and j not in source_list:
            for n in range(len(case.bus)):
                nk = nk_dict[n]
                j_temp, i_temp = np.sum(this[nk, :] / (1 - x[nk]).reshape(-1, 1), axis=0)
                sense[n, edge_ind] = i_temp + ij_data * j_temp
        elif i not in source_list and j in source_list:
            for n in range(len(case.bus)):
                nk = nk_dict[n]
                j_temp, i_temp = np.sum(this[nk, :] / (1 - x[nk]).reshape(-1, 1), axis=0)
                sense[n, edge_ind] = i_temp * ji_data + j_temp
        else:
            for n in range(len(case.bus)):
                nk = nk_dict[n]
                j_temp, i_temp = np.sum(this[nk, :] / (1 - x[nk]).reshape(-1, 1), axis=0)
                sense[n, edge_ind] = i_temp + j_temp

        # 2. 计算多重边灵敏度
        if len(edge_ind) > 1:
            for kk in edge_ind:
                sense[:, kk] *= prod([ls[others] for others in edge_ind if others != kk])

    for node in range(case.bus.shape[0]):
        sense[node, :] *= value_mat[node] * (1 - ref_p[node])
    if node_wise:
        return sense
    else:
        return np.sum(sense, axis=0)


def sensitivity(case, method='a', solver_result=None, node_wise=True, risk=False, need_id=False):
    """

    :param case: CPSCase 算例对象
    :param method: 'a'：解析法  'n'：数值法
    :param solver_result: 求解得到的信息（p, jac, x）
    :param node_wise: 是否返回对每个节点的灵敏度数值
    :param risk: 返回风险灵敏度(True)或概率灵敏度(False)
    :param need_id:
    :return:
    """
    if method == 'n':
        sense_res = _perturbation_for_edges(case)
    elif method == 'a':
        sense_res = _analytical_sensitivity(case, solver_result=solver_result, node_wise=node_wise, risk=risk)
    else:
        raise ValueError("输入方法参数有误，应为'n'(数值法)或'a'(解析法)")

    if need_id and node_wise:
        df = pd.DataFrame(sense_res)
        df.columns = case.branch[:, BRH_I]
        df.index = case.bus[:, BUS_I]

        def _source_mapping():
            append_df = pd.DataFrame(data=[df.loc[case.contracted_mapping[cls]] for cls in case.contracted_mapping],
                                     index=case.contracted_mapping.keys())
            return pd.concat([df, append_df])

        if case.contracted_mapping:
            df = _source_mapping()
        return df.sort_index()
    else:
        return sense_res


def merge_result(p):
    if p:
        total_df = pd.concat(p, axis=1, keys=range(len(p)))
        return total_df.sort_index()


def merge_multi_scenario_results(origin: GridCase, p, prob, t=0):
    res = merge_result(p).to_numpy()
    return np.sum(np.multiply(res, prob[:, t]), axis=1)


def merge_multi_scenario_sensitivity(sen, c: GridCase, tree_list, prob_res, tree_prob, node_wise=True, t=0,
                                     is_parallel=True):
    assert node_wise
    order = c.branch[:, BRH_I].tolist()
    ls = c.linesafe[:, t]
    loc_dict = dict(zip(order, range(c.branch.shape[0])))
    get_loc = np.vectorize(loc_dict.get, cache=True)

    def single_scenario(tree_res, data, tr, prb):
        for n in tr[1]['on']:
            data[n] = prb / ls[get_loc(n)]
        for f in tr[1]['off']:
            data[f] = -prb / (1 - ls[get_loc(f)])
        for s in tr[1]['semi']:
            for j in s:
                data[j] = prb / (1 - prod([1 - ls[get_loc(kk)] for kk in s])) * prod(
                    [1 - ls[get_loc(jj)] for jj in s if jj != j])
        return data[order] * tree_res

    if sen:
        if is_parallel:
            ss_parallel = ray.remote(single_scenario)
            sense_list = ray.get(
                [ss_parallel.remote(tree_prob[i, t], sen[i], tree_list[i], prob_res[i]) for i in range(len(tree_list))])
            return sense_list
        else:
            return [single_scenario(tree_prob[i, t], sen[i], tree_list[i], prob_res[i]) for i in range(len(tree_list))]


if __name__ == "__main__":
    from line_profiler import LineProfiler
    from GridResilience.GridCase import case4_loop, case_32_modified, case28_microgrid
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from mpl_toolkits.mplot3d import Axes3D
    import time

    # grp = nx.generators.random_tree(50)
    # node_list = list(grp.nodes)
    # for i in node_list:
    #     grp.add_edge(10010, i)
    import matplotlib

    grid = case_32_modified()
    # grid.linesafe = np.array([0.95, 0.9, 0.85, 0.77]).reshape(-1, 1)
    st = time.time()
    grp = nx.relabel_nodes(G=grid.graph, mapping=grid.bus_dict, copy=True)

    data = _adj_data(grid, local_graph=grp)
    prob, _ = prob_solver(grid, adj_data=data)
    # res = sensitivity(grid, node_wise=True)

    # Solving procedure
    # p, jac, x = _solve_with_data(grid, funtol=1e-5, itermax=100, t=0, data_tuple=data, need_x=True)
    print(f'Total Time = {time.time() - st}')
    # matplotlib.rc("font", family='HarmonyOS Sans SC')
    # grp = nx.Graph()
    # grp.add_nodes_from([0, 1, 2, 3])
    # grp.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    # edge_index = list(grp.edges)
    # edge_index.extend([(v, u) for (u, v) in grp.edges])
    # edge_index = [(int(u), int(v)) for (u, v) in grp.edges]
    # edge_index.extend([(int(v), int(u)) for (u, v) in grp.edges])
    # calc_graph = nx.DiGraph()
    # calc_graph.add_nodes_from(edge_index)
    # src = [0]
    # pos = {}
    # src_count = 0
    # other_count = 0
    # src_node = []
    # oth_node = []
    # bus_node = []
    # bias = 0
    # calc_graph.add_nodes_from([100, 101, 102, 103])
    # for ind, i in enumerate([100, 101, 102, 103]):
    #     pos[i] = (-2, - 3 * ind + 2)
    # for ind, i in enumerate(edge_index):
    #     u, v = i
    #     calc_graph.add_node(i)
    #     if ind <= 3:
    #         calc_graph.add_edge(ind + 100, i)
    #     else:
    #         calc_graph.add_edge(ind + 96, i)
    #     if u in src or v in src:
    #         pos[i] = (0, src_count)
    #         src_count -= 2
    #         src_node.append(i)
    #     else:
    #         neighbor = [(v, k) for k in grp.adj[v] if k != u]
    #         for line in neighbor:
    #             calc_graph.add_edge(line, i)
    #         pos[i] = (1 + random.random(), other_count)
    #         other_count -= 2
    #         oth_node.append(i)
    # node_count = 0
    # calc_graph.add_nodes_from(grp.nodes)
    # for i in grp.nodes:
    #     bus_node.append(i)
    #     if i not in src:
    #         for j in grp.adj[i]:
    #             calc_graph.add_edge((i, j), i)
    #     pos[i] = (4, node_count)
    #     node_count -= 2
    # nx.draw_networkx_nodes(calc_graph, nodelist=[100, 101, 102, 103], pos=pos, node_color='black', label='数据节点')
    # nx.draw_networkx_nodes(calc_graph, nodelist=src_node, pos=pos, node_color='purple', label='计算基础节点（与数据相关的常量）')
    # nx.draw_networkx_nodes(calc_graph, nodelist=oth_node, pos=pos, node_color='blue', label='中间层（边条件概率）')
    # nx.draw_networkx_nodes(calc_graph, nodelist=bus_node, pos=pos, node_color='red', label='节点层（边缘概率）')
    # nx.draw_networkx_edges(calc_graph, pos=pos)
    # plt.legend(fontsize=18)
    # print(nx.is_directed_acyclic_graph(calc_graph))
    # plt.plot(range(1, 100), results)
    # plt.show()
    # popt, pcov = curve_fit(linear_expr, range(1, 100), results)
    # plt.plot(range(1, 100), linear_expr(range(1, 100), *popt))
    # plt.legend(fontsize=16)
    # plt.show()

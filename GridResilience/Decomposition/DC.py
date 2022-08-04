"""
Implement Deletion-Contraction Algorithm
"""
import json

import networkx as nx
import jax.numpy as np
from GridResilience.SparseSolver import *
from GridResilience.Environment import *
import pandas as pd
from copy import deepcopy, copy
import time
import _pickle as pickle
import ray


def deletion_contraction(c: GridCase, ray_calculate=False):
    if not nx.is_connected(c.graph):
        raise NotImplementedError()
    case_name = c.case_info['station_name']
    if len(c.source_idx[0]) > 1:
        c.contract_source_nodes(copy=False)
    tree_stack = []  # 树列表
    tsk = []

    def _loads_subgraph(case: GridCase):
        return case.graph.subgraph(case.bus[case.load_list, BUS_I].tolist())

    # 计算场景概率的边列表初始化
    prob_template = ['on', 'off', 'semi']
    prob_init = {key: [] for key in prob_template}

    if nx.is_tree(_loads_subgraph(c)):
        print(f'------原始算例为辐射状------')
        tree_stack.append((c, prob_init))
    else:
        tree_counter = 0
        print(f'------原始算例存在环，开始分解过程------')
        start = time.time()
        cyclic_stack = [(c, prob_init)]  # 有环网络列表
        while cyclic_stack:
            cyclic = cyclic_stack.pop()
            prob = cyclic[1]
            grid_case = cyclic[0]
            prob_js = json.dumps(prob)
            gc_local = _loads_subgraph(grid_case)
            T = nx.minimum_spanning_tree(gc_local, algorithm='prim')
            T_edge = list(T.edges)
            link_edge = [edge for edge in gc_local.edges if edge not in T_edge]
            gc_pickle = pickle.dumps(grid_case)
            for num in range(2 ** len(link_edge)):
                this_prob = json.loads(prob_js)
                bin_list = [int(i) for i in bin(num)[2:].zfill(len(link_edge))]  # 通过二进制编码开断场景
                edges_to_append, edges_to_delete = [], []
                for j in range(len(link_edge)):
                    if bin_list[j] == 1:
                        edges_to_append.append(link_edge[j])
                    else:
                        edges_to_delete.append(link_edge[j])
                new_grid = pickle.loads(gc_pickle)
                for e in edges_to_delete:
                    u_id, v_id = e
                    index_1 = np.where((new_grid.branch[:, FBUS] == u_id) & (new_grid.branch[:, TBUS] == v_id))[0]
                    index_2 = np.where((new_grid.branch[:, FBUS] == v_id) & (new_grid.branch[:, TBUS] == u_id))[0]
                    total_index = np.concatenate((index_1, index_2))
                    idx = new_grid.branch[total_index, BRH_I].tolist()
                    this_prob['off'].extend(idx)
                for e in edges_to_append:
                    u_id, v_id = e
                    bh = new_grid.branch.copy()
                    bh[bh[:, FBUS] == v_id, FBUS] = u_id
                    bh[bh[:, TBUS] == v_id, TBUS] = u_id
                    index_list = np.arange(0, np.size(bh, axis=0))
                    index = index_list[bh[:, FBUS] == bh[:, TBUS]]
                    idx = new_grid.branch[index, BRH_I].tolist()
                    if len(idx) == 1:
                        this_prob['on'].extend(idx)
                    else:
                        this_prob['semi'].append(idx)

                for e in edges_to_delete:
                    this_u = new_grid.contracted_mapping.get(e[0], e[0])
                    this_v = new_grid.contracted_mapping.get(e[1], e[1])
                    new_grid.delete_edge(this_u, this_v, copy=False, need_edge_id=False, static=True)
                for e in edges_to_append:
                    this_u = new_grid.contracted_mapping.get(e[0], e[0])
                    this_v = new_grid.contracted_mapping.get(e[1], e[1])
                    new_grid.contract_edge(this_u, this_v, copy=False, need_edge_id=False, static=True)

                new_grid.update()  # Lazy update
                _lg = _loads_subgraph(new_grid)
                if _lg.size() + 1 == _lg.order():
                    new_grid.case_info['station_name'] = case_name + f'Scenario{tree_counter}'
                    tree_counter += 1
                    tree_stack.append((new_grid, this_prob))
                    if ray_calculate:
                        tsk.append(_local_prob_solver.remote(new_grid))
                else:
                    cyclic_stack.append((new_grid, this_prob))
        print(f'分解完成，分解生成{len(tree_stack)}个子网络，用时{time.time() - start}')
    if ray_calculate:
        prob = ray.remote(calculate_tree_prob).remote(tree_stack, c)
        return tree_stack, tsk, prob
    else:
        return tree_stack


def calculate_tree_prob(tree_list: list, c: GridCase):
    loc_dict = dict(zip(c.branch[:, BRH_I].tolist(), range(c.branch.shape[0])))
    trb = np.ones((len(tree_list), np.size(c.linesafe, axis=1)))
    c0 = c.linesafe
    c1 = 1 - c0
    for index, t in enumerate(tree_list):
        for n in t[1]['on']:
            trb[index, :] *= c0[loc_dict[n], :]
        for f in t[1]['off']:
            trb[index, :] *= c1[loc_dict[f], :]
        for s in t[1]['semi']:
            trb[index, :] *= 1 - np.prod(c1[[loc_dict[para] for para in s], :], axis=0)
    return trb


def deletion_contraction_sensitivity(c: GridCase, node_wise=True, deletion_contraction_results=None):
    if deletion_contraction_results is None:
        tree, task, prob = deletion_contraction(c, ray_calculate=True)
    else:
        tree, task, prob = deletion_contraction_results
    ps = ray.get(task)
    ts = ray.get(prob)
    sen_list = ray.get([_local_sense_solver.remote(tr[0]) for tr in tree])
    df_list = merge_multi_scenario_sensitivity(sen_list, c, tree, ps, ts)
    final_res = df_list[0].to_numpy().copy()
    for df in df_list[1:]:
        final_res += df.to_numpy().copy()
    if node_wise and not deletion_contraction_results:
        return tree, ps, ts, final_res
    elif node_wise and deletion_contraction_results:
        return final_res
    elif not node_wise and not deletion_contraction_results:
        return tree, ps, ts, np.sum(final_res, axis=0)
    else:
        return np.sum(final_res, axis=0)


@ray.remote
def _local_prob_solver(c):
    p, _ = prob_solver(c)
    return p


@ray.remote
def _local_sense_solver(c):
    return sensitivity(c, need_id=True)


if __name__ == "__main__":
    from GridResilience.GridCase import *
    from line_profiler import LineProfiler
    import ray


    def traffic_case12():
        branch = np.array([10001, 3001, 3002, 311, 0,
                           10002, 3001, 3003, 311, 0,
                           10003, 3001, 3004, 311, 0,
                           10004, 3002, 3005, 311, 0,
                           10005, 3002, 3006, 311, 0,
                           10006, 3003, 3004, 311, 0,
                           10007, 3003, 3007, 311, 0,
                           10008, 3004, 3005, 311, 0,
                           10009, 3004, 3008, 311, 0,
                           10010, 3005, 3006, 311, 0,
                           10011, 3005, 3009, 311, 0,
                           10012, 3006, 3010, 311, 0,
                           10013, 3007, 3008, 311, 0,
                           10014, 3007, 3011, 311, 0,
                           10015, 3008, 3009, 311, 0,
                           10016, 3008, 3011, 311, 0,
                           10017, 3009, 3010, 311, 0,
                           10018, 3009, 3012, 311, 0,
                           10019, 3010, 3012, 311, 0,
                           10020, 3011, 3012, 311, 0]).reshape(-1, 5)
        bus = np.array([3001, 2, 6, 110, 1, 0, 1, 0,
                        3002, 6, 6, 333, 1, 0, 0, 0,
                        3003, 0, 4, 333, 1, 0, 0, 0,
                        3004, 2, 4, 333, 1, 0, 0, 0,
                        3005, 6, 4, 333, 1, 0, 0, 0,
                        3006, 8, 4, 333, 1, 0, 0, 0,
                        3007, 0, 0, 333, 1, 0, 0, 0,
                        3008, 2, 0, 333, 1, 0, 0, 0,
                        3009, 6, 0, 333, 1, 0, 0, 0,
                        3010, 8, 0, 333, 1, 0, 0, 0,
                        3011, 2, -2, 333, 1, 0, 0, 0,
                        3012, 6, -2, 333, 1, 0, 0, 0]).reshape(-1, 8)
        linesafe = np.array([0.9] * len(branch)).reshape(-1, 1)
        return GridCase().from_array(branch, bus, linesafe, station_name='case4_diamond')


    ray.init()
    grid = case_32_modified()  # type: GridCase
    st = time.time()
    tree_stack, tsk, prob = deletion_contraction(grid, ray_calculate=True)
    prb = ray.get(prob)
    res = ray.get(tsk)
    print(f'分解用时{time.time() - st}')
    off_lines = tree_stack[26][1]['off']
    on_lines = tree_stack[26][1]['on']
    ls = pd.DataFrame(grid.linesafe, index=grid.branch[:, 0])
    ls.loc[on_lines] = 1.0
    ls.loc[off_lines] = 0.0
    ls.loc[tree_stack[26][1]['semi'][0][0]] = 0.0
    grid.linesafe = ls.to_numpy()
    new_res_0 = prob_solver(grid)[0].to_numpy()

    ls.loc[tree_stack[26][1]['semi'][0][0]] = 1.0
    grid.linesafe = ls.to_numpy()
    new_res_1 = prob_solver(grid)[0].to_numpy()
    new_res = new_res_0 * 0.1 + new_res_1 * 0.9
    old_res = res[26].sort_index().to_numpy()

    # lp = LineProfiler()
    # lp.add_function(deletion_contraction)
    # lp.add_function(calculate_tree_prob)
    # lp.add_function(grid.delete_edge)
    # lp.add_function(merge_multi_scenario_sensitivity)
    # lp_wrp = lp(deletion_contraction_sensitivity)
    # tree, ps, ts, fr = lp_wrp(grid, node_wise=True)
    # lp_wrp = lp(deletion_contraction)
    # lp_wrp(grid)
    # lp.print_stats()

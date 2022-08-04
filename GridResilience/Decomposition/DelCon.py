import pickle
from copy import deepcopy
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import time
from tqdm import tqdm
import os
from GridResilience.Environment import *
import ray
from GridResilience.SparseSolver import *
from line_profiler import LineProfiler
import json
from GridResilience.utils import NumpyEncoder
import matplotlib.pyplot as plt
import pydot


def cycle_graph_decomposition(case: nx.Graph, case_edge_dict: dict, base):
    """
    :param case: 网络图(nx.Graph对象)，对于一个CPSCase对象，用case.graph即可
    :param case_edge_dict: 图的边字典（按照从小到大排好序的tuple(u, v) -> 边在linesafe中的序号(int)）
    :param base: CPSCase对象，基图
    :return: TreeStack: 树的堆栈，包含图对象[0]、图对应场景概率[1]、集群字典[2]、并架线字典[3]和新的边索引字典[4](有序tuple作为key)
    """

    def keys_to_int(x):
        return {int(k): v for k, v in x.items()}

    case_frozen = deepcopy(base.graph)
    node_frozen_dict = {}
    for k in case_frozen.nodes:
        node_frozen_dict[int(k)] = int(k)
    sources = base.source_idx[0]
    source_id = sources[0]
    source_edges = []
    edge_dict = deepcopy(case_edge_dict)
    for src in sources:
        edge_temp = list(base.graph.edges(src))
        for e in edge_temp:
            number = case_edge_dict[edge_to_key(e)]
            if e[0] == src:
                t = (source_id, e[1])
            else:
                t = (e[0], source_id)
            t = edge_to_key(t)
            source_edges.append(t)
            edge_dict[t] = number
    case = nx.Graph(case)
    case.remove_nodes_from(sources)
    count = 0  # 分解次数计数
    total_counter = 0
    TreeStack = []
    prob_init = {'off': [], 'on': [], 'semi': []}
    node_list = list(case.nodes)
    node_init_dict = {}
    for k in node_list:
        node_init_dict[int(k)] = int(k)
    if nx.is_tree(case):
        print('原始算例为辐射状')
        TreeStack.append([case_frozen, prob_init, node_frozen_dict, {}, deepcopy(case_edge_dict)])
    else:
        decomposition_tree = nx.DiGraph()
        tree_counter = 0
        decomposition_tree.add_node(tree_counter)
        print('原始算例有环，开始分解过程......')
        CyclicGraphStack = [(case, prob_init, node_init_dict, {}, deepcopy(case_edge_dict), tree_counter)]
        start_time = time.time()
        cyclic_count = 0
        while CyclicGraphStack:  # 有环图堆栈循环直到清空堆栈
            cyclic_graph_list = CyclicGraphStack.pop()  # 顶层的图出栈
            cyclic_graph_prob = cyclic_graph_list[1]
            cyclic_case = cyclic_graph_list[0]
            merge_dict_tree = deepcopy(cyclic_graph_list[2])
            linkage_dict_total = deepcopy(cyclic_graph_list[3])
            this_id = cyclic_graph_list[5]
            cluster_dict = {}
            for i in merge_dict_tree:
                if not cluster_dict.__contains__(merge_dict_tree[i]) and i == merge_dict_tree[i]:
                    cluster_dict[int(merge_dict_tree[i])] = [i]
                elif not cluster_dict.__contains__(merge_dict_tree[i]):
                    cluster_dict[int(merge_dict_tree[i])] = [merge_dict_tree[i], i]
                else:
                    cluster_dict[int(merge_dict_tree[i])].append(i)
            del cyclic_graph_list
            T = nx.Graph(nx.minimum_spanning_tree(cyclic_case, algorithm='prim'))  # 找到一个最小生成树
            tree_data = json_graph.node_link_data(T)
            edge_class = {'tree': [], 'link': []}
            cyclic_edge_set = set(cyclic_case.edges)
            tree_edge_set = set(T.edges)
            for i in tree_edge_set:  # 边分类
                edge_class['tree'].append((edge_to_key(i), edge_dict[edge_to_key(i)]))
            for i in cyclic_edge_set - tree_edge_set:
                edge_class['link'].append((edge_to_key(i), edge_dict[edge_to_key(i)]))

            # serialize dict data
            merge_dict_tree_json = json.dumps(merge_dict_tree, cls=NumpyEncoder)
            cluster_dict_json = json.dumps(cluster_dict, cls=NumpyEncoder)
            linkage_dict_total_json = json.dumps(linkage_dict_total, cls=NumpyEncoder)
            for num in range(2 ** len(edge_class['link'])):
                tree_counter += 1
                decomposition_tree.add_edge(this_id, tree_counter)
                cluster = json.loads(cluster_dict_json, object_hook=keys_to_int)
                merge_dict = json.loads(merge_dict_tree_json, object_hook=keys_to_int)
                linkage_dict = json.loads(linkage_dict_total_json, object_hook=keys_to_int)
                count += 1
                subgraph = json_graph.node_link_graph(tree_data)
                bin_list = [int(i) for i in bin(num)[2:].zfill(len(edge_class['link']))]  # 通过二进制编码开断场景
                # 计算场景概率
                prob = deepcopy(cyclic_graph_prob)
                for k in range(len(edge_class['link'])):
                    k_key = edge_class['link'][k][0]
                    k_parallel = [edge_class['link'][k][1]]
                    for cc in linkage_dict:
                        for line in linkage_dict[cc]:
                            left = merge_dict[linkage_dict[cc][line][0]]
                            right = merge_dict[linkage_dict[cc][line][1]]
                            if edge_to_key((left, right)) == k_key:
                                k_parallel.append(edge_dict[edge_to_key((left, right))])
                    if bin_list[k] == 0:
                        prob['off'].extend(k_parallel)
                    else:
                        if len(k_parallel) == 1:
                            prob['on'].extend(k_parallel)
                        else:
                            prob['semi'].append(k_parallel)

                edges_to_append = [edge_class['link'][j][0] for j in range(len(edge_class['link'])) if
                                   bin_list[j] == 1]  # 找到需要添加的边索引

                if len(edges_to_append) != 0:
                    subgraph.add_edges_from(edges_to_append)
                    for e in edges_to_append:  # e是按大小顺序排好的tuple
                        if merge_dict[e[0]] < merge_dict[e[1]]:
                            u, v = merge_dict[e[0]], merge_dict[e[1]]
                        else:
                            u, v = merge_dict[e[1]], merge_dict[e[0]]
                        u_neighbor, v_neighbor = list(subgraph.neighbors(u)), list(subgraph.neighbors(v))
                        """
                            1. 节点之间的公共邻接节点合并
                            u - o - v 合并 (u, v)， 导致 u(v) = o， 需要记录该情况
                        """
                        common_neighbor = list(nx.common_neighbors(subgraph, u, v))
                        if not linkage_dict.__contains__(u) and common_neighbor:
                            linkage_dict[int(u)] = {}
                        for neighbor in common_neighbor:  # Maybe neighbor is also a cluster
                            idx_v = edge_to_key((neighbor, v))
                            for v_node in cluster[v]:
                                for nb in cluster[neighbor]:
                                    if case.has_edge(v_node, nb) and v_node != v:
                                        linkage_dict[int(u)][edge_dict[edge_to_key((nb, v_node))]] = edge_to_key(
                                            (v, neighbor))
                                        edge_dict[idx_v] = edge_dict[edge_to_key((nb, v_node))]
                                    elif case.has_edge(v_node, nb):
                                        linkage_dict[int(u)][edge_dict[edge_to_key((nb, v_node))]] = edge_to_key(
                                            (v, neighbor))
                        if linkage_dict.__contains__(v):
                            links_to_pop = []
                            for v_link in linkage_dict[v]:
                                if linkage_dict[v][v_link][0] == u or linkage_dict[v][v_link][1] == u:
                                    links_to_pop.append(v_link)
                            for pop in links_to_pop:
                                del linkage_dict[v][pop]
                            if not linkage_dict.__contains__(u):
                                linkage_dict[u] = linkage_dict[v]
                            else:
                                linkage_dict[u].update(linkage_dict[v])
                            del linkage_dict[v]
                        """
                            2. 接入被归并节点的线路
                            u - v - o -> u(v) - o 生成新边 u - o
                        """
                        for neighbor in v_neighbor:
                            idx_u = edge_to_key((neighbor, u))
                            idx_v = edge_to_key((neighbor, v))
                            if neighbor not in common_neighbor and neighbor != u:
                                edge_dict[idx_u] = edge_dict[idx_v]
                        """
                            3. 节点合并(序号合并)
                            u - v -> u, cluster[u].append(v)
                        """
                        nx.contracted_nodes(subgraph, u, v, self_loops=False, copy=False)
                        if cluster.__contains__(v) and cluster.__contains__(u):
                            cluster[u] += cluster[v]
                            del cluster[v]
                        elif cluster.__contains__(v) and not cluster.__contains__(u):
                            cluster[u] = cluster[v]
                            del cluster[v]
                        for c in cluster:
                            for node in cluster[c]:
                                merge_dict[node] = c
                tree_scale = len(subgraph.nodes)
                if len(subgraph.edges) == tree_scale - 1:  # 分解过程不改变图的连通性（全联通图）
                    subgraph.add_node(source_id)
                    links_for_source = {}
                    merge_dict[source_id] = source_id
                    for se in source_edges:
                        if merge_dict.__contains__(se[0]) and merge_dict.__contains__(se[1]):
                            if se[0] == source_id:
                                load_node = merge_dict[se[1]]
                            else:
                                load_node = merge_dict[se[0]]
                            edge_dict[edge_to_key((source_id, load_node))] = edge_dict[edge_to_key(se)]
                            if links_for_source.__contains__(load_node):
                                linkage_dict[load_node][edge_dict[edge_to_key(se)]] = edge_to_key(
                                    (source_id, load_node))
                            else:
                                links_for_source[load_node] = []
                                subgraph.add_edge(source_id, load_node)
                    TreeStack.append(
                        (subgraph, prob, merge_dict, linkage_dict, edge_dict, tree_counter))
                    total_counter += 1
                else:
                    CyclicGraphStack.append(
                        (subgraph, prob, merge_dict, linkage_dict, edge_dict, tree_counter))
            cyclic_count += 1
        print(f'共进行{count}次组合分解，共分解成{len(TreeStack)}个场景，分解用时{time.time() - start_time}')
        # from netgraph import Graph
        # Graph(decomposition_tree, node_layout='dot', node_labels=True, node_label_fontdict=dict(size=14))
        # import matplotlib.pyplot as plt
        # plt.show()
    return TreeStack


def calculate_tree_prob(Tree, linesafe, parallel=False):
    @ray.remote
    def _parallel_prob(tr, ls):
        trp = np.ones(np.size(linesafe, axis=1))
        for on in tr[1]['on']:
            trp *= ls[on, :]
        for off in tr[1]['off']:
            trp *= 1 - ls[off, :]
        for sem in tr[1]['semi']:
            trp *= 1 - np.prod(1 - ls[sem, :])
        return trp

    if parallel:
        ray_result = [_parallel_prob.remote(tree, linesafe) for tree in Tree]
        TreeProb = np.array(ray.get(ray_result))
    else:
        TreeProb = np.ones((len(Tree), np.size(linesafe, axis=1)))
        for index, tree in enumerate(Tree):
            for n in tree[1]['on']:
                TreeProb[index, :] *= linesafe[n, :]
            for f in tree[1]['off']:
                TreeProb[index, :] *= 1 - linesafe[f, :]
            for s in tree[1]['semi']:
                TreeProb[index, :] *= 1 - np.prod(1 - linesafe[s, :])
    return TreeProb


def edge_to_key(edge: tuple):
    return tuple(np.sort(np.array(edge)))


def __single_case__(tree_case, origin, origin_pickle, tree_prob, need_case=False):
    sub_case = Case.generate_from_tree_to_case_cycle(tree_case, origin_pickle)
    prob, jac = prob_solver(sub_case)
    prob_array = np.zeros(len(origin.bus))
    for i in tree_case[2]:
        bus_id = origin.bus[tree_case[2][i], BUS_I]
        bus_index = np.nonzero(sub_case.bus[:, BUS_I] == bus_id)[0][0]
        prob_array[i] = prob[bus_index] * tree_prob
    if need_case:
        return prob_array, sub_case
    else:
        return prob_array


def parallel_case_calc(origin, tree_list, prob, if_parallel=True):
    base_pickle = pickle.dumps(origin)

    @ray.remote
    def _case_remote(tree_case, prb):
        return __single_case__(tree_case, origin, base_pickle, tree_prob=prb)

    if if_parallel:
        print(f'开始计算分解后各场景概率...')
        prob_arrays = ray.get(
            [_case_remote.remote(tree_list[tr], prob[tr]) for tr in tqdm(range(len(tree_list)), desc='并行计算生成场景')])
    else:
        print(f'开始计算分解后各场景概率...')
        prob_arrays = []
        for tree_ind, tree in tqdm(enumerate(tree_list)):
            prob_arrays.append(__single_case__(tree, origin, base_pickle, prob[tree_ind]))
    return np.sum(np.array(prob_arrays), axis=0)


def single_source_multi_section_graph_decomposition(case: Case):
    if len(case.source_idx[0]) > 1:
        case.contract_source_nodes()
    connected_components = case.find_connected_components()
    src_idx = case.source_idx[0][0]
    sub_tree = []
    for cc in connected_components:
        cc.add(src_idx)
        sub_graph = nx.subgraph(case.graph, cc)
        sub_tree.append(
            cycle_graph_decomposition(sub_graph, case.edge_dict, case))
    return sub_tree


def sensitivity_for_decomposition(tree_list, prob_list, origin, t=0):
    base_pickle = pickle.dumps(origin)

    def _single_case_sense(tree, prob, org, org_pickle):
        this_res, sub_case = __single_case__(tree, org, org_pickle, prob, need_case=True)
        sen = np.zeros((np.size(org.bus, axis=0), np.size(org.branch, axis=0)))
        line_total = set(range(np.size(org.branch, axis=0)))
        line_calculated = []
        graph_sense = sensitivity(sub_case)
        sense_array = np.zeros((np.size(org.bus, axis=0), np.size(graph_sense, axis=1)))
        # edge_list = []
        # for kk in range(len(sub_case.branch)):
        #     edge_list.append((int(sub_case.branch[kk, FBUS]), int(sub_case.branch[kk, TBUS])))
        edge_list = list(sub_case.graph.edges)
        org_id_map = {}
        for i in range(len(sub_case.branch)):
            u, v = org.bus_dict[sub_case.branch[i, FBUS]], org.bus_dict[sub_case.branch[i, TBUS]]
            u_id = int(sub_case.branch[i, FBUS])
            v_id = int(sub_case.branch[i, TBUS])
            org_id_map[(u, v)] = (sub_case.bus_dict[u_id], sub_case.bus_dict[v_id])
            org_id_map[(v, u)] = (sub_case.bus_dict[u_id], sub_case.bus_dict[v_id])

        for i in tree[2]:
            bus_id = tree[2][i]
            real_ind = np.nonzero(sub_case.bus[:, BUS_I] == org.bus[bus_id, BUS_I])[0][0]
            sense_array[i, :] = graph_sense[real_ind, :] * prob
        for k in tree[1]:
            line_calculated.extend(tree[1][k])
        for on in tree[1]['on']:
            sen[:, on] = this_res / org.linesafe[on, t]
        for off in tree[1]['off']:
            sen[:, off] = - this_res / (1 - org.linesafe[off, t])
        for para in tree[1]['semi']:
            for j in para:
                sen[:, j] = - this_res / (1 - prod([1 - org.linesafe[kk, t] for kk in para])) * (
                    prod([1 - org.linesafe[jj, t] for jj in para if jj != j]))
        remains = list(line_total - set(line_calculated))
        for line in remains:
            u, v = org.branch[line, FBUS], org.branch[line, TBUS]
            line_id = np.int64(org.branch[line, BRH_I])
            find_id = np.nonzero(np.int64(sub_case.branch[:, BRH_I]) == line_id)[0][0]
            ind = (tree[2][org.bus_dict[u]], tree[2][org.bus_dict[v]])
            new_ind = org_id_map[ind]
            if new_ind in edge_list:
                sen[:, line] = sense_array[:, find_id]
            elif (new_ind[1], new_ind[0]) in edge_list:
                sen[:, line] = sense_array[:, find_id]
        return sen

    res = np.zeros((np.size(case.bus, axis=0), np.size(case.branch, axis=0)))
    for ind, tr in enumerate(tree_list):
        aa = _single_case_sense(tr, prob_list[ind, t], origin, base_pickle)
        res += aa
    return res


if __name__ == "__main__":
    filepath = '../../FinalCodes/data/'
    station_name = '4'
    case = Case().from_file(filepath=filepath, station_name=station_name)
    bus = case.bus
    bus[:, BUS_I] -= 1
    branch = case.branch
    branch[:, FBUS] -= 1
    branch[:, TBUS] -= 1
    case = Case().from_array(branch, bus, case.linesafe)
    Tree = []
    start = time.time()
    if len(case.bus) > 10:
        ray.init()
    # res = res[0]
    t = time.time()
    sub = single_source_multi_section_graph_decomposition(case)
    for i in sub:
        Tree.extend(i)
    st = time.time()
    Tree_prob = calculate_tree_prob(Tree, case.linesafe)
    # sense_res = sensitivity_for_decomposition(Tree, Tree_prob, case)
    prob_res = parallel_case_calc(case, Tree, Tree_prob, if_parallel=True if len(Tree) > 100 else False)
    # case_pickle = pickle.dumps(case)
    #
    # sub_case_a = Case.generate_from_tree_to_case_cycle(Tree[0], case_pickle)
    # prob_a = sensitivity(sub_case_a, node_wise=True)
    # sub_case_b = Case.generate_from_tree_to_case_cycle(Tree[1], case_pickle)
    # prob_b = sensitivity(sub_case_b, node_wise=True)
    # lp = LineProfiler()
    # lp.add_function(__single_case__)
    # lp.add_function(Case.generate_from_tree_to_case_cycle)
    # lp_wrapper = lp(parallel_case_calc)
    # res = lp_wrapper(case, Tree)
    # lp.print_stats()
    print(f'并行计算时间为{time.time() - st}')
    print(f'总计算时间为{time.time() - t}')

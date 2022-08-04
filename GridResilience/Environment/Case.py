"""
Setting up modeling environments
"""
import logging
import pickle
from copy import deepcopy

import networkx as nx
import numpy as np

from line_profiler import LineProfiler
import pandas as pd
from .idx_cpsbrh import *
from .idx_cpsbus import *


class Case:

    def __init__(self):
        self.linesafe_dict = None
        self.station_name = ""
        self.bus = None
        self.branch = None
        self.busfail = None
        self.linesafe = None
        self.source_idx = None
        self.pos = None
        self.bus_dict = None
        self.edge_dict = None
        self.edge_array = None
        self.graph = None
        self.load_list = None
        self.multiple_edge = {}
        self.generated_edge_map = {}

    def from_file(self, filepath="./data/", station_name=""):
        """

        :param filepath: 文件路径
        :param station_name: 站点名称
        """
        __SHEET_NAME__ = ["line", "node", "device_proba"]

        self.linesafe_dict = None
        self.station_name = station_name
        # 文件读取部分
        for sheet in __SHEET_NAME__:
            file_name = filepath + station_name + "_" + sheet + ".csv"
            df = pd.read_csv(file_name, header=None, index_col=None)
            if sheet == "node":
                try:
                    self.bus = df.iloc[1:, :PLOAD + 1].to_numpy(dtype='float64')
                    del df
                except IOError:
                    logging.error("不存在文件", file_name)
            elif sheet == "line":
                try:
                    self.branch = df.iloc[1:, :].to_numpy(
                        dtype='float64')
                    del df
                except IOError:
                    logging.error("不存在文件", file_name)
            elif sheet == "device_proba":
                try:
                    cum_array = np.array(df.iloc[:, 4]).T.reshape(-1, 1)
                    dev_id = np.int64(np.array(df.iloc[:, 0:2]).reshape(-1, 2))
                    num_dev = np.size(np.nonzero(dev_id[:, 0] == 1))
                    cum_shaped_array = cum_array.reshape(-1, num_dev).T
                    self.linesafe = np.ones(
                        (np.size(self.branch, axis=0), np.max(dev_id[:, 0])))
                    self.busfail = np.zeros(
                        (np.size(self.bus, axis=0), np.max(dev_id[:, 0])))
                    for i in range(np.size(self.branch, axis=0)):
                        idx = np.where((dev_id[:, 0] == 1) & (
                                dev_id[:, 1] == self.branch[i, BRH_I]))[0]
                        self.linesafe[i, :] = 1 - cum_shaped_array[idx, :]
                    for j in range(np.size(self.bus, axis=0)):
                        idx = np.where((dev_id[:, 0] == 1) & (
                                dev_id[:, 1] == self.bus[j, BUS_I]))[0]
                        if np.size(idx, axis=0) != 0:
                            self.busfail[j, :] = cum_shaped_array[idx, :]
                    self.linesafe = np.array(self.linesafe).astype('float64')
                except IOError:
                    logging.error("不存在文件", file_name)
        self.__gen_other_attr__()
        return self

    def from_array(self, line, node, device_proba):
        self.branch = line
        self.bus = node
        self.linesafe = device_proba
        self.__gen_other_attr__()
        return self

    def from_pd(self, line: pd.DataFrame, node: pd.DataFrame, device_proba: pd.DataFrame):
        self.branch = line.to_numpy()
        self.bus = node.to_numpy()
        self.linesafe = device_proba.to_numpy()
        self.__gen_other_attr__()
        return self

    def from_nx_graph(self, base):
        ...

    def __gen_other_attr__(self, graph=None):
        # 节点类型对应电源转换
        self.load_list = []
        self.source_idx = np.nonzero(self.bus[:, BUS_TYPE] == SOURCE)
        issource = np.int_(np.zeros((np.size(self.bus, axis=0), 1)))
        issource[self.source_idx] = 1
        source = np.int_(np.zeros((np.size(self.bus, axis=0), 1)))
        self.bus = np.concatenate((self.bus, issource, source), axis=1)
        bus_num = np.size(self.bus, axis=0)
        bus_id = {}
        self.pos = {}
        source_list = list(self.source_idx[0])
        for i in range(bus_num):
            bus_id[self.bus[i, BUS_I]] = i
            self.pos[i] = (self.bus[i, LOCX], self.bus[i, LOCY])
            if i not in source_list:
                self.load_list.append(i)
        self.bus_dict = bus_id
        bus_array = np.int_(np.array([i for i in range(bus_num)]))
        edge_array = []
        self.edge_dict = {}
        for i in range(np.size(self.branch, axis=0)):
            edge_array.append((bus_id[self.branch[i, FBUS]], bus_id[self.branch[i, TBUS]]))
            self.edge_dict[Case.edge_to_key([bus_id[self.branch[i, FBUS]], bus_id[self.branch[i, TBUS]]])] = i
        if graph is None:
            G = nx.Graph()
            G.add_nodes_from(bus_array)
            G.add_edges_from(edge_array)
            self.graph = G
        else:
            self.graph = graph
        for i, (u, v) in enumerate(edge_array):
            if self.multiple_edge.__contains__(Case.edge_to_key((u, v))):
                self.multiple_edge[Case.edge_to_key((u, v))].append(i)
            else:
                self.multiple_edge[Case.edge_to_key((u, v))] = [i]
        self.edge_array = np.array(edge_array)

        # temp = np.int_(np.zeros((np.size(self.branch, axis=0), 1)))
        # self.branch = np.concatenate((self.branch, temp, temp), axis=1)

    def is_tree(self):
        """
        判断当前算例是否为树
        :return: Bool 是否为树
        """
        return nx.is_tree(self.graph)

    def export_line_dataframe(self):
        """
        :return: branch_df: 边DataFrame
        """
        branch_df = pd.DataFrame(self.branch[:, :3], columns=['id', 'from_id', 'to_id'], dtype=np.int64)
        from_num = []
        to_num = []
        for row in range(np.size(self.branch, axis=0)):
            from_num.append(self.bus_dict[branch_df.iloc[row]['from_id']])
            to_num.append(self.bus_dict[branch_df.iloc[row]['to_id']])
        branch_df.insert(loc=len(branch_df.columns), column='index',
                         value=range(np.size(self.branch, axis=0)))
        branch_df.insert(loc=len(branch_df.columns), column='from', value=from_num)
        branch_df.insert(loc=len(branch_df.columns), column='to', value=to_num)
        return branch_df

    def contract_source_nodes(self):
        if len(self.source_idx[0]) > 1:
            src_node = self.source_idx[0][0]
            bh = pd.DataFrame(self.branch)
            bs = pd.DataFrame(self.bus)
            ls = pd.DataFrame(self.linesafe)
            other_src = list(self.source_idx[0][1:])
            other_src_id = list(bs[0].astype(np.int64).iloc[other_src].values)

            def _replace(x):
                if x in other_src_id:
                    return src_node
                else:
                    return x

            bh[1] = bh[1].map(_replace)
            bh[2] = bh[2].map(_replace)
            bs.drop(other_src, inplace=True)
            self.from_pd(bh, bs, ls)

    def to_tree(self):
        G = self.graph
        G_tree = nx.minimum_spanning_tree(G)
        source_length = nx.single_source_shortest_path_length(G_tree, source=self.source_idx[0][0])
        self.tree_edges = list(nx.minimum_spanning_edges(G, data=False))
        self.bridges = list(nx.bridges(G))
        bi_dir_line = {}
        for i in range(np.size(self.edge_array, axis=0)):
            bi_dir_line[str(np.sort(self.edge_array[i, :]))] = i
        bridge_index = []
        tree_edge_index = []
        for edge in self.tree_edges:
            tree_edge_index.append(bi_dir_line[str(np.int_(np.sort(np.array(edge))))])
        for bridge in self.bridges:
            bridge_index.append(bi_dir_line[str(np.int_(np.sort(np.array(bridge))))])
        # bridge_index = np.array(bridge_index)
        self.bridge_index = bridge_index
        self.bi_line_index = [x for x in list(range(len(self.edge_array))) if x not in self.bridge_index]
        # 线路信息侧可靠
        self.branch = self.branch[tree_edge_index, :]
        self.linesafe = self.linesafe[tree_edge_index, :]
        for i in range(len(self.branch)):
            if self.bus_dict[self.branch[i, FBUS]] in source_length.keys() and self.bus_dict[
                self.branch[i, TBUS]] in source_length.keys():
                if source_length[self.bus_dict[self.branch[i, FBUS]]] > source_length[
                    self.bus_dict[self.branch[i, TBUS]]]:
                    self.branch[i, FBUS], self.branch[i, TBUS] = self.branch[i, TBUS], self.branch[i, FBUS]
        edge_array = np.zeros((np.size(self.branch, axis=0), 2))
        for i in range(np.size(self.branch, axis=0)):
            edge_array[i, 0] = self.bus_dict[self.branch[i, FBUS]]
            edge_array[i, 1] = self.bus_dict[self.branch[i, TBUS]]
        self.edge_array = edge_array.astype(int)
        iso_list = list(nx.isolates(G_tree))
        G_tree.remove_nodes_from(iso_list)
        self.graph = G_tree

    @staticmethod
    def generate_from_tree_to_case_cycle(tree, bs):
        """
        :param tree: TreeStack中的一个tuple，包含了树状网络的拓扑图(nx.Graph对象)，场景概率等...详见TreeStack的生成方式
        :param base: 基础算例，直接从文件中读取的结果
        :return: case: 生成的适用于solver的算例格式（调用runr_jac_sparse.py中的wrapper_without_pool计算）
        """
        tree_graph = tree[0]
        tree_merge_dict = tree[2]
        tree_linkage_dict = tree[3]
        tree_edge_dict = tree[4]
        case = pickle.loads(bs)
        base = pickle.loads(bs)
        case.branch = np.zeros((len(tree_graph.edges), np.size(base.branch, axis=1)))
        case.linesafe = np.zeros((len(tree_graph.edges), np.size(base.linesafe, axis=1)))
        edge_count = 0
        for edge in tree_graph.edges:
            case.branch[edge_count, :] = base.branch[tree_edge_dict[Case.edge_to_key(edge)], :]
            case.branch[edge_count, FBUS] = base.bus[edge[0], BUS_I]
            case.branch[edge_count, TBUS] = base.bus[edge[1], BUS_I]
            case.generated_edge_map[(base.bus[edge[0], BUS_I], base.bus[edge[1], BUS_I])] = edge_count
            case.linesafe[edge_count, :] = base.linesafe[tree_edge_dict[Case.edge_to_key(edge)], :]
            edge_count += 1
        for cluster in tree_linkage_dict.keys():
            for line in tree_linkage_dict[cluster].keys():
                left = tree_merge_dict[tree_linkage_dict[cluster][line][0]]
                right = tree_merge_dict[tree_linkage_dict[cluster][line][1]]
                if Case.edge_to_key((left, right)) in tree_graph.edges:
                    new_linesafe = base.linesafe[line, :].reshape(1, -1)
                    new_branch = base.branch[line, :].reshape(1, -1)
                    new_branch[0, FBUS] = base.bus[left, BUS_I]
                    new_branch[0, TBUS] = base.bus[right, BUS_I]
                    case.branch = np.concatenate((case.branch, new_branch), axis=0)
                    case.linesafe = np.concatenate((case.linesafe, new_linesafe), axis=0)
        node_count = 0
        case.bus = np.zeros((len(tree_graph.nodes), np.size(base.bus, axis=1)))
        source_indices = base.source_idx[0]
        source_cluster = {}
        for src in list(source_indices):
            source_cluster[tree_merge_dict[src]] = src
        for node in tree_graph.nodes:
            case.bus[node_count, :] = base.bus[node, :]
            if node in source_cluster.keys():
                case.bus[node_count, BUS_TYPE] = 110
                case.bus[node_count, ISSOURCE] = 1
            node_count += 1
        edge_array = []
        case.edge_dict = {}
        bus_id = {}
        for i in range(len(case.bus)):
            bus_id[case.bus[i, BUS_I]] = i
        for i in range(np.size(case.branch, axis=0)):
            edge_array.append((bus_id[case.branch[i, FBUS]], bus_id[case.branch[i, TBUS]]))
            case.edge_dict[Case.edge_to_key([bus_id[case.branch[i, FBUS]], bus_id[case.branch[i, TBUS]]])] = i
        for src in source_indices:
            source_cluster[tree_merge_dict[src]] = src
        case.edge_array = edge_array
        case.graph = nx.Graph()
        case.bus_dict = bus_id
        case.graph.add_nodes_from(np.int_(np.array([i for i in range(len(case.bus))])))
        case.graph.add_edges_from(case.edge_array)
        case.source_idx = np.nonzero(case.bus[:, BUS_TYPE] == SOURCE)
        case.multiple_edge = {}
        for i, (u, v) in enumerate(edge_array):
            if case.multiple_edge.__contains__(Case.edge_to_key((u, v))):
                case.multiple_edge[Case.edge_to_key((u, v))].append(i)
            else:
                case.multiple_edge[Case.edge_to_key((u, v))] = [i]
        return case

    @staticmethod
    def edge_to_str(edge: tuple):
        return str(np.sort(np.array(edge)))

    @staticmethod
    def edge_to_key(edge):
        return tuple(np.sort(np.array(edge)))

    def find_connected_components(self):
        graph = deepcopy(self.graph)
        src = self.source_idx[0]
        graph.remove_nodes_from(src)
        connected_components = list(nx.connected_components(graph))
        return connected_components

    def linesafe_to_dict(self):
        if self.graph != None:
            self.linesafe_dict = []
            for i in range(np.size(self.linesafe, axis=1)):
                self.linesafe_dict.append({})
                self.linesafe_dict[-1] = dict.fromkeys(list(self.graph.edges))
                for e in list(self.graph.edges):
                    self.linesafe_dict[-1][e] = self.linesafe[self.edge_dict[Case.edge_to_key(e)], i]

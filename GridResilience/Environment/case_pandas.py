"""
Setting up modeling environments
"""
import logging
import json
import time
import networkx as nx
import pickle
# import numpy as np
import numpy as np
from line_profiler import LineProfiler
from copy import copy, deepcopy
from networkx.readwrite import json_graph
import pandas as pd
from GridResilience.Environment.idx_cpsbrh import *
from GridResilience.Environment.idx_cpsbus import *


def edge_to_key(edge):
    return tuple(sorted(edge))


def edge_to_key_np(u, v):
    return tuple(np.sort([u, v]))


class GridCase:
    def __init__(self):
        _attribute = ['station_name', 'bus_num', 'branch_num', 'source_num', 'connected']
        self.case_info = dict.fromkeys(_attribute)
        self.baseMVA = 0.0
        self.bus = None
        self.branch = None
        self.linesafe = None
        self.source_idx = None
        self.pos = None
        self.bus_dict = None
        self.edge_array = None
        self.graph = None
        self.load_list = None
        self.scenarios = {}
        self.contracted_mapping = {}
        self.extend_attr = {}

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f"System {self.case_info['station_name']} has {np.size(self.bus, axis=0)} buses and {np.size(self.branch, axis=0)} branches"

    def from_csv(self, filepath="./data/", station_name="", delimiter='_'):
        """
        从csv文件读取算例文件
        :param filepath: 文件路径
        :param station_name: 站点名称
        :param delimiter:
        """
        __SHEET_NAME__ = ["line", "node", "device_proba"]
        self.case_info['station_name'] = station_name
        # 文件读取部分
        for sheet in __SHEET_NAME__:
            file_name = filepath + station_name + delimiter + sheet + ".csv"
            df = pd.read_csv(file_name, header=None, index_col=None)
            if sheet == "node":
                try:
                    self.bus = df.iloc[1:, :PLOAD + 1].to_numpy(dtype='float64')
                except IOError:
                    logging.error("不存在文件", file_name)
            elif sheet == "line":
                try:
                    self.branch = df.iloc[1:, :].to_numpy(dtype='float64')
                except IOError:
                    logging.error("不存在文件", file_name)
            elif sheet == "device_proba":
                try:
                    cum_array = np.array(df.iloc[:, 4]).T.reshape(-1, 1)
                    dev_id = np.int64(np.array(df.iloc[:, 0:2]).reshape(-1, 2))
                    num_dev = np.size(np.nonzero(dev_id[:, 0] == 1))
                    cum_shaped_array = cum_array.reshape(-1, num_dev).T
                    self.linesafe = np.ones((np.size(self.branch, axis=0), np.max(dev_id[:, 0])))
                    for i in range(np.size(self.branch, axis=0)):
                        idx = np.where((dev_id[:, 0] == 1) & (dev_id[:, 1] == self.branch[i, BRH_I]))[0]
                        self.linesafe[i, :] = 1 - cum_shaped_array[idx, :]
                    self.linesafe = np.array(self.linesafe).astype('float64')
                except IOError:
                    logging.error("不存在文件", file_name)
        self._gen_other_attr()
        return self

    def generate_scenario(self, fail_line=None, nums=1):
        """

        :param fail_line:
        :param nums:
        """
        scn = np.ones((np.size(self.branch, axis=0), 1))
        if fail_line is None:
            ...
        elif fail_line == "sample" and nums >= 1:
            ...
        elif isinstance(fail_line, list):
            ...
        else:
            ...
        self.scenarios = scn

    def _gen_other_attr(self, graph=None, cm=None, static=False):
        self.source_idx = np.nonzero(self.bus[:, BUS_TYPE] == SOURCE)  # 找到电源索引
        bus_num = self.bus.shape[0]
        source_list = set(self.source_idx[0])
        self.load_list = [i for i in range(bus_num) if i not in source_list]
        self.pos = dict(zip(self.bus[:, BUS_I].tolist(), self.bus[:, [LNG, LAT]].tolist()))
        self.bus_dict = dict(zip(self.bus[:, BUS_I], range(bus_num)))

        edge_array = self.branch[:, [FBUS, TBUS]]
        edge_list = edge_array.tolist()

        #  Creating graph using global id
        if graph is None and not static:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(self.bus_dict)
            self.graph.add_edges_from(edge_list)
        elif not static:
            self.graph = graph
        else:
            self.graph = self.graph

        if edge_list:
            edge_array = np.frompyfunc(self.bus_dict.get, 1, 1)(edge_array)

        self.edge_array = edge_array
        self.case_info['bus_num'] = bus_num
        self.case_info['branch_num'] = self.branch.shape[0]
        self.case_info['source_num'] = len(source_list)
        # self.case_info['connected'] = nx.is_connected(self.graph)  # 连通性检查：较为耗时
        if cm is not None and isinstance(cm, dict):
            self.contracted_mapping = cm

    def from_array(self, line: np.array, node: np.array, device_proba: np.array, station_name='TestCase'):
        self.case_info['station_name'] = station_name
        self.branch = line.copy()
        self.bus = node.copy()
        self.linesafe = device_proba.copy()
        self._gen_other_attr()
        return self

    def from_pandas(self, branch: pd.DataFrame, bus: pd.DataFrame, linesafe: pd.DataFrame, station_name='TestCase'):
        self.case_info['station_name'] = station_name
        self.branch = branch.to_numpy(na_value=0.0)
        self.bus = bus.to_numpy(na_value=0.0)
        self.linesafe = linesafe.to_numpy(na_value=0.0)
        self._gen_other_attr()
        return self

    def contract_source_nodes(self, copy=True):
        if len(self.source_idx[0]) > 1:
            src_node = self.bus[self.source_idx[0][0], BUS_I]
            bh = self.branch.copy()
            bs = self.bus.copy()
            ls = self.linesafe.copy()

            other_src = list(self.source_idx[0][1:])
            other_src_id = list(bs[0].astype(np.int64).iloc[other_src].values)
            bh[:, [FBUS, TBUS]] = np.frompyfunc(lambda x: src_node if x in other_src_id else x, 1, 1)(
                bh[:, [FBUS, TBUS]])
            bs = np.delete(bs, other_src)
            if copy:
                temp_case = GridCase().from_array(bh, bs, ls, station_name=self.case_info['station_name'])
                temp_case.contracted_mapping = dict.fromkeys(other_src_id, src_node)
                return temp_case
            else:
                self.from_array(bh, bs, ls, station_name=self.case_info['station_name'])
                self.contracted_mapping = dict.fromkeys(other_src_id, src_node)
        elif len(self.source_idx[0]) == 1:
            print('算例仅含有单电源')

    def contract_load_nodes(self, node_list, copy=True):
        if len(node_list) > 1:
            src_node = self.bus[node_list[0], BUS_I]
            bh = self.branch.copy()  # type:np.array
            bs = self.bus.copy()  # type:np.array
            ls = self.linesafe.copy()  # type:np.array

            other_src = node_list[1:]
            other_src_id = bs[other_src, BUS_I].tolist()
            bh[:, [FBUS, TBUS]] = np.frompyfunc(lambda x: src_node if x in other_src_id else x, 1, 1)(
                bh[:, [FBUS, TBUS]])
            bs = np.delete(bs, other_src, axis=0)

            index_list = np.arange(0, np.size(bh, axis=0))
            equal_list = index_list[bh[:, FBUS] == bh[:, TBUS]]  # type:list
            bh = np.delete(bh, equal_list, axis=0)
            ls = np.delete(ls, equal_list, axis=0)
            if copy:
                temp_case = GridCase().from_array(bh, bs, ls, station_name=self.case_info['station_name'])
                temp_case.contracted_mapping = dict.fromkeys(other_src_id, src_node)
                return temp_case
            else:
                self.from_array(bh, bs, ls, station_name=self.case_info['station_name'])
                self.contracted_mapping = dict.fromkeys(other_src_id, src_node)
        else:
            print('输入仅含有单节点')

    def update(self, stat=False):
        """
        操作GridCase的信息数组时需要更新其他参数
        """
        self._gen_other_attr(static=stat)

    def delete_edge(self, u_id, v_id, copy=False, need_edge_id=False, static=False):
        if u_id != v_id:
            if self.bus_dict.__contains__(u_id) and self.bus_dict.__contains__(v_id):
                bs = self.bus.copy()
                ls = self.linesafe.copy()
                bh = self.branch.copy()
                index_1 = np.where((bh[:, FBUS] == u_id) & (bh[:, TBUS] == v_id))[0]
                index_2 = np.where((bh[:, FBUS] == v_id) & (bh[:, TBUS] == u_id))[0]
                total_index = np.concatenate((index_1, index_2))
                bh = np.delete(bh, total_index, axis=0)
                ls = np.delete(ls, total_index, axis=0)
                if copy:
                    case = GridCase().from_array(bh, bs, ls)
                    if need_edge_id:
                        idx = self.branch[index_1, BRH_I].tolist()
                        idx.extend(self.branch[index_2, BRH_I])
                        return case, idx
                    else:
                        return case
                else:
                    if need_edge_id:
                        idx = self.branch[index_1, BRH_I].tolist()
                        idx.extend(self.branch[index_2, BRH_I].tolist())
                        self.branch, self.bus, self.linesafe = bh, bs, ls
                        self.update(static)
                        return idx
                    else:
                        self.branch, self.bus, self.linesafe = bh, bs, ls
                        self.update(static)
        else:
            raise IndexError(
                f"节点{u_id}在系统中？{self.bus_dict.get(u_id) is not None} 节点{v_id}在系统中？{self.bus_dict.get(v_id) is not None}")

    def contract_edge(self, u_id, v_id, copy=False, need_edge_id=False, static=False):
        if u_id != v_id:
            if self.bus_dict.__contains__(u_id) and self.bus_dict.__contains__(v_id):
                bh = self.branch.copy()
                bh[bh[:, FBUS] == v_id, FBUS] = u_id
                bh[bh[:, TBUS] == v_id, TBUS] = u_id
                bs = np.delete(self.bus, self.bus_dict[v_id], axis=0)
                index_list = np.arange(0, np.size(bh, axis=0))
                index = index_list[bh[:, FBUS] == bh[:, TBUS]]
                bh = np.delete(bh, index, axis=0)
                ls = np.delete(self.linesafe, index, axis=0)
                contract_map = {}
                for k in self.contracted_mapping:
                    if self.contracted_mapping[k] == v_id:
                        contract_map[k] = u_id
                    else:
                        contract_map[k] = self.contracted_mapping[k]
                contract_map[v_id] = u_id
                if copy:
                    case = GridCase().from_array(bh, bs, ls)
                    case.contracted_mapping = contract_map
                    if need_edge_id:
                        return case, self.branch[index, BRH_I].tolist()
                    else:
                        return case
                else:
                    if need_edge_id:
                        idx = self.branch[index, BRH_I].tolist()
                        self.branch, self.bus, self.linesafe = bh, bs, ls
                        self.update(static)
                        self.contracted_mapping = contract_map
                        return idx
                    else:
                        self.branch, self.bus, self.linesafe = bh, bs, ls
                        self.update(static)
                        self.contracted_mapping = contract_map
            else:
                raise IndexError(
                    f"节点{u_id}在系统中？{self.bus_dict.get(u_id) is not None} 节点{v_id}在系统中？{self.bus_dict.get(v_id) is not None}")

    def change_bus_type(self, bus_map: dict, copy=False, index=True):
        """

        :param bus_map
        :param copy
        :param index
        """
        if copy:
            bs = deepcopy(self.bus)
            for k in bus_map:
                if index:
                    b_m = k
                else:
                    b_m = np.nonzero(bs[:, BUS_I] == k)[0][0]
                bs[b_m, BUS_TYPE] = bus_map[k]
            return GridCase().from_array(self.branch, bs, self.linesafe)
        else:
            for k in bus_map:
                if index:
                    b_m = k
                else:
                    b_m = np.nonzero(self.bus[:, BUS_I] == k)[0][0]
                self.bus[b_m, BUS_TYPE] = bus_map[k]
            self.update()

    def add_edge(self, u_id, v_id, edge_id=None, linesafe=None, copy=False):
        if linesafe is None:  # 生成线路完好概率
            linesafe = np.ones((1, np.size(self.linesafe, axis=1)))
        if edge_id is None:  # 线路ID生成
            edge_id = np.int_(self.branch[-1, BRH_I] + 1)
        if u_id in self.bus_dict and v_id in self.bus_dict:
            if copy:
                bh = self.branch.copy()
                bh_new_row = bh[-1, :].copy().reshape(1, -1)
                bh_new_row[0, BRH_I] = edge_id
                bh_new_row[0, FBUS] = u_id
                bh_new_row[0, TBUS] = v_id
                bh = np.concatenate((bh, bh_new_row), axis=0)
                ls = self.linesafe.copy()
                ls = np.concatenate((ls, linesafe.reshape(1, -1)), axis=0)
                return GridCase().from_array(bh, self.bus, ls)
            else:
                bh_new_row = self.branch[-1, :].copy().reshape(1, -1)
                bh_new_row[0, BRH_I] = edge_id
                bh_new_row[0, FBUS] = u_id
                bh_new_row[0, TBUS] = v_id
                self.branch = np.concatenate((self.branch, bh_new_row), axis=0)
                self.linesafe = np.concatenate((self.linesafe, linesafe.reshape(1, -1)), axis=0)
                self.update()
        else:
            raise IndexError(
                f'{u_id}在系统中？{self.bus_dict.get(u_id) is not None}，{v_id}在系统中？{self.bus_dict.get(v_id) is not None}')

    def add_node(self, node_id, node_type=LOAD, copy=False, lng=0.0, lat=0.0, power=1.0, value=1.0):
        bus_new_row = self.bus[-1, :].copy().reshape(1, -1)
        bus_new_row[0, [BUS_I, LNG, LAT, BUS_TYPE, PLOAD, VALUE]] = np.array(
            [node_id, lng, lat, node_type, power, value])
        if copy:
            bs = np.concatenate((self.bus.copy(), bus_new_row), axis=0)
            return GridCase().from_array(self.branch.copy(), bs, self.linesafe.copy())
        else:
            self.bus = np.concatenate((self.bus, bus_new_row), axis=0)
            self.update()

    def remove_node(self, node_id, copy=False):
        # TODO: 删除节点
        pass

    def extract_subgraph_by_nodes(self, nodes: list):
        """
        提取包含部分节点的子图

        :param nodes: [list] 子图包含的节点
        :return: 返回一个新的GridCase对象
        """
        if nodes:
            node_filter = np.vectorize(lambda x: 1 if x in nodes else 0)
            bh = self.branch.copy()
            bs = self.bus.copy()
            ls = self.linesafe.copy()
            bs_index = np.arange(np.size(bs, axis=0))
            bh_index = np.arange(np.size(bh, axis=0))
            bs_cond = node_filter(bs[:, BUS_I])
            bh_cond = np.sum(node_filter(bh[:, [FBUS, TBUS]]), axis=1)
            bs_index = bs_index[bs_cond == 1]
            bh_index = bh_index[bh_cond == 2]
            return GridCase().from_array(bh[bh_index], bs[bs_index], ls[bh_index],
                                         station_name=self.case_info['station_name'] + f'_subgraph_{nodes}')
        else:
            return pickle.loads(pickle.dumps(self))

    def extract_load_subgraph(self):
        return self.extract_subgraph_by_nodes(self.bus[self.load_list, BUS_I].tolist())

    def profile_creation(self):
        lp = LineProfiler()
        lp.add_function(GridCase.__init__)
        lp.add_function(GridCase._gen_other_attr)
        lp_wrapper = lp(self.from_array)
        lp_wrapper(self.branch, self.bus, self.linesafe)
        lp.print_stats()

    def profile_generation(self):
        lp = LineProfiler()
        lp.add_function(GridCase._gen_other_attr)
        lp_wrapper = lp(self.update)
        lp_wrapper()
        lp.print_stats()

    def profile_deletion(self, copy=True):
        lp = LineProfiler()
        lp.add_function(GridCase._gen_other_attr)
        lp_wrapper = lp(self.delete_edge)
        lp_wrapper(self.bus[0, BUS_I], self.bus[1, BUS_I], copy=copy)
        lp.print_stats()

    def profile_contraction(self, copy=True):
        lp = LineProfiler()
        lp.add_function(GridCase._gen_other_attr)
        lp_wrapper = lp(self.contract_edge)
        lp_wrapper(self.bus[0, BUS_I], self.bus[1, BUS_I], copy=copy)
        lp.print_stats()

    def delete_edge_by_order(self, order: list, copy=False):
        """
        TODO: 多条边同时delete，防止冗余self.update()
        """
        ...

    def contract_edges_by_order(self, order: list, copy=False):
        """
        TODO: 多条边同时Contract
        """

    def update_charging_station_data(self, cs: pd.DataFrame):
        self.extend_attr['charging_station'] = cs.set_index('充电站位置')

    def update_meteorological_data(self, met: pd.DataFrame):
        station_attrs = ["观测站id", "经度", "纬度"]
        data_cat = set(met.columns) - set(station_attrs)
        self.extend_attr['met'] = {}
        stations = met['观测站id'].unique().tolist()
        indexed_met = met.set_index('观测站id')

        lnt = [indexed_met['经度'].loc[st].iloc[0] for st in stations]
        lat = [indexed_met['纬度'].loc[st].iloc[0] for st in stations]
        self.extend_attr['met']['经纬度'] = pd.DataFrame(list(zip(stations, lnt, lat)),
                                                      columns=station_attrs).set_index("观测站id")
        for cat in data_cat:
            this_cat = pd.DataFrame(
                np.concatenate([indexed_met[cat].loc[st].to_numpy().reshape(1, -1) for st in stations],
                               axis=0), index=stations)

            self.extend_attr['met'][cat] = this_cat


if __name__ == "__main__":
    from GridResilience.GridCase import *
    from GridResilience.SparseSolver import *

    # grid = case_32_modified()

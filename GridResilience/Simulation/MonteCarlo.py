from copy import deepcopy

import ray
from tqdm import tqdm
from Environment import Case
import numpy as np
import igraph as ig
import networkx as nx
import ray.util.multiprocessing as mp


def MonteCarlo_Prob(case: Case, num_batch=100, batch=1000, t=0):
    num_batch = int(num_batch)
    bus_num = len(case.bus)
    h = nx.Graph()
    h.add_nodes_from(case.graph.nodes)
    source_list = list(case.source_idx[0])
    loads = [i for i in case.graph.nodes if i not in source_list]
    edge_set = np.random.binomial(np.ones(num_batch * batch).astype(int).reshape(-1, 1),
                                  (1 - (1 - case.linesafe[:, t])).reshape(1, -1))

    def __calc__(k):
        bus_fail = np.zeros(bus_num)
        bus_fail[source_list] = batch
        bus_fail_single = np.zeros((num_batch, bus_num))
        cursor = 0
        for s in range(k * batch, (k + 1) * batch):
            temp_case = case.edge_array[np.nonzero(edge_set[s, :])[0], :]
            temp_graph = deepcopy(h)
            temp_graph.add_edges_from(temp_case)
            for i in loads:
                is_connected = False
                for j in source_list:
                    if nx.has_path(temp_graph, i, j):
                        is_connected = True
                        break
                    else:
                        is_connected |= False
                bus_fail[i] += is_connected
                bus_fail_single[cursor, i] = is_connected
            cursor += 1
        return bus_fail / batch

    pool = mp.Pool()
    res = pool.map_async(__calc__, range(num_batch))
    return np.sum(res.get(), axis=0) / num_batch


def MonteCarlo_conditional_prob_node(case: Case, condition=1, query=0, t=0):
    bus_num = len(case.bus)
    h = nx.Graph()
    h.add_nodes_from(case.graph.nodes)
    source_list = list(case.source_idx[0])
    loads = [i for i in case.graph.nodes if i not in source_list]
    num = 1000000
    edge_set = np.random.binomial(np.ones(num).astype(int).reshape(-1, 1),
                                  (1 - (1 - case.linesafe[:, t])).reshape(1, -1))
    bus_fail = np.zeros(bus_num)
    bus_fail[source_list] = 1
    bus_fail_single = np.zeros((num, bus_num))
    cursor = 0
    for s in tqdm(range(num)):
        temp_case = case.edge_array[np.nonzero(edge_set[s, :])[0], :]
        temp_graph = deepcopy(h)
        temp_graph.add_edges_from(temp_case)
        for i in loads:
            is_connected = False
            for j in source_list:
                if nx.has_path(temp_graph, i, j):
                    is_connected = True
                    break
                else:
                    is_connected |= False
            bus_fail[i] += is_connected
            bus_fail_single[cursor, i] = is_connected
        cursor += 1
    return bus_fail_single


if __name__ == '__main__':
    from GridCase import *

    if not ray.is_initialized():
        ray.init()
    filepath = '../../FinalCodes/data/'
    station_name = '4'
    case = case_32_modified()
    import time

    st = time.time()
    res = MonteCarlo_Prob(case)
    print(f'蒙特卡洛用时{time.time() - st}')

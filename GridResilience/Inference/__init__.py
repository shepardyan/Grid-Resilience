import networkx as nx
import numpy as np
from SparseSolver import *
from Inference.pyAgrumInf import *
import os


def network_outage_inference(case: GridCase, survive: list, fail: list):
    # 先验知识层
    first_layer = None
    # 后验推断层
    fail = fail[0]
    inference_graph = deepcopy(case.graph)
    inference_graph.remove_node(fail)
    line_inf = np.zeros(len(case.branch))
    frozen_line_safe = deepcopy(case.linesafe)
    line_fail = 1 - case.linesafe
    ref_p, _ = prob_solver(case)
    print(f'ref = {ref_p}')
    local_case = deepcopy(case)
    ind = []
    for i in inference_graph.edges:
        ind.append(local_case.edge_dict[i])
    for i in tqdm(ind):
        local_case.linesafe = deepcopy(frozen_line_safe)
        local_case.linesafe[i, :] *= 0.0
        p, _ = prob_solver(local_case, funtol=1e-10)
        line_inf[i] = 1 - (1 - p[fail]) * line_fail[i] / (1 - ref_p[fail])
    print(line_inf)
    local_case.linesafe = line_inf.reshape(-1, 1)
    ind = np.array(ind)
    branch = case.branch[ind, :]
    bus = case.bus
    linesafe = line_inf[ind].reshape(-1, 1)
    inf_case = GridCase().from_array(branch, bus, linesafe)
    final_p, _ = prob_solver(inf_case, funtol=1e-10)
    return final_p


if __name__ == "__main__":
    # station_name = 'IEEE14'
    # filepath = '../../data/'
    # case = Case().from_file(filepath=filepath, station_name=station_name)
    from GridCase import *
    from ZDDSolver import *
    import subprocess

    case = case4_loop()
    bus_num = len(case.bus)
    survive = []
    solver = ZDDSolver()
    fail = [2, 3]
    fail_name = case.bus[fail[0], BUS_I]

    case.contract_load_nodes(fail, copy=False)

    fail_index = np.nonzero(case.bus[:, BUS_I] == fail_name)[0][0]
    no_fail_res = solver.solve(case)
    R_new = case.linesafe.copy()

    for i in range(len(R_new)):
        temp_case = deepcopy(case)
        u, v = case.branch[i, [FBUS, TBUS]]
        temp_case.delete_edge(u, v, copy=False)
        try:
            new_prob = solver.solve(temp_case)
            R_new[i] = 1 - ((1 - case.linesafe[i, 0]) * (new_prob[fail_index]) / (no_fail_res[fail_index]))
        except subprocess.CalledProcessError:
            print(f'u={u}, v={v}')
    case.linesafe = R_new
    case.update()
    delete_index = []
    for i in range(len(case.branch)):
        if case.branch[i, FBUS] == fail_name or case.branch[i, TBUS] == fail_name:
            delete_index.append(i)
    bh = case.branch.copy()
    ls = case.linesafe.copy()
    bh = np.delete(bh, delete_index, axis=0)
    ls = np.delete(ls, delete_index, axis=0)
    case = GridCase().from_array(bh, case.bus, ls)

    final_res = solver.solve(case)

    import matplotlib.pyplot as plt

    plt.plot(range(len(case.bus)), no_fail_res)
    plt.plot(range(len(case.bus)), final_res)

    post_prob = np.zeros(bus_num)
    for i in range(len(final_res)):
        post_prob[case.bus_dict[case.bus[i, BUS_I]]] = final_res[i]
    # res = network_outage_inference(case, survive, fail)

    # plt.plot(range(len(case.bus)), res)
    # plt.show()

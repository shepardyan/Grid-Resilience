import networkx as nx
import numpy as np
from pandapower.networks import case33bw
import pandapower as pp
from pandapower.topology import create_nxgraph
from tqdm import tqdm
import ray
import pandapower.networks as pn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    net = pn.case30()  # 生成一个新网络
    pp.rundcpp(net)
    ray.init()  # 并行计算Cluster初始化
    line_reliability = 0.9  # 所有线路都为0.9可靠性
    num = 100  # 仿真场景数
    edge_set = np.random.binomial(np.ones(num).astype(int).reshape(-1, 1),
                                  line_reliability * np.ones(len(net.line))).T  # 随机开断线路
    edge_set_lst = []
    for scenario in range(num):
        edge_set_lst.append(list(np.nonzero(edge_set[:, scenario] == 0)[0]))  # 随机开断线路列表


    @ray.remote
    def _pf_simulation(edge_set_list):
        res = np.zeros(len(net.line))
        bus_res = np.zeros(len(net.bus))
        simulation_net = pn.case30()
        simulation_net.load['p_mw'] = simulation_net.load['p_mw'].map(lambda x: 1.1 * x)
        simulation_net.line['in_service'] = simulation_net.line['in_service'].map(lambda x: True)
        # simulation_net.line.drop(edge_set_list, inplace=True)
        simulation_net.line['r_ohm_per_km'].iloc[edge_set_list] = simulation_net.line['r_ohm_per_km'].iloc[
            edge_set_list].map(lambda x: 1000.0)
        simulation_net.line['x_ohm_per_km'].iloc[edge_set_list] = simulation_net.line['x_ohm_per_km'].iloc[
            edge_set_list].map(lambda x: 1000.0)
        try:
            pp.rundcpp(simulation_net, check_connectivity=True, algorithm='iwamoto_nr', max_iteration=20)
            cursor = 0
            for k in range(len(net.line)):
                if k in edge_set_list:
                    res[k] = 0.0
                else:
                    res[k] = np.abs(simulation_net.res_line['i_from_ka'].iloc[cursor])
                    cursor += 1
            bus_res = simulation_net.res_bus['vm_pu'].to_numpy()
        except pp.LoadflowNotConverged:
            print('产生潮流不收敛问题')
        return res, bus_res, simulation_net


    ray_result = ray.get([_pf_simulation.remote(ed) for ed in tqdm(edge_set_lst, desc='潮流仿真')])
    edge_res = np.zeros((len(net.line), num))
    node_res = np.zeros((len(net.bus), num))
    net_list = []
    #  edge_res是线路电流(kA)，node_res是节点电压(p.u.)
    for i, result in enumerate(ray_result):
        edge_res[:, i] = result[0]
        node_res[:, i] = result[1]
        net_list.append(result[2])

    #
    edge_dec = np.zeros((np.size(edge_res, axis=0), np.size(edge_res, axis=1)))
    for k in range(np.size(edge_res, axis=1)):
        edge_dec[:, k] = edge_res[:, k] > net.line['max_i_ka']
    node_dec = (node_res > 0.0) & (node_res < 0.9)
    edge_dec = edge_dec > 0.5
    ax1 = plt.subplot(2, 1, 1)
    plt.matshow(edge_dec)
    ax2 = plt.subplot(2, 1, 2)
    plt.matshow(node_dec)
    plt.show()

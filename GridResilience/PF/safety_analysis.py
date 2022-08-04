import pandapower as pp
import pandas as pd
from pandapower.networks import case_ieee30
import scipy.sparse.linalg as ssl
import scipy.sparse as ss
import scipy
import numpy as np
from math import pi

if __name__ == "__main__":
    net = case_ieee30()
    pp.rundcpp(net)
    B0 = net._ppc['internal']['Bbus']  # B矩阵
    ext_B = []
    ext_B_i = []
    ext_B_j = []
    for i in range(len(net.line)):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        ext_B.extend([1 / net.line['x_ohm_per_km'].iloc[i]] * 2)
        ext_B_i.extend([u, v])
        ext_B_j.extend([v, u])
    ext_B = ss.coo_matrix((ext_B, (ext_B_i, ext_B_j)), shape=(len(net.bus), len(net.bus)))
    B0 = B0.toarray()[1:, 1:]
    Z = np.linalg.inv(B0)  # B矩阵的逆(满阵)
    theta0 = (net.res_bus['va_degree'].to_numpy()) * pi / 360.0  # 节点电压相角
    # 试验：假设线路8-9故障，计算开断潮流
    k = 11 - 1
    m = 13 - 1
    line_index = 10
    b_km = B0[k, m]
    M_k, M_m = np.zeros(30), np.zeros(30)
    M_k[k], M_m[m] = 1.0, 1.0
    M_k = M_k.reshape(-1, 1)
    M_m = M_m.reshape(-1, 1)
    theta = theta0[1:] - np.real(
        -(1 / b_km + Z[k, k] + Z[m, m] - Z[k, m] - Z[m, k]) ** (-1) * (theta0[k] - theta0[m]) * (
                Z[k, :] - Z[m, :]))

    theta = np.insert(theta, 0, 0.0)
    P = np.zeros(len(net.line))
    P_pf = net.res_line['p_from_mw'].to_numpy()
    ind = []
    for i in range(len(net.line)):
        u, v = net.line['from_bus'].iloc[i], net.line['to_bus'].iloc[i]
        if i != line_index:
            P[i] = P_pf[i] * (theta[u] - theta[v]) / (theta0[u] - theta0[v])
        ind.append(i)
    res = pd.DataFrame(data=np.concatenate((P_pf.reshape(-1, 1), P.reshape(-1, 1)), axis=1), columns=['直流潮流', '开断潮流'],
                       index=ind)


    def simulation(line):
        sim = case_ieee30()
        sim.line = sim.line.drop(index=line)
        pp.rundcpp(sim, check_connectivity=True)
        this_res = sim.res_line['p_from_mw'].to_numpy()
        this_res = np.insert(this_res, line, 0.0)
        return this_res


    import matplotlib.pyplot as plt

    # plt.plot(ind, np.abs(P_pf), label="DC PowerFlow")
    plt.plot(ind, np.abs(P) / net._ppc['baseMVA'], label="N-1 Breaking")
    plt.plot(ind, np.abs(simulation(line_index)) / net._ppc['baseMVA'], label="N-1 Simulation")
    plt.legend(fontsize=14)
    plt.xlabel('Line ID', fontsize=14)
    plt.ylabel('Active Power/p.u.', fontsize=14)
    plt.show()

    p0 = np.abs(P) / net._ppc['baseMVA']
    p1 = np.abs(simulation(line_index)) / net._ppc['baseMVA']
    np.seterr(divide='ignore')
    error = 100 * np.abs(p0 - p1) / p1

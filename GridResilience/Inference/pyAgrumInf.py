from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import numpy as np
import networkx as nx
from Environment import *
import pyAgrum as gum
import os
from tqdm import tqdm


def pgm_formulation(pgm_case: GridCase, survive=None, time=-1):
    DG = nx.DiGraph()
    DG.add_nodes_from([str(i) for i in pgm_case.graph.nodes])
    DG.add_edges_from([(str(u), str(v)) for u, v in pgm_case.edge_array])
    BN = BayesianNetwork(DG.edges())
    if survive is None:
        survive = pgm_case.source_idx[0]
    tabular_source, evidence = generate_tabular_sources(survive)
    pred_dict = {}
    for i in DG.nodes:
        pred_dict[str(i)] = {}
    for i in range(len(pgm_case.branch)):
        pred_dict[str(pgm_case.bus_dict[pgm_case.branch[i, TBUS]])][str(pgm_case.bus_dict[pgm_case.branch[i, FBUS]])] = \
            pgm_case.linesafe[i, time]
    for i in tqdm(pgm_case.graph.nodes, desc='CPT Generating Process'):
        if i not in list(pgm_case.source_idx[0]):
            cpd = generate_tabular_load(list(pred_dict[str(i)].keys()), str(i), list(pred_dict[str(i)].values()))
            BN.add_cpds(cpd)
    for tab in tabular_source:
        BN.add_cpds(tab)
    return BN, evidence


def generate_tabular_sources(sources):
    tab_sources = []
    evidence = {}
    for i in sources:
        tab_sources.append(TabularCPD(variable=str(i), variable_card=2, values=[[0], [1]]))
        evidence[str(i)] = 1
    return tab_sources, evidence


def generate_tabular_load(pred: list, succ, prob: list):
    """
    规定0为失电状态，1为有电状态
    :param pred: 上游节点 指向待分析节点
    :param succ: 下游节点（待分析节点）
    :param prob: 有电概率
    :return: succ节点的TabularCPD
    """
    pred_num = len(pred)
    prob_yes = np.array(prob)
    up_list = []
    low_list = []
    for i in range(2 ** pred_num):
        factor_str = str.zfill(bin(i)[2:], pred_num)
        factor = np.array(list(factor_str)).astype(int)
        temp_prob = np.prod(1 - prob_yes.reshape(1, -1) * factor.reshape(1, -1))
        up_list.append(temp_prob)
        low_list.append(1 - temp_prob)
    return TabularCPD(variable=succ, variable_card=2, evidence=pred, evidence_card=[2 for _ in range(len(pred))],
                      values=[up_list, low_list])


def construct_radial_bayesian_network(case: GridCase, new_file=False):
    if not os.path.exists(case.case_info['station_name'] + '.bifxml') or new_file:
        BN, evidence = pgm_formulation(case)
        BN.save(case.case_info['station_name'] + '.bifxml', filetype='xmlbif')
    bnet = gum.loadBN(case.case_info['station_name'] + '.bifxml')
    return bnet


def do_inference(case: GridCase, survive=None, lost=None, new_file=False):
    bn = construct_radial_bayesian_network(case, new_file=new_file)
    ie = gum.LazyPropagation(bn)
    if survive:
        for i in survive:
            ie.addEvidence(str(i), 1)
    if lost:
        for j in lost:
            ie.addEvidence(str(j), 0)
    ie.makeInference()
    prob_array = np.zeros((len(case.bus), 2))
    for i in tqdm(case.graph.nodes, desc='推断进度'):
        if i not in case.source_idx[0]:
            prob_array[i, :] = ie.posterior(bn.idFromName(str(i))).toarray()
        else:
            prob_array[i, :] = np.array([0.0, 1.0])
    return prob_array[:, 1].flatten()


if __name__ == "__main__":
    from GridCase import *

    branch = np.array([10001, 0, 1,
                       10002, 0, 2,
                       10003, 1, 3, ]).reshape(-1, 3)
    bus = np.array([0, 0, 5, 110, 1, 0, 1, 0,
                    1, 2, 3, 333, 1, 21.7, 0, 0,
                    2, -2, 3, 333, 1, 94.2, 0, 0,
                    3, 0, 1, 333, 1, 47.8, 0, 0]).reshape(-1, 8)
    linesafe = np.array([0.9] * len(branch)).reshape(-1, 1)
    grid = GridCase().from_array(branch, bus, linesafe)

    do_inference(grid, survive=[0, 2], lost=[3])

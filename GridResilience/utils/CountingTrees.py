import networkx as nx
from GridResilience.Environment import *
import scipy.sparse.linalg as ssl
import numpy as np


def counting_spanning_trees(case: GridCase, exp=False):
    """
    返回网络的生成树个数（基尔霍夫定律）
    :param case: 算例
    :return: integer 生成树个数
    """
    lap = nx.laplacian_matrix(case.graph)
    i = 0
    small_lap = lap[i + 1:, i + 1:]
    lu = ssl.splu(small_lap.tocsc())
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    if exp:
        return np.log10(diagL.prod()) + np.log10(diagU.prod())
    else:
        det = diagL.prod() * diagU.prod()
        return int(np.round(det))


if __name__ == "__main__":
    from GridCase import *

    case = case_32_modified()
    res = counting_spanning_trees(case, exp=False)

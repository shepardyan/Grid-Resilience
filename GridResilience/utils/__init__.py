import json
from json import JSONEncoder

import scipy.sparse.linalg as ssl

from .Topology import *
import numpy as np


class Profiler:
    def __init__(self, func):
        from line_profiler import LineProfiler
        self.res = LineProfiler()
        self.wrapper = self.res(func)


class NumpyEncoder(json.JSONEncoder):
    """
    重写json模块JSONEncoder类中的default方法
    """

    def default(self, obj):
        # np整数转为内置int
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(JSONEncoder, self).default(obj)


def counting_spanning_trees(case: Case, exp=False):
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


def accessibility_matrix(G: nx.Graph):
    return list(nx.connected_components(G))


def is_all_connected(case: Case):
    return len(list(nx.connected_components(case.graph))) == 1





if __name__ == "__main__":
    station_name = 'beiliuzhan'
    filepath = '../../data/'
    case = Case().from_file(filepath=filepath, station_name=station_name)
    print(is_all_connected(case))

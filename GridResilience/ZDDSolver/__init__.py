from copy import deepcopy

import numpy as np
import subprocess
from tqdm import tqdm
from GridResilience.Environment import *
import platform
import os


class ZDDSolver:
    def __init__(self):
        abs_file = os.path.abspath(__file__)
        self._sysstr = platform.system()
        if self._sysstr == 'Windows':
            exec_path = os.path.dirname(abs_file)
            self._zdd = exec_path + "\\reliability.exe"
        elif self._sysstr == 'Linux':
            exec_path = abs_file[:abs_file.rfind("/")]
            self._zdd = exec_path + "/reliability"
        else:
            raise EnvironmentError
        self._terminal_file = "terminal.dat"
        self._graph_file = "graph.dat"
        self._probability_file = "probability.dat"
        self.solve_cmd = str.split(f"{self._zdd} {self._graph_file} {self._terminal_file} {self._probability_file}")

    def solve(self, case, node=None, source=None, t=0):
        np.savetxt(self._graph_file, np.int_(case.edge_array), fmt="%d")
        np.savetxt(self._probability_file, case.linesafe[:, t].reshape(1, -1), fmt="%.4f")

        def _gen_terminal(src, sink):
            np.savetxt(self._terminal_file, np.int_(np.array([src, sink]).reshape(1, -1)), fmt="%d")

        def _gen_terminal_list(src, loads):
            node_list = [src]
            node_list.extend(loads)
            np.savetxt(self._terminal_file, np.int_(np.array(node_list).reshape(1, -1)), fmt="%d")

        if node is None:
            prob = np.ones(len(case.bus))

            for i in tqdm(range(1, len(case.bus)), desc='节点概率计算进度：'):
                if source is None:
                    _gen_terminal(src=case.source_idx[0][0], sink=i)
                else:
                    _gen_terminal(src=source, sink=i)
                try:
                    output = subprocess.check_output(self.solve_cmd, stderr=subprocess.STDOUT).decode()
                    start = output.find('prob = ')
                    output = output[start:].split('\n')[0]
                    end = output.find('\r')
                    res_str = output[7:end]
                    prob[i] = float(res_str)
                except subprocess.CalledProcessError:
                    prob[i] = 0.0
        elif isinstance(node, int):
            i = node
            if source is None:
                _gen_terminal(src=case.source_idx[0][0], sink=i)
            else:
                _gen_terminal(src=source, sink=i)
            output = subprocess.check_output(self.solve_cmd, stderr=subprocess.STDOUT).decode().split('\n')[5]
            start = output.find('prob = ')
            prob = float(output[start + 7:])
        elif isinstance(node, list):
            if source is None:
                _gen_terminal_list(src=case.source_idx[0][0], loads=node)
            else:
                _gen_terminal_list(src=source, loads=node)
            output = subprocess.check_output(self.solve_cmd, stderr=subprocess.STDOUT).decode().split('\n')[5]
            start = output.find('prob = ')
            prob = float(output[start + 7:])
        else:
            raise ValueError
        return prob

    def sensitivity_matrix(self, case, edge=None, node=None, delta=1e-4):
        if edge is None and node is None:
            sense = np.zeros((np.size(case.bus, axis=0), np.size(case.branch, axis=0)))
            init_res = self.solve(case)
            for i in range(np.size(case.branch, axis=0)):
                sub = deepcopy(case)
                sub.linesafe[i] += delta
                delta_res = self.solve(sub)
                sense[:, i] = np.abs(delta_res - init_res) / delta
            return sense


if __name__ == "__main__":
    from GridResilience.GridCase import *
    from GridResilience.SparseSolver import *
    from pandapower.networks import case33bw
    import matplotlib.pyplot as plt

    plt.style.use(['science', 'no-latex', 'std-colors'])
    plt.rcParams['font.family'] = 'STsong'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.unicode_minus'] = False

    case = case_32_modified()

    case.linesafe *= 0.9

    solver = ZDDSolver()
    sense_mat = solver.sensitivity_matrix(case)

from GridResilience.Environment import *
from .Prob import *
import time


class SparseSolver:
    def __init__(self):
        self.iter_count = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.Converged = False

    def solve(self, case):
        self.start_time = time.time()
        prob_solver(case)
        self.end_time = time.time()

    def print_stats(self):
        if self.Converged:
            print(f'牛顿法共进行{self.iter_count}次迭代， 于{self.end_time - self.start_time}s内收敛')
        else:
            raise ValueError('SparseSolver未完成求解')

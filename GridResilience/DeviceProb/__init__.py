from GridResilience.Environment import *
import numpy as np


def markov_process(lam, mu, t, p0=1.0):
    assert len(lam) == len(mu)
    assert len(lam) == len(t) - 1
    np.seterr(all='ignore')
    ts = range(max(t) + 1)
    p = [1.0 for _ in ts]
    for i in range(len(lam)):
        l, m = lam[i], mu[i]
        t_init, t_end = t[i], t[i + 1]
        p_init = p[t_init]
        for j in range(t_init + 1, t_end + 1):
            t_now = j - t_init
            if m > 50:
                coef = 1.0
            elif m == 0.0 and l == 0.0:
                coef = 1.0
            else:
                coef = np.divide(m, (l + m))
            p[j] = coef * (1 - np.exp(- (l + m) * t_now)) + p_init * np.exp(- (l + m) * t_now)
    return ts, p


if __name__ == "__main__":
    λ = [0.02, 0.00005, 0.00, 0.00]
    μ = [0.0, 0.01, 0.05, 0.10]
    t = [0, 20, 30, 180, 200]
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    plt.style.use(['science', 'no-latex', 'std-colors'])
    plt.rcParams['font.family'] = 'STsong'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.unicode_minus'] = False
    ts, p = markov_process(λ, μ, t)
    xs = np.linspace(0, 21, 201)
    model = interp1d(ts, p, kind='cubic')

    from SparseSolver import *
    from GridCase import case_32_modified

    grid = case_32_modified()
    grid.branch = grid.branch[:31]
    grid.linesafe = grid.linesafe[:31, :]
    grid.update()
    grid.linesafe = np.ones((31, 201))
    for i in ts:
        grid.linesafe[:, i] *= p[i]
    res = prob_multi_period(grid)

    r = np.sum(res[0].to_numpy(), axis=0) / 32

    plt.plot(xs, p, '--', label='元件可用概率', linewidth=2)
    plt.plot(xs, r, label='系统性能指标', linewidth=2)
    plt.fill_between(xs, r, np.ones_like(xs), facecolor='blue', alpha=0.3, label='系统失电风险')
    plt.legend()

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == "__main__":
    from GridCase import *

    case = case_32_modified()
    post_prob = [1.00000, 0.69263, 0.61284, 0.56419, 0.54473, 0.50245, 0.48439, 0.39648, 0.35417, 0.00000, 0.00000,
                 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.71502, 0.55214, 0.48632, 0.40827, 0.61296,
                 0.55465, 0.51602, 0.51695, 0.49973, 0.49277, 0.49597, 0.41889, 0.35380, 0.29882, 0.00000
                 ]
    plt.rcParams['figure.figsize'] = (15, 10)
    matplotlib.rc("font", family='HarmonyOS Sans SC')
    fail = [9, 10, 11, 12, 13, 31]
    total_set = np.nonzero(np.array(post_prob) == 0.0)[0].tolist()
    lost_power = [i + 1 for i in total_set if i not in fail]
    maybe = np.nonzero(np.array(post_prob) < 0.5)[0].tolist()

    others = [i + 1 for i in range(32) if i not in total_set and i not in maybe]
    maybe = [i + 1 for i in maybe if i not in total_set]
    fail = [i + 1 for i in fail]
    # nx.draw_networkx_nodes(case.graph, nodelist=fail, pos=case.pos, node_shape='s', label='探明失电节点',
    #                        node_color='tab:red')
    # nx.draw_networkx_nodes(case.graph, nodelist=lost_power, pos=case.pos, node_shape='^', label='推断失电节点',
    #                        node_color="tab:brown")
    # nx.draw_networkx_nodes(case.graph, nodelist=maybe, pos=case.pos, node_shape='p', label='可能失电节点',
    #                        node_color="tab:blue")
    #
    # nx.draw_networkx_nodes(case.graph, nodelist=others, pos=case.pos, node_shape='o', label='其他节点', node_color='green')
    # nx.draw_networkx_edges(case.graph, pos=case.pos)
    # labels = {}
    # for i in range(len(post_prob)):
    #     labels[i + 1.0] = i
    # nx.draw_networkx_labels(case.graph, pos=case.pos, labels=labels, font_size=14)
    # plt.legend(fontsize=17)
    # plt.show()
    plt.bar(range(32), post_prob)
    plt.axhline(y=0.5, c="gray", ls="--", lw=2)
    plt.ylabel('Posterior probability', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('Node ID', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

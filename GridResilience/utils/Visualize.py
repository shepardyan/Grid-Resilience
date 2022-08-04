"""
Visualize network graph using NetGraph package
"""
from matplotlib import pyplot as plt
from netgraph import InteractiveGraph, Graph
import networkx as nx
from GridResilience.Environment import *
import numpy as np
import matplotlib as mpl


def weighted_graph_edge_plot(case):
    draw_graph = []
    draw_graph_dict = {}
    for i, (u, v) in enumerate(list(case.edge_array)):
        sorted_tuple = tuple(sorted([u, v]))
        if draw_graph_dict.__contains__(sorted_tuple):
            draw_graph_dict[sorted_tuple] *= case.linesafe[i, -1]
        else:
            draw_graph_dict[sorted_tuple] = case.linesafe[i, -1]

    for key in draw_graph_dict:
        draw_graph.append((key[0], key[1], draw_graph_dict[key]))
    cmap = mpl.colormaps['viridis']
    node_dict = {node: node for node in case.graph.nodes}
    fig = plt.figure()
    Graph(draw_graph, edge_cmap=cmap, edge_width=1.5, node_layout='dot', node_labels=node_dict,
          node_label_fontdict={'size': 14})
    plt.show()
    c = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap))


if __name__ == "__main__":
    from GridCase import *
    from pandapower.plotting import *
    import pandapower as pp

    plt.rcParams['figure.figsize'] = (15, 10)
    import matplotlib
    from pandapower.plotting import cmap_continuous, create_bus_collection, draw_collections, create_line_collection

    cmap_list = [(0.00, "blue"), (1.5, "green"), (3.0, "red")]
    cmap, norm = cmap_continuous(cmap_list)
    matplotlib.rc("font", family='HarmonyOS Sans SC')
    # case = case_32_modified()
    from pandapower.networks import case30

    case = case_32_modified()
    case30_prob = np.array(
        [0.10000, 0.10000, 0.10000, 0.10000, 0.10000, 0.10001, 0.10000, 0.10001, 0.10000, 0.10001, 0.10000, 0.10002,
         0.10000, 0.10000, 0.10000, 0.10000, 0.10002, 0.10019, 0.10022, 0.10039, 0.10086, 0.10170, 0.10070, 0.10001,
         0.10001, 0.10007, 0.10001, 0.10016, 0.10034, 0.11212, 0.10212, 0.11168, 0.10127, 0.10000, 0.10128, 0.10000,
         0.10000, 0.10000, 0.10000, 0.10001, 0.10002
         ])
    sense = np.array(
        [8.33683, 2.77964, 2.54029, 2.47143, 2.77154, 2.76058, 1.59401, 0.60936, 0.49514, 0.55324, 0.77252, 0.66939,
         0.73468, 2.62046, 2.21448, 1.94073, 5.66792, 5.27206, 5.04333, 1.30255, 2.97354, 2.59062, 2.36450, 2.01754,
         1.75227, 1.64120, 1.68309, 2.49896, 2.12864, 1.88961, 1.77919, 1.30639, 1.17476, 1.14943, 1.79617, 2.29269])
    new_sense = np.array(
        [1.22826, 1.05978, 0.91534, 0.95554, 1.04807, 1.16678, 0.88287, 0.86738, 0.74102, 0.80301, 1.01534, 0.90139,
         0.97170, 2.61592, 2.11767, 1.76552, 0.53393, 0.39694, 0.45583, 0.77385, 2.04895, 1.70888, 1.53482, 1.94281,
         1.64369, 1.50853, 1.53585, 2.11405, 1.76312, 1.55435, 1.48541, 0.66731, 0.71340, 1.42656, 1.55554, 1.52482])
    zdd_sense = np.array(
        [8.33684, 2.77964, 2.54029, 2.47144, 2.77154, 2.76058, 1.59402, 0.60936, 0.49514, 0.55324, 0.77252, 0.66939,
         0.73468, 2.62046, 2.21448, 1.94073, 5.66792, 5.27206, 5.04333, 1.30256, 2.97355, 2.59062, 2.36451, 2.01754,
         1.75227, 1.64120, 1.68309, 2.49896, 2.12865, 1.88961, 1.77920, 1.30640, 1.17476, 1.14944, 1.79617, 2.29269
         ])

    net = nx.Graph()
    for i, e in enumerate(case.branch[:, [FBUS, TBUS]]):
        net.add_edge(e[0], e[1], weight=sense[i])
    label = dict(zip(range(len(case.bus)), range(len(case.bus))))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    nx.draw_networkx_nodes(net, pos=case.pos, node_size=500)
    colors = [net.edges[u, i]['weight'] for u, i in net.edges]
    Edges = []
    edges = nx.draw_networkx_edges(net, pos=case.pos, edge_color=colors,
                                   width=5.5, edge_cmap=plt.cm.Reds, edge_vmin=0, alpha=0.9)
    pc = mpl.collections.PatchCollection(Edges, cmap=plt.cm.Reds)
    pc.set_array(colors)
    nx.draw_networkx_labels(net, pos=case.pos, labels=case.bus_dict)

    plt.colorbar(pc)
    ax.set_axis_off()
    plt.savefig(r"C:\Users\yunqi\Desktop\Sense", dpi=300)
    plt.show()

    # 潮流可视化
    # net = nx.Graph()
    # pp.rundcpp(case)
    # label = dict(zip(range(len(case.bus)), range(len(case.bus))))
    # pf_res = case.res_line['loading_percent'].to_numpy() / 100
    # pos = {}
    # for i in range(len(case.bus_geodata)):
    #     pos[i] = (case.bus_geodata['x'].loc[i], case.bus_geodata['y'].loc[i])
    # for i in range(len(case.line)):
    #     u, v = case.line['from_bus'].iloc[i], case.line['to_bus'].iloc[i]
    #     net.add_edge(u, v, weight=case30_prob[i])
    # bc = create_bus_collection(case, case.bus.index, size=0.1, zorder=2)
    # gc = create_bus_collection(case, case.gen['bus'], size=0.1, zorder=2, color='r')
    # ec = create_bus_collection(case, case.ext_grid['bus'], size=0.1, zorder=2, color='purple')
    # lc = create_line_collection(case, use_bus_geodata=True, z=case30_prob, cmap=plt.cm.Reds,
    #                             cbar_title="线路越限概率", linewidths=2, zorder=1)
    # ax = draw_collections([lc, bc, gc, ec], figsize=(15, 12), plot_colorbars=False)
    # nx.draw_networkx_labels(net, pos=pos, labels=label, font_size=14, font_color="whitesmoke")
    # cb = plt.colorbar(lc, ax=ax)
    # cb.ax.tick_params(labelsize=16)
    # font = {
    #     'size': 22,
    # }
    # cb.set_label("线路停运概率", fontdict=font)
    #
    # ax.set_axis_off()
    # import matplotlib.patches as mpatches
    # from matplotlib.legend_handler import HandlerPatch
    #
    #
    # class HandlerEllipse(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                        xdescent, ydescent, width, height, fontsize, trans):
    #         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
    #         p = mpatches.Circle(xy=center)
    #         self.update_prop(p, orig_handle, legend)
    #         p.set_transform(trans)
    #         return [p]
    #
    #
    # pv_patch = mpatches.Circle((0, 0), color='r', label="PV节点")
    # pq_patch = mpatches.Circle((0, 0), color='blue', label="PQ节点")
    # slack_patch = mpatches.Circle((0, 0), color='purple', label="Vθ节点")
    # plt.legend([pv_patch, pq_patch, slack_patch], ["PV节点", "PQ节点", "Vθ节点"], fontsize=22,
    #            handler_map={mpatches.Circle: HandlerEllipse()})
    # plt.title("线路越限场景概率", fontsize=22)
    # plt.savefig(r"C:\Users\yunqi\Desktop\PFRisk.jpg", dpi=300)
    # plt.show()

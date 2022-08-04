import networkx as nx


def accessibility_matrix(G: nx.Graph):
    return list(nx.connected_components(G))


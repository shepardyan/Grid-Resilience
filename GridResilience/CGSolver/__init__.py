import GridResilience.Environment
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, hessian
import networkx as nx
from jax import lax
from functools import partial
import jax

jax.config.update('jax_platform_name', 'cpu')
import time
import numpy as np


def rsub(x):
    """
    输出 element-wise 1 - x

    :param x: 输入JAX数组
    :type x: jnp.array
    """
    return jnp.subtract(1.0, x)


def rprod(input_nodes):
    return jnp.prod(jnp.array([rsub(node) for node in input_nodes]))


def mul_sub_prod(rel, input_nodes):
    return rel * rsub(rprod(input_nodes))


def sub_prod(input_nodes):
    return rsub(rprod(input_nodes))


def risk(value_tensor, input_tensor):
    return value_tensor @ input_tensor


def risk_evaluation(branch_rel, calc_graph: nx.DiGraph, a_list: list, b_list: list, c_list: list,
                    edge_dict: dict, G: nx.Graph, src: list):
    branch_var_list = [None for _ in range(calc_graph.number_of_nodes())]

    for i in a_list:
        branch_var_list[i] = jnp.zeros(shape=(), dtype=jnp.float32)
    for i in b_list:
        branch_var_list[i] = branch_rel[i // 2]
    for i in c_list:
        branch_var_list[i] = mul_sub_prod(branch_rel[i // 2], [branch_var_list[j] for j in calc_graph.pred[i]])
    nodal_var_list = [None for _ in range(G.number_of_nodes())]
    load = [n for n in G.nodes if n not in src]

    for k in load:
        nodal_var_list[k] = rsub(rprod([branch_var_list[edge_dict[(k, j)]] for j in G.adj[k]]))
    for k in src:
        nodal_var_list[k] = jnp.ones(shape=(), dtype=jnp.float32)
    nodal_var = jnp.array(nodal_var_list)
    return nodal_var
    # return nodal_var_list


def topology_analysis(case, requires_function=False):
    _local_graph = nx.relabel_nodes(case.graph, case.bus_dict, copy=True)
    a_type_list = []
    b_type_list = []
    source_list = case.source_idx[0].tolist()
    G_edge_list = [e for e in _local_graph.edges]
    G_edge_dict = {}
    computation_graph = nx.DiGraph()
    for index, (from_node, to_node) in enumerate(G_edge_list):
        computation_graph.add_nodes_from([(from_node, to_node), (to_node, from_node)])
        G_edge_dict[(from_node, to_node)] = index * 2
        G_edge_dict[(to_node, from_node)] = index * 2 + 1
        if from_node in source_list:
            a_type_list.append(index * 2)
            b_type_list.append(index * 2 + 1)
        elif to_node in source_list:
            b_type_list.append(index * 2)
            a_type_list.append(index * 2 + 1)
        else:
            for nbr in _local_graph.adj[to_node]:
                if nbr != from_node:
                    computation_graph.add_edge((to_node, nbr), (from_node, to_node))
            for nbr in _local_graph.adj[from_node]:
                if nbr != to_node:
                    computation_graph.add_edge((from_node, nbr), (to_node, from_node))
    nx.relabel_nodes(computation_graph, mapping=G_edge_dict, copy=False)
    assert nx.is_directed_acyclic_graph(computation_graph)
    topology_order = list(nx.topological_sort(computation_graph))
    c_type_list = [var_index for var_index in topology_order if
                   var_index not in a_type_list and var_index not in b_type_list]
    depth = {}
    for i in topology_order:
        try:
            depth_nbr = [depth[j] for j in computation_graph.pred[i]]
            if depth_nbr:
                depth[i] = int(np.max(depth_nbr)) + 1
            else:
                depth[i] = 0
        except KeyError:
            depth[i] = 0
    layer = {}
    for i in c_type_list:
        this_depth = depth[i]
        if not layer.__contains__(this_depth):
            layer[this_depth] = [i]
        else:
            layer[this_depth].append(i)

    if requires_function:
        return lambda rel: risk_evaluation(rel, computation_graph, a_type_list, b_type_list, c_type_list, G_edge_dict,
                                           _local_graph, source_list)
    else:
        return computation_graph, a_type_list, b_type_list, c_type_list, list(
            _local_graph.nodes), G_edge_dict, _local_graph, source_list, depth, layer


if __name__ == "__main__":
    from jax import random
    from GridResilience.GridCase import *
    from GridResilience.Environment import *

    grid = case4_loop()

    loss_evaluation = jit(topology_analysis(grid, requires_function=True))  # 拓扑分析
    loss_jacobian = jit(jacfwd(loss_evaluation))

    risk_vmap = jit(vmap(risk))


    @jit
    def repair_effect(rel):
        data = jnp.broadcast_to(rel, (rel.shape[0], rel.shape[0])) + jnp.diag((1.0 - rel).flatten())
        return risk(rel) - risk_vmap(data)


    risk_jacobian = jit(jacfwd(risk))
    grid.linesafe *= 0.5
    start = time.time()
    loss = risk(grid.linesafe)
    a = 0.45 * np.ones((4, 4))
    for i in range(4):
        a[i, i] = 1

    effect = repair_effect(grid.linesafe)
    print(loss)
    print(risk_jacobian(grid.linesafe))
    grid.linesafe[0, :] = 1
    print(risk(grid.linesafe))
    print(risk(grid.linesafe) - loss)
    print(f'计算时间{time.time() - start}')


    def hessian(f):
        return jacfwd(jacrev(f))


    hess_mat = hessian(risk)(grid.linesafe.flatten())


    @jit
    def distance(p, result):
        return jnp.linalg.norm(p - result, ord=jnp.inf)


    @jit
    def optim(line_prob):
        return distance(loss_evaluation(line_prob), aa)


    @jit
    def fun_gen(line_prob):
        return loss_evaluation(line_prob) - aa


    jac_gen = jit(jacfwd(fun_gen))


    @jit
    def optimal(lp):
        line_prob = jax.nn.sigmoid(lp)
        cost = line_prob @ jnp.array([0.8, 0.8, 1.3, 1.3])
        return jnp.array([0.0, 0.2, 1.3, 2.7]) @ (1 - loss_evaluation(line_prob)) + cost


    optimal_jac = jit(grad(optimal))

    optim_jac = jit(jacfwd(optim))
    optim_hess = jit(hessian(optim))


    @jit
    def one_optim(line, index, line_prob):
        lp = line_prob.at[index].set(line)
        return optim(lp)


    # import optax
    from tqdm import tqdm

    # one_optim_jac = jit(jacfwd(one_optim))
    # iter_max = 30
    # init_x0 = jnp.array([0.5, 0.5])
    # x0 = init_x0
    # optimizer = optax.adam(learning_rate=0.0001)
    # params = init_x0
    # opt_state = optimizer.init(params)
    # res = minimize(optim, x0=init_x0, jac=optim_jac)
    # params = res.x
    # print(params)
    # print(ls)
    # print(loss_evaluation(params))
    # print(aa)

    # optim(jnp.array([0.99, 0.99, 0.99, 0.99]))

    # while optim(x0) >= 1e-7:
    #     fun = fun_gen(x0)
    #     jac = jac_gen(x0)
    #     x0 -= jnp.linalg.solve(jac, fun)
    #     print(f'Optimal = {optim(x0)}')

import pickle
from jax import jit
import jax
import numpy as np
import jax.numpy as jnp
from flax.nnx import Module


# === a) ===
class Graph:
    @classmethod
    def from_edge_list(cls, edge_list, weight_list):
        graph = cls()

        edge_list_array = jnp.array(edge_list)
        weight_list_array = jnp.array(weight_list)

        graph.src = jnp.concatenate([edge_list_array[:, 0], edge_list_array[:, 1]])
        graph.dst = jnp.concatenate([edge_list_array[:, 1], edge_list_array[:, 0]])
        graph.w = jnp.concatenate([weight_list_array, weight_list_array])

        graph.edges = jnp.array(edge_list)
        graph.weights = jnp.array(weight_list)

        return graph


# Pytree registration functions for Graph:
def _graph_flatten(graph):
    children = (graph.src, graph.dst, graph.w)
    return children, (graph.edges, graph.weights)


def _graph_unflatten(aux_data, children):
    graph = Graph()
    graph.src, graph.dst, graph.w = children
    graph.edges, graph.weights = aux_data
    return graph


# Register Graph as a pytree
jax.tree_util.register_pytree_node(Graph, _graph_flatten, _graph_unflatten)


# === b) ===
@jit
def gcn_layer(params, graph, data):
    # extract source and destination indices from edges
    src = graph.src
    dst = graph.dst
    weights = graph.w
    
    # Weighted data (in this case weights are all 1s)
    src_weighted_data = graph.w[:, None] * data[graph.src]

    num_nodes = data.shape[0]
    # perform W * X
    WX = jax.ops.segment_sum(data=src_weighted_data, segment_ids=src, num_segments=num_nodes)

    Y = jnp.dot(WX, params)

    return Y


# === c) ===
class GCN(Module):
    pass


# === d) ===
class Network(Module):
    pass


def main():
    with open("zachary.pickle", "rb") as f:
        social_graph, labels, dummy_input, dummy_output, dummy_params = pickle.load(f)
    G = Graph.from_edge_list(social_graph, [1] * len(social_graph))
    output = gcn_layer(dummy_params, G, dummy_input)
    assert np.isclose(output, dummy_output).all()

    layer = GCN()
    layer.theta = dummy_params
    output = layer(G, dummy_input)
    assert np.isclose(output, dummy_output).all()


if __name__ == "__main__":
    main()

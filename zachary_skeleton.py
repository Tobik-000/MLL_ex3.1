import pickle
from jax import jit
import jax
import numpy as np
import optax
import jax.numpy as jnp
from flax.nnx import Module
from flax.nnx import Optimizer, value_and_grad, apply_updates
from typing import Callable, Optional


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
        graph.num_nodes = jnp.max(graph.dst) + 1

        graph.edges = jnp.array(edge_list)
        graph.weights = jnp.array(weight_list)

        # Add self-loops
        src_i = jnp.arange(graph.num_nodes)
        dst_i = jnp.arange(graph.num_nodes)
        w_i = jnp.ones(graph.num_nodes)

        # Combine original edges with self-loops
        src_tilde = jnp.concatenate([graph.src, src_i])
        dst_tilde = jnp.concatenate([graph.dst, dst_i])
        w_tilde = jnp.concatenate([graph.w, w_i])

        # Calculate degree
        tilde_D = jax.ops.segment_sum(
            data=w_tilde, segment_ids=dst_tilde, num_segments=graph.num_nodes
        )

        # Calculate D_tilde^{-1/2}
        D_inv_sqrt = 1.0 / jnp.sqrt(tilde_D + 1e-10)

        # Calculate W_hat
        w_hat = w_tilde * D_inv_sqrt[src_tilde] * D_inv_sqrt[dst_tilde]

        graph.w = w_hat
        graph.src = src_tilde
        graph.dst = dst_tilde

        return graph


# Pytree registration functions for Graph:
def _graph_flatten(graph):
    children = (graph.src, graph.dst, graph.w)
    aux_data = (graph.edges, graph.weights, graph.num_nodes)
    return children, aux_data


def _graph_unflatten(aux_data, children):
    graph = Graph()
    graph.src, graph.dst, graph.w = children
    graph.edges, graph.weights, graph.num_nodes = aux_data
    return graph


# Register Graph as a pytree
jax.tree_util.register_pytree_node(Graph, _graph_flatten, _graph_unflatten)


# === b) ===
@jit
def gcn_layer(params, graph, data):

    # Calculate the aggregation
    weighted_data = graph.w[:, None] * data[graph.src]

    W_hat_X = jax.ops.segment_sum(
        data=weighted_data, segment_ids=graph.dst, num_segments=graph.num_nodes
    )

    # Linear transformation
    output = jnp.dot(W_hat_X, params)
    return output


# === c) ===
class GCN(Module):

    def __init__(self, activation: Optional[Callable] = None):
        self.activation = activation

    def __call__(self, graph: Graph, data: jnp.ndarray) -> jnp.ndarray:
        if self.theta is None:
            raise ValueError("Parameter theta has not been initialized.")

        output = gcn_layer(self.theta, graph, data)
        if self.activation is not None:
            output = self.activation(output)
        return output


# === d) ===
class Network(Module):
    def __init__(
        self,
        activation: Optional[Callable] = None):
        
        self.gcn1 = GCN(activation=activation)
        self.gcn2 = GCN(activation=jax.nn.log_softmax)
        
        
    def __call__(self, graph: Graph, data: jnp.ndarray) -> jnp.ndarray:
        hidden = self.gcn1(graph, data)
        output = self.gcn2(graph, hidden)
        return output
    


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
    
    # Build the full network
    
    


if __name__ == "__main__":
    main()

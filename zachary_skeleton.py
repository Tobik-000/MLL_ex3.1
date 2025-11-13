import pickle
from jax import jit
import jax
import numpy as np
import optax
import jax.numpy as jnp
from flax.nnx import Module
from flax.nnx import Optimizer, value_and_grad
from typing import Callable, Optional
import flax.nnx as nnx


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

    def __init__(
        self,
        rng: jax.random.PRNGKey = None,
        num_nodes: int = 0,
        hidden_dim: int = 0,
        activation: Optional[Callable] = None,
    ):
        self.activation = activation
        #self.theta: Optional[jnp.ndarray] = None
        # Initialize parameters if rng and dimensions are provided
        if rng is not None and num_nodes > 0 and hidden_dim > 0:
            self.theta = nnx.Param(jax.random.uniform(rng, (num_nodes, hidden_dim)))

    def __call__(self, graph: Graph, data: jnp.ndarray) -> jnp.ndarray:
        if self.theta is None:
            raise ValueError("Parameter theta has not been initialized.")

        output = gcn_layer(self.theta.value, graph, data)
        if self.activation is not None:
            output = self.activation(output)
        return output


# === d) ===
class Network(Module):
    def __init__(
        self,
        rng: jax.random.PRNGKey,
        num_nodes: int,
        hidden_dim: int,
        activation: Optional[Callable] = None,
    ):
        self.gcn1 = GCN(
            rng=rng, num_nodes=num_nodes, hidden_dim=hidden_dim, activation=jax.nn.relu
        )
        self.gcn2 = GCN(rng=rng, num_nodes=hidden_dim, hidden_dim=2)

    def __call__(self, graph: Graph, data: jnp.ndarray) -> jnp.ndarray:
        hidden = self.gcn1(graph, data)
        output = self.gcn2(graph, hidden)
        return output


def loss_fn(model: Network, graph: Graph, data: jnp.ndarray, labels: jnp.ndarray):
    logits = model(graph, data)

    labels = jnp.asarray(labels)
    if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[-1] != 2):
        labels = jax.nn.one_hot(labels, 2).astype(jnp.float32)
    else:
        labels = labels.astype(jnp.float32)

    losses = optax.sigmoid_binary_cross_entropy(logits, labels=labels)

    # Create a mask to only consider first and last nodes
    mask = jnp.zeros_like(losses)
    mask = mask.at[0].set(1.0)
    mask = mask.at[-1].set(1.0)

    loss = jnp.mean(losses * mask)
    return loss


def accuracy_fn(model: Network, graph: Graph, data: jnp.ndarray, labels: jnp.ndarray):
    logits = model(graph, data)
    preds = jnp.argmax(logits, axis=1)
    labels = jnp.asarray(labels)

    N = logits.shape[0]
    mask = jnp.ones(N, dtype=bool)
    mask = mask.at[0].set(False)
    mask = mask.at[-1].set(False)

    acc = jnp.mean((preds == labels)[mask])

    return acc


def train_step(
    model: Network,
    graph: Graph,
    data: jnp.ndarray,
    labels: jnp.ndarray,
    optimizer: Optimizer,
):
    loss, grads = value_and_grad(lambda m: loss_fn(m, graph, data, labels))(model)

    optimizer.update(grads)

    return optimizer.model, optimizer, loss


def main():
    with open("zachary.pickle", "rb") as f:
        social_graph, labels, dummy_input, dummy_output, dummy_params = pickle.load(f)
    G = Graph.from_edge_list(social_graph, [1] * len(social_graph))
    output = gcn_layer(dummy_params, G, dummy_input)
    assert np.isclose(output, dummy_output).all()

    layer = GCN()
    layer.theta = nnx.Param(dummy_params)
    output = layer(G, dummy_input)
    assert np.isclose(output, dummy_output).all()

    # Parameters for network
    N = G.num_nodes
    hidden_dim = 16
    X = jnp.eye(N)

    key = jax.random.PRNGKey(0)

    # Build the full network
    model = Network(rng=key, num_nodes=N, hidden_dim=hidden_dim, activation=jax.nn.relu)

    optimizer = nnx.ModelAndOptimizer(model, optax.adam(learning_rate=0.01))

    for epoch in range(100):
        model, optimizer, loss = train_step(model, G, X, labels, optimizer)
        if epoch % 10 == 0:
            acc = accuracy_fn(model, G, X, labels)
            print(f"Epoch {epoch:03d}, Loss: {float(loss):.4f}, Acc: {float(acc):.4f}")

    final_acc = accuracy_fn(model, G, X, labels)
    print(f"Final Loss: {float(loss):.4f}, Final Accuracy: {float(final_acc):.4f}")


if __name__ == "__main__":
    main()

import torch
import numba
import numpy as np
from numba.typed import List


class TokenCharacterTrie:
    """A trie data structure for efficient token-to-character mapping."""

    def __init__(self, decode):
        """Initialize a `TokenCharacterTrie`.

        Args:
            decode (list): List representing the token vocabulary.
                Each element of the list must be iterable.
        """
        self.decode = decode
        self.word2leaf = {}
        self.children = [{}]  # First node is root
        self.root = 0
        self.token_id_to_leaf = []

        for token_id, word in enumerate(self.decode):
            curr = self.root
            for letter in word:
                if letter not in self.children[curr]:
                    self.children[curr][letter] = len(self.children)
                    self.children.append({})
                curr = self.children[curr][letter]

            self.children[curr][None] = last = len(self.children)
            self.children.append({})
            assert word not in self.word2leaf, (
                "Can't have duplicate words in vocabulary"
            )
            self.word2leaf[word] = last

            self.token_id_to_leaf.append((token_id, last))

        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))
        self.jump = List(
            [np.array(sorted(x.values()), dtype=np.int32) for x in self.children]
        )
        self.ordering = np.array(list(self._order(self.root)), np.int32)

        # Renumber the states of the trie so that they are named by a contiguous
        # range of integers and those integers respect the are topologically
        # ordering of the trie topology.  This improves the efficiency of the
        # updating the trie as it improves memory locality.
        ordering = {}
        for i, x in enumerate(self._order_full(self.root)):
            ordering[x] = i
        self._rename(f=lambda x: ordering[x])

        node2prefix = {self.root: []}
        for x in reversed(range(len(self.children))):
            for letter, y in self.children[x].items():
                if letter is None:
                    node2prefix[y] = node2prefix[x]
                else:
                    node2prefix[y] = node2prefix[x] + [letter]
        self.node2prefix = node2prefix

    def _rename(self, f):
        """Rename all node indices in the trie using the provided mapping function.

        Args:
            f (callable): Function that maps old node indices to new node indices
        """
        N = len(self.children)

        new_children = [{} for _ in range(N)]
        nodes = range(N)

        for x in nodes:
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        self.root = f(self.root)
        self.children = new_children
        self.word2leaf = {w: f(x) for w, x in self.word2leaf.items()}
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))

        self.token_id_to_leaf = np.array(
            [(i, f(x)) for i, x in self.token_id_to_leaf], dtype=np.int32
        )

        self.ordering = np.array([f(x) for x in self.ordering])
        self.jump = List(
            [np.array(sorted(x.values()), dtype=np.int32) for x in new_children]
        )

    def _alloc_weights(self):
        """Allocate an array to store weight values for all nodes.

        Returns:
            np.ndarray: Zero-initialized array for storing weight values
        """
        return np.zeros(len(self.children), dtype=np.float64)

    def _preprocess_ws(self, ws):
        """Preprocess the weight vector to ensure it is a numpy array and on the correct device.

        Args:
            ws (torch.Tensor|np.ndarray): Token weights over the vocabulary of shape `(len(self.decode),)`

        Returns:
            (np.ndarray): Weight vector
        """
        if isinstance(ws, torch.Tensor):
            if ws.device.type != "cpu":
                ws = ws.cpu()
            ws = ws.numpy()
        return ws

    def weight_sum(self, ws):
        """Compute weight sum for each node in the trie.

        For each node in the trie, this computes the sum of weights of all leaf nodes (tokens)
        that are descendants of that node.

        Args:
            ws (torch.Tensor|np.ndarray): Token weights over the vocabulary of shape `(len(self.decode),)`

        Returns:
            (np.ndarray): Summed weights for each node in the trie.
        """
        ws = self._preprocess_ws(ws)
        node_ws = self._alloc_weights()
        _update_trie_numba_sum(
            node_ws=node_ws,
            ws=ws,
            token_id_to_leaf=self.token_id_to_leaf,
            jump=self.jump,
            ordering=self.ordering,
        )
        return node_ws

    def weight_max(self, ws):
        """Compute weight max for each node in the trie.

        For each node in the trie, this computes the maximum weight among all leaf nodes (tokens)
        that are descendants of that node.

        Args:
            ws (torch.Tensor|np.ndarray): Token weights over the vocabulary of shape `(len(self.decode),)`

        Returns:
            (np.ndarray): Weight max values for each node in the trie.
        """
        ws = self._preprocess_ws(ws)
        node_ws = self._alloc_weights()
        _update_trie_numba_max(
            node_ws=node_ws,
            ws=ws,
            token_id_to_leaf=self.token_id_to_leaf,
            jump=self.jump,
            ordering=self.ordering,
        )
        return node_ws

    def batch_weight_sum(self, ws):
        """Batched equivalent of `weight_sum`.

        Args:
            ws (list[torch.Tensor|np.ndarray]): Batch of token weights, each of shape `(len(self.decode),)`

        Returns:
            (np.ndarray): Batch of weight values of `len(ws)` for each node in the trie
        """
        return np.array([self.weight_sum(ws) for ws in ws])

    def batch_weight_max(self, ws):
        """Batched equivalent of `weight_max`.

        Args:
            ws (list[torch.Tensor|np.ndarray]): Batch of token weights, each of shape `(len(self.decode),)`

        Returns:
            (np.ndarray): Batch of weight max values of `len(ws)` for each node in the trie
        """
        return np.array([self.weight_max(ws) for ws in ws])

    def _order(self, node):
        """Generate a topological ordering of nodes beneath the given node.

        Args:
            node (int): Starting node index

        Yields:
            int: Node indices in topological order
        """
        for a in self.children[node]:
            if a is None:
                pass
            else:
                yield from self._order(self.children[node][a])
        yield node

    def _order_full(self, node):
        """Generate a complete topological ordering including all child nodes.

        Args:
            node (int): Starting node index

        Yields:
            (int): Node indices in complete topological order
        """
        for a in self.children[node]:
            yield from self._order_full(self.children[node][a])
        yield node

    def visualize(self, ws=None):
        """Visualize the trie structure using Graphviz.

        Args:
            ws (np.ndarray|None): Optional weight vector to display at each node.
                                Should be of length `len(self.children)`.

        Returns:
            (graphviz.Digraph): The generated graph object
        """
        try:
            import graphviz
        except ImportError:  # pragma: no cover
            raise ImportError(
                "Please install graphviz: pip install graphviz"
            )  # pragma: no cover

        if ws is not None and len(ws) != len(self.children):
            raise ValueError(
                f"Weight vector length ({len(ws)}) must match number of nodes ({len(self.children)})"
            )

        dot = graphviz.Digraph(comment="Token Character Trie")
        dot.attr(rankdir="LR")

        # Create a subgraph for the legend
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize="10")
            legend.attr("node", fontsize="7", width="0.1", height="0.1")

            # Example internal node
            legend.node(
                "legend_internal",
                "Internal Node ID\n'Prefix'\nWeight (if provided)",
                shape="circle",
            )

            # Example leaf node
            legend.node("legend_leaf", "Complete Token", shape="doublecircle")

            legend.edge(
                "legend_internal",
                "legend_leaf",
                label="Token item",
                fontsize="10",
            )

            # Align legend horizontally
            legend.attr(rankdir="TB")
            legend.attr(rank="same")

        # Add the main trie nodes and edges
        for node_id in range(len(self.children)):
            prefix = self.node2prefix[node_id]

            if ws is not None:
                label = f"{node_id}\n'{prefix}'\n{ws[node_id]:.4f}"
            else:
                label = f"{node_id}\n'{prefix}'"

            # Color nodes based on mass if provided
            if ws is not None:
                max_ws = ws.max()
                if max_ws > 0:
                    intensity = int(255 * (1 - ws[node_id] / max_ws))
                    color = f"#{intensity:02x}{255:02x}{intensity:02x}"
                else:
                    color = "#ffffff"  # white for zero mass
            else:
                color = "#ffffff"  # default white

            if node_id in self.leaf2word:
                dot.node(
                    str(node_id),
                    label,
                    shape="doublecircle",
                    style="filled",
                    fillcolor=color,
                )
            else:
                dot.node(
                    str(node_id), label, shape="circle", style="filled", fillcolor=color
                )

        for node_id, children in enumerate(self.children):
            for char, child_id in children.items():
                if char is not None:
                    edge_label = str(char)
                else:
                    edge_label = "End-of-Token"

                dot.edge(str(node_id), str(child_id), label=edge_label)

        return dot


@numba.jit(nopython=True)
def _update_trie_numba_sum(
    node_ws: numba.float64[:],
    ws: numba.float64[:],
    jump: List[numba.int32[:]],
    token_id_to_leaf: numba.int32[:, :],
    ordering: numba.int32[:],
):  # pragma: no cover
    # update leaves
    M = token_id_to_leaf.shape[0]
    for k in range(M):
        i = token_id_to_leaf[k, 0]
        x = token_id_to_leaf[k, 1]
        node_ws[x] = ws[i]

    # update internal nodes
    N = ordering.shape[0]
    for i in range(N):
        node = ordering[i]
        total_ws = 0
        for child in jump[node]:
            total_ws += node_ws[child]
        node_ws[node] = total_ws


@numba.jit(nopython=True)
def _update_trie_numba_max(
    node_ws: numba.float64[:],
    ws: numba.float64[:],
    jump: List[numba.int32[:]],
    token_id_to_leaf: numba.int32[:, :],
    ordering: numba.int32[:],
):  # pragma: no cover
    # update leaves
    M = token_id_to_leaf.shape[0]
    for k in range(M):
        i = token_id_to_leaf[k, 0]
        x = token_id_to_leaf[k, 1]
        node_ws[x] = ws[i]

    # update internal nodes
    N = ordering.shape[0]
    for i in range(N):
        node = ordering[i]
        total_w = 0
        for child in jump[node]:
            total_w = max(total_w, node_ws[child])
        node_ws[node] = total_w

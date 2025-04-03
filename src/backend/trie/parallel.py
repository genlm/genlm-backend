import torch
from genlm_backend.trie.base import TokenCharacterTrie


class ParallelTokenCharacterTrie(TokenCharacterTrie):
    """A GPU-optimized version of `TokenCharacterTrie` that performs weight sum and max operations in parallel."""

    def __init__(self, decode, device=None, **kwargs):
        super().__init__(decode, **kwargs)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda' or None")

        self._build_reachability_matrix()
        self.token_ids = torch.tensor(
            self.token_id_to_leaf[:, 0], dtype=torch.long, device=self.device
        )

    def _build_parent_map(self):
        """Builds a mapping from each node to its parent node in the trie.

        Returns:
            (dict): A dictionary where keys are child nodes and values are their parent nodes.
        """
        parent = {}
        for node in range(len(self.children)):
            for child in self.jump[node]:
                parent[child] = node
        return parent

    def _build_reachability_matrix(self):
        """Constructs a sparse reachability matrix for efficient weight propagation.

        The matrix M is constructed such that M[i,j] = 1 if node j is either:
        - The leaf node i itself (self-connection)
        - An ancestor of leaf node i in the trie
        """
        leaf_indices = self.token_id_to_leaf[:, 1]
        parent = self._build_parent_map()

        rows, cols = [], []
        for i, node in enumerate(leaf_indices):
            # self connections
            rows.append(i)
            cols.append(node)

            current = node
            while current in parent:  # Walk up to root
                ancestor = parent[current]
                rows.append(i)
                cols.append(ancestor)
                current = ancestor

        self.src_indices = torch.tensor(rows, dtype=torch.long, device=self.device)
        self.dst_indices = torch.tensor(cols, dtype=torch.long, device=self.device)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)

        self.M = torch.sparse_coo_tensor(
            indices, values, (len(leaf_indices), len(self.children))
        ).to_sparse_csr()

    def _preprocess_ws(self, batch_ws):
        processed_batch_ws = []
        for ws in batch_ws:
            if not isinstance(ws, torch.Tensor):
                ws = torch.tensor(ws, device=self.device, dtype=torch.float32)
            elif ws.device != self.device or ws.dtype != torch.float32:
                ws = ws.to(device=self.device, dtype=torch.float32)
            assert ws.shape[0] == len(self.decode), [ws.shape[0], len(self.decode)]
            processed_batch_ws.append(ws)
        return torch.stack(processed_batch_ws)

    def weight_sum(self, ws):
        """Computes weight sums given token weights.

        For each node in the trie, this computes the sum of weights of all leaf nodes (tokens)
        that are descendants of that node. This is efficiently implemented using sparse matrix multiplication
        with a pre-computed reachability matrix.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.decode)`,).

        Returns:
            (numpy.ndarray): Summed weights for each node in the trie, shape (`len(self.decode)`,).
        """
        return self.batch_weight_sum(self._preprocess_ws([ws]))[0]

    def batch_weight_sum(self, ws):
        """Batch version of `weight_sum`.

        Args:
            ws (torch.Tensor): Batch of token weights, shape (batch_size × `len(self.decode)`).

        Returns:
            numpy.ndarray: Summed weights for each node in the trie, shape (batch_size × num_nodes).
        """
        ws = self._preprocess_ws(ws)
        masses = torch.sparse.mm(ws[:, self.token_ids], self.M)
        return masses.cpu().numpy()

    def weight_max(self, ws):
        """Computes the max weights given the token weights.

        For each node in the trie, this computes the maximum weight among all leaf nodes (tokens)
        that are descendants of that node. This is efficiently implemented using parallel scatter_reduce
        operations on GPU.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.decode)`,).

        Returns:
            (numpy.ndarray): Maximum weights for each node in the trie, shape (`len(self.decode)`,).
        """
        return self.batch_weight_max(self._preprocess_ws([ws]))[0]

    def batch_weight_max(self, ws):
        """Batch version of `weight_max`.

        Args:
            ws (torch.Tensor): Batch of token weights, shape (batch_size × `len(self.decode)`).

        Returns:
            (numpy.ndarray): Maximum weights for each node in the trie, shape (batch_size × num_nodes).
        """
        ws = self._preprocess_ws(ws)

        # Get leaf weights
        leaf_weights = ws[:, self.token_ids]  # shape: (batch_size × num_leafs)
        batch_size = leaf_weights.shape[0]

        # Use scatter_reduce to propagate maximum values in parallel
        result = torch.zeros((batch_size, len(self.children)), device=self.device)
        result.scatter_reduce_(
            dim=1,
            index=self.dst_indices.expand(batch_size, -1),
            src=leaf_weights[:, self.src_indices],
            reduce="amax",
            include_self=False,
        )

        return result.cpu().numpy()

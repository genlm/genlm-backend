import torch
from genlm_backend.trie.base import TokenCharacterTrie


class ParallelTokenCharacterTrie(TokenCharacterTrie):
    """A GPU-optimized version of `TokenCharacterTrie` that performs `mass_sum` in parallel.

    Inherits from `TokenCharacterTrie`.

    The mass at leaf nodes is propagated to their ancestors through sparse matrix
    multiplication with a reachability matrix. The reachability matrix is constructed at initialization.

    Implementation details:\n
        The reachability matrix M is a num_leafs × num_nodes matrix
        where M[i,j] = 1 if:\n
            - leaf_indices[i] == j (self connection) or
            - j is an ancestor of leaf_indices[i] in the trie

        Example:\n
            Trie:          M:
                 0           [[1, 1, 0, 1],
                / \\           [1, 0, 1, 0]]
               1   2 (leaf index = 1)
               |
               3 (leaf index = 0)

        The matrix is stored as a sparse tensor in CSR (Compressed Sparse Row) format,
        built from COO (Coordinate) format. For example,\n
            rows = [1, 0, 1, 0, 0] (index of leaf node)
            cols = [2, 3, 0, 1, 0] (connections)
            vals = [1, 1, 1, 1, 1] (connection weights)

        When computing masses (batch_size × num_leafs) @ M, each leaf node's mass
        flows up to all its ancestors.
    """

    def __init__(self, decode, device=None, **kwargs):
        super().__init__(decode, **kwargs)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda' or None")

        self.M = self._build_reachability_matrix()
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
        """Constructs a sparse reachability matrix for efficient mass propagation.

        The matrix M is constructed such that M[i,j] = 1 if node j is either:
        - The leaf node i itself (self-connection)
        - An ancestor of leaf node i in the trie

        Returns:
            (torch.Tensor): A sparse CSR matrix of shape (num_leafs × num_nodes)
        """
        rows, cols = [], []
        leaf_indices = self.token_id_to_leaf[:, 1]

        # add self connections
        for i, node in enumerate(leaf_indices):
            rows.append(i)
            cols.append(node)

        # add all ancestor connections
        parent = self._build_parent_map()
        for i, node in enumerate(leaf_indices):
            current = node
            while current in parent:  # Walk up to root
                ancestor = parent[current]
                rows.append(i)
                cols.append(ancestor)
                current = ancestor

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)
        M = torch.sparse_coo_tensor(
            indices, values, (len(leaf_indices), len(self.children))
        ).to_sparse_csr()

        return M

    def mass_sum(self, p_llm):
        """Computes the sum of masses for a single probability distribution.

        Args:
            p_llm (torch.Tensor): Probability distribution over tokens from the LLM.

        Returns:
            (numpy.ndarray): Summed masses for each node in the trie.
        """
        return self.batch_mass_sum(p_llm.unsqueeze(0))[0]

    def batch_mass_sum(self, p_llms):
        """Computes mass sums for a batch of probability distributions.

        Args:
            p_llms (torch.Tensor): Batch of probability distributions over tokens,
                shape (batch_size × vocab_size).

        Returns:
            (numpy.ndarray): Summed masses for each node in the trie,
                shape (batch_size × num_nodes).
        """
        if p_llms.device != self.device:
            p_llms = p_llms.to(self.device)
        masses = torch.sparse.mm(p_llms[:, self.token_ids], self.M)
        return masses.cpu().numpy()

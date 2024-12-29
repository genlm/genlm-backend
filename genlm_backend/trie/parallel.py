import torch
from genlm_backend.util import resolve_device
from genlm_backend.trie.base import TokenCharacterTrie

class ParallelTokenCharacterTrie(TokenCharacterTrie):
    """A GPU-optimized version of TokenCharacterTrie that performs mass_sum in parallel.

    The mass at leaf nodes is propagated to their ancestors
    through sparse matrix multiplication with a reachability matrix.

    The reachability matrix M is a num_leafs × num_nodes matrix
    where M[i,j] = 1 if:
        - leaf_indices[i] == j (self connection) or
        - j is an ancestor of leaf_indices[i] in the trie

    Example:
        Trie:          M:
              0           [[1, 1, 0, 1],
             / \           [1, 0, 1, 0]]
            1   2 (1)
            |
            3 (0)

    The matrix is stored as a sparse tensor in CSR (Compressed Sparse Row) format,
    built from COO (Coordinate) format. For example,
        rows = [1, 0, 1, 0, 0] (index of leaf node)
        cols = [2, 3, 0, 1, 0] (connections)
        vals = [1, 1, 1, 1, 1] (connection weights)

    When computing masses (batch_size × num_leafs) @ M, each leaf node's mass
    flows up to all its ancestors.
    """
    def __init__(self, decode, old_eos, new_eos, device=None):
        super().__init__(decode, old_eos, new_eos)
        self.device = resolve_device(device)
        self.M = self._build_reachability_matrix()
        self.token_ids = torch.tensor(
            self.token_id_to_leaf[:, 0], dtype=torch.long, device=self.device
        )

    def _build_parent_map(self):
        parent = {}
        for node in range(len(self.children)):
            for child in self.jump[node]:
                parent[child] = node
        return parent

    def _build_reachability_matrix(self):
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
            while current in parent:        # Walk up to root
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

    def batch_mass_sum(self, p_llms):
        if p_llms.device != self.device:
            p_llms = p_llms.to(self.device)
        masses = torch.sparse.mm(p_llms[:, self.token_ids], self.M)
        return masses.cpu().numpy()
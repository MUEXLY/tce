from itertools import permutations, pairwise
from typing import Sequence

from scipy.spatial import KDTree
import numpy as np
import sparse
from opt_einsum import contract

from .constants import STRUCTURE_TO_CUTOFF_LISTS, LatticeStructure, STRUCTURE_TO_THREE_BODY_LABELS


def get_adjacency_tensors_binning(
        positions: np.typing.NDArray,
        boxsize: Sequence[float],
        max_distance: float,
        max_adjacency_order: int,
        tolerance: float = 0.01
) -> sparse.COO:
    distances = get_distance_matrix(positions, boxsize, max_distance)
    _, edges = np.histogram(distances.data, bins=max_adjacency_order)

    return sparse.stack([
        sparse.where(
            sparse.logical_and(distances > (1.0 - tolerance) * low, distances < (1.0 + tolerance) * high),
            x=True, y=False
        ) for low, high in pairwise(edges)
    ])


def get_adjacency_tensors_shelling(
        positions: np.typing.NDArray,
        boxsize: Sequence[float],
        max_distance: float,
        lattice_parameter: float,
        lattice_structure: LatticeStructure,
        max_adjacency_order: int,
        tolerance: float = 0.01
) -> sparse.COO:
    distances = get_distance_matrix(positions, boxsize, max_distance)
    cutoffs = [lattice_parameter * c for c in STRUCTURE_TO_CUTOFF_LISTS[lattice_structure]]
    cutoffs.pop(0)

    return sparse.stack([
        sparse.where(
            sparse.logical_and(distances > (1.0 - tolerance) * c, distances < (1.0 + tolerance) * c),
            x=True, y=False
        ) for c in cutoffs[:max_adjacency_order]
    ])


def get_distance_matrix(
        positions: np.typing.NDArray,
        boxsize: Sequence[float],
        max_distance: float
) -> sparse.COO:

    tree = KDTree(positions, boxsize=boxsize)
    distances = tree.sparse_distance_matrix(tree, max_distance=max_distance).tocsr()
    distances.eliminate_zeros()
    return sparse.COO.from_scipy_sparse(distances)


def get_three_body_tensors(
        lattice_structure: LatticeStructure,
        adjacency_tensors: sparse.COO,
        max_three_body_order: int,
) -> sparse.COO:

    three_body_labels = [
        STRUCTURE_TO_THREE_BODY_LABELS[lattice_structure][order] for order in range(max_three_body_order)
    ]

    three_body_tensors = sparse.stack([
        sum(
            sparse.einsum(
                "ij,jk,ki->ijk",
                adjacency_tensors[i],
                adjacency_tensors[j],
                adjacency_tensors[k]
            ) for i, j, k in set(permutations(labels))
        ) for labels in three_body_labels
    ])

    return three_body_tensors


def get_feature_vector(
        adjacency_tensors: sparse.COO,
        three_body_tensors: sparse.COO,
        state_matrix: np.typing.NDArray
) -> np.typing.NDArray:

    return np.concatenate([
        contract(
            "nij,iα,jβ->nαβ",
            adjacency_tensors,
            state_matrix,
            state_matrix
        ).flatten(),
        contract(
            "nijk,iα,jβ,kγ->nαβγ",
            three_body_tensors,
            state_matrix,
            state_matrix,
            state_matrix
        ).flatten()
    ])


def symmetrize(tensor: sparse.COO, axes=None) -> sparse.COO:

    r"""
    symmetrize a tensor $T$:

    $$T_{(i_1 i_2 \cdots i_n)} = \frac{1}{n!}\sum_{\sigma\in S_n} T_{\sigma(i_1) \sigma(i_2) \cdots \sigma(i_n)}$$

    where $S_n$ is the symmetric group on $n$ elements, so we are summing over the permutations of the indices.

    e.g. $T_{(12)} = \frac{T_{12} + T_{21}}{2}$, or equivalently $\text{symmetrize}(T) = \frac{T + T^\intercal}{2}$
    """

    if not axes:
        axes = tuple(range(tensor.ndim))

    perms = list(permutations(axes))

    return sum(sparse.moveaxis(tensor, axes, perm) for perm in perms) / len(perms)


def get_feature_vector_difference(
        adjacency_tensors: sparse.COO,
        three_body_tensors: sparse.COO,
        initial_state_matrix: np.typing.NDArray,
        final_state_matrix: np.typing.NDArray
) -> np.typing.NDArray:

    sites, _ = np.where(initial_state_matrix != final_state_matrix)
    sites = np.unique(sites).tolist()

    truncated_adj = sparse.take(adjacency_tensors, sites, axis=1)
    truncated_thr = sparse.take(three_body_tensors, sites, axis=1)

    initial_feature_vec_truncated = np.concatenate(
        [
            2 * symmetrize(contract(
                "nij,iα,jβ->nαβ",
                truncated_adj,
                initial_state_matrix[sites, :],
                initial_state_matrix
            ), axes=(1, 2)).flatten(),
            3 * symmetrize(contract(
                "nijk,iα,jβ,kγ->nαβγ",
                truncated_thr,
                initial_state_matrix[sites, :],
                initial_state_matrix,
                initial_state_matrix
            ), axes=(1, 2, 3)).flatten()
        ]
    )

    final_feature_vec_truncated = np.concatenate(
        [
            2 * symmetrize(contract(
                "nij,iα,jβ->nαβ",
                truncated_adj,
                final_state_matrix[sites, :],
                final_state_matrix
            ), axes=(1, 2)).flatten(),
            3 * symmetrize(contract(
                "nijk,iα,jβ,kγ->nαβγ",
                truncated_thr,
                final_state_matrix[sites, :],
                final_state_matrix,
                final_state_matrix
            ), axes=(1, 2, 3)).flatten()
        ]
    )

    return final_feature_vec_truncated - initial_feature_vec_truncated

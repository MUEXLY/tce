r"""
This model mostly defines the `tce.structures.Supercell` object, which is mostly intended to be a private object. This
is what `tce.monte_carlo.monte_carlo` uses internally to avoid recomputing large topological tensors.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Union
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
import sparse

from .constants import (
    LatticeStructure,
    STRUCTURE_TO_ATOMIC_BASIS,
    STRUCTURE_TO_CUTOFF_LISTS,
    STRUCTURE_TO_THREE_BODY_LABELS,
    load_three_body_labels
)
from . import topology


LOGGER = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Supercell:

    r"""
    class representing a simulation supercell. `eq=True` and `frozen=True` ensure we can hash a `Supercell` instance,
    which we need to cache the topology tensors later
    """

    lattice_structure: LatticeStructure
    r"""lattice structure of the unit cell"""

    lattice_parameter: float
    r"""lattice parameter of the unit cell"""

    size: tuple[int, int, int]
    r"""size of the supercell, eg `size=(10, 10, 10)` generates a $10\times 10\times 10$ supercell"""

    @cached_property
    def num_sites(self) -> Union[int, np.integer]:

        r"""
        number of total lattice sites (NOT the number of unit cells!)
        """

        return np.prod(self.size) * STRUCTURE_TO_ATOMIC_BASIS[self.lattice_structure].shape[0]

    @cached_property
    def positions(self) -> NDArray[np.floating]:

        r"""
        positions of lattice sites
        create a meshgrid of unit cell positions, and add lattice sites at atomic basis positions in each unit cell
        """

        i, j, k = (np.arange(s) for s in self.size)

        unit_cell_positions = np.array(np.meshgrid(i, j, k, indexing='ij')).reshape(3, -1).T
        positions = unit_cell_positions[:, np.newaxis, :] + \
            STRUCTURE_TO_ATOMIC_BASIS[self.lattice_structure][np.newaxis, :, :]
        return self.lattice_parameter * positions.reshape(-1, 3)

    @lru_cache
    def adjacency_tensors(self, max_order: int, tolerance: float = 1.0e-6) -> sparse.COO:

        r"""
        two-body adjacency tensors $A_{ij}^{(n)}$. computed by binning interatomic distances

        Args:
            max_order (int):
                maximum nearest neighbor order
            tolerance (float):
                The tolerance $\varepsilon$ to include when binning interatomic distances. for example, when searching
                for a neighbor at distance $d$, we search in the shell $[(1 - \varepsilon)d, (1 + \varepsilon)d]$. this
                should be a small number. defaults to $0.01$.
        """

        return topology.get_adjacency_tensors(
            tree=KDTree(self.positions, boxsize=self.lattice_parameter * np.array(self.size)),
            cutoffs=[self.lattice_parameter * c for c in STRUCTURE_TO_CUTOFF_LISTS[self.lattice_structure][:max_order]],
            tolerance=tolerance
        )

    @lru_cache
    def three_body_tensors(self, max_order: int) -> sparse.COO:

        r"""
        three-body tensors, computed by summing the two-body tensors

        a set of labels defines each three-body tensor. e.g., in a fcc solid, the first-order triplet is formed
        by three first-nearest neighbor pairs, so its label is $ (0, 0, 0) $. similarly, the second-order triplet in fcc
        is formed by two first-nearest neighbor pairs, and one second-nearest neighbor pair, so its label is
        $ (0, 0, 1) $. we sum over the different permutations, and then stack them over the labels

        Args:
            max_order (int): Maximum three body order
        """
        try:
            labels = STRUCTURE_TO_THREE_BODY_LABELS[self.lattice_structure]
            LOGGER.debug(f"labels loaded cached entry for {self.lattice_structure}")
        except KeyError:
            labels = load_three_body_labels()[self.lattice_structure]
            LOGGER.debug(f"labels computed for {self.lattice_structure}")
        three_body_labels = [labels[order] for order in range(max_order)]

        return topology.get_three_body_tensors(
            lattice_structure=self.lattice_structure,
            adjacency_tensors=self.adjacency_tensors(max_order=np.concatenate(three_body_labels).max() + 1),
            max_three_body_order=max_order
        )

    def feature_vector(
        self,
        state_matrix: sparse.COO,
        max_adjacency_order: int,
        max_triplet_order: int
    ) -> NDArray[np.floating]:

        r"""
        feature vector $\mathbf{t}$ extracting topological features, i.e., number of bonds and number of triplets

        Args:
            state_matrix (sparse.COO):
                The state tensor $\mathbf{X}$, defined by $X_{i\alpha} = [\text{site $i$ occupied by type $\alpha$}]$,
                where $[\cdot]$ is the [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket).
            max_adjacency_order (int):
                The maximum nearest neighbor order
            max_triplet_order (int):
                The maximum three body order
        """

        return topology.get_feature_vector(
            adjacency_tensors=self.adjacency_tensors(max_order=max_adjacency_order),
            three_body_tensors=self.three_body_tensors(max_order=max_triplet_order),
            state_matrix=state_matrix
        )

    def clever_feature_diff(
        self,
        initial_state_matrix: sparse.COO,
        final_state_matrix: sparse.COO,
        max_adjacency_order: int,
        max_triplet_order: int,
    ) -> NDArray[np.floating]:

        r"""
        clever shortcut for computing feature vector difference
        $\Delta\mathbf{t} = \mathbf{t}(\mathbf{X}') - \mathbf{t}(\mathbf{X})$ between two nearby states. here, we
        perform a truncated contraction, only caring about "active" sites, or lattice sites that changed

        Args:
            initial_state_matrix (sparse.COO):
                The initial state tensor $\mathbf{X}$.
            final_state_matrix (sparse.COO):
                The final state tensor $\mathbf{X}'$.
            max_adjacency_order (int):
                The maximum nearest neighbor order
            max_triplet_order (int):
                The maximum three body order
        """

        three_body_tensors = self.three_body_tensors(max_order=max_triplet_order)

        return topology.get_feature_vector_difference(
            adjacency_tensors=self.adjacency_tensors(max_order=max_adjacency_order),
            three_body_tensors=three_body_tensors,
            initial_state_matrix=initial_state_matrix,
            final_state_matrix=final_state_matrix
        )

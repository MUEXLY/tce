from itertools import product

import numpy as np

from tce import structures, topology
from tce.constants import LatticeStructure, STRUCTURE_TO_THREE_BODY_LABELS, STRUCTURE_TO_CUTOFF_LISTS


STRUCTURE = LatticeStructure.BCC_WITH_SADDLES


def unique_distances():

    supercell = structures.Supercell(
        lattice_structure=STRUCTURE,
        lattice_parameter=1.0,
        size=(5, 5, 5)
    )

    distance_matrix = topology.get_distance_matrix(
        positions=supercell.positions,
        boxsize=supercell.size,
        max_distance=4.0
    )

    STRUCTURE_TO_CUTOFF_LISTS[STRUCTURE] = np.insert(
        np.unique(distance_matrix.data), 0, 0.0
    )

    print(STRUCTURE_TO_CUTOFF_LISTS[STRUCTURE])


def three_body_labels():

    supercell = structures.Supercell(
        lattice_structure=STRUCTURE,
        lattice_parameter=1.0,
        size=(3, 3, 3)
    )

    max_label = 4

    triplets = [[i, j, k] for i, j, k in product(range(max_label + 1), repeat=3) if max(i, j, k) <= max_label and i <= j <= k]

    labels = np.array(triplets)
    STRUCTURE_TO_THREE_BODY_LABELS[STRUCTURE] = labels

    three_body_tensors = supercell.three_body_tensors(max_order=len(triplets))
    STRUCTURE_TO_THREE_BODY_LABELS[STRUCTURE] = np.array([
        label for label, thr in zip(labels, three_body_tensors) if thr.sum(axis=0).sum(axis=0).mean() > 0
    ])

    print(STRUCTURE_TO_THREE_BODY_LABELS[STRUCTURE])


if __name__ == "__main__":

    unique_distances()
    three_body_labels()

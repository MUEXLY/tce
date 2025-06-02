from enum import Enum, auto
from typing import Dict

import numpy as np


class LatticeStructure(Enum):

    """
    lattice structure enum. will be useful for defining things like cutoff distances and atomic bases
    """

    SC = auto()
    BCC = auto()
    FCC = auto()
    IDEAL_HCP = auto()
    BCC_WITH_SADDLES = auto()


STRUCTURE_TO_ATOMIC_BASIS: Dict[LatticeStructure, np.typing.NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([
        [0.0, 0.0, 0.0]
    ]),
    LatticeStructure.BCC: np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]),
    LatticeStructure.FCC: np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]),
    LatticeStructure.IDEAL_HCP: np.array([
        np.array([0.0, 0.0, 0.0]),
        np.array([2/3, 1/3, 0.5])
    ]),
    LatticeStructure.BCC_WITH_SADDLES: np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75]
    ])
}

STRUCTURE_TO_CUTOFF_LISTS: Dict[LatticeStructure, np.typing.NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([0.0, 1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0]),
    LatticeStructure.BCC: np.array([0.0, 0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0)]),
    LatticeStructure.FCC: np.array([0.0, 0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0)]),
    LatticeStructure.IDEAL_HCP: np.array([0.0, 1.0, np.sqrt(2.0), np.sqrt(8.0 / 3.0), np.sqrt(3.0)])
}

STRUCTURE_TO_THREE_BODY_LABELS: Dict[LatticeStructure, np.typing.NDArray[np.integer]] = {
    LatticeStructure.SC: np.array([
        [0, 0, 1],
        [1, 1, 1]
    ]),
    LatticeStructure.BCC: np.array([
        [0, 0, 1],
        [0, 0, 2],
        [1, 1, 2],
        [2, 2, 2]
    ]),
    LatticeStructure.FCC: np.array([
        [0, 0, 0],
        [0, 0, 1]
    ]),
    LatticeStructure.IDEAL_HCP: np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [1, 1, 2],
        [2, 2, 2],
    ])
}

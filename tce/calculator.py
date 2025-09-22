r"""
this module provides an `ase.calculator.Calculator` class that wraps `tce-lib`
"""


from dataclasses import dataclass, field
from typing import Optional
from itertools import pairwise
from enum import Enum, auto
import logging

from ase.calculators.calculator import Calculator
from ase import Atoms
import numpy as np
from numpy.typing import NDArray

from .training import ClusterExpansion
from .topology import FeatureComputer, topological_feature_vector_factory


LOGGER = logging.getLogger(__name__)


class ASEProperty(Enum):

    r"""
    supported ASE properties to compute
    """

    ENERGY = auto()
    STRESS = auto()


STR_TO_PROPERTY: dict[str, ASEProperty] = {
    "energy": ASEProperty.ENERGY,
    "stress": ASEProperty.STRESS
}
r"""mapping from ase's string to our Enum class for properties"""

INTENSIVE_PROPERTIES: set[ASEProperty] = {
    ASEProperty.STRESS
}
r"""set of intensive properties"""


@dataclass
class TCECalculator(Calculator):

    """
    ASE calculator wrapper for `tce-lib`.
    """

    cluster_expansions: dict[ASEProperty, ClusterExpansion]
    feature_computers: dict[ASEProperty, FeatureComputer] = field(init=False)

    def __post_init__(self):

        for e1, e2 in pairwise(self.cluster_expansions.values()):
            if e1.cluster_basis != e2.cluster_basis:
                raise ValueError(f"cluster bases are different in {self.__class__.__name__}")
            if np.any(e1.type_map != e2.type_map):
                raise ValueError(f"type maps are different in {self.__class__.__name__}")

        self.feature_computers = {}

        expansion_ids = list(self.cluster_expansions.keys())
        extensive_feature_computer = topological_feature_vector_factory(
            basis=self.cluster_expansions[expansion_ids[0]].cluster_basis,
            type_map=self.cluster_expansions[expansion_ids[0]].type_map,
        )

        expansion_ids = list(self.cluster_expansions.keys())
        extensive_feature_computer = topological_feature_vector_factory(
            basis=self.cluster_expansions[expansion_ids[0]].cluster_basis,
            type_map=self.cluster_expansions[expansion_ids[0]].type_map,
        )

        def intensive_feature_computer(atoms: Atoms) -> NDArray:

            return extensive_feature_computer(atoms) / len(atoms)

        for key in expansion_ids:
            if key in INTENSIVE_PROPERTIES:
                self.feature_computers[key] = intensive_feature_computer
                LOGGER.debug(f"intensive feature computer stored for property {key}")
            else:
                self.feature_computers[key] = extensive_feature_computer
                LOGGER.debug(f"extensive feature computer stored for property {key}")

    def get_property(self, name: str, atoms: Optional[Atoms] = None, allow_calculation: bool = True):

        r"""
        compute property from `ase.Atoms` object

        Args:
            name (str): name of property
            atoms (ase.Atoms): atoms object
            allow_calculation (bool): allow calculation
        """

        prop = STR_TO_PROPERTY[name]
        computer = self.feature_computers[prop]

        if atoms is None:
            raise ValueError("please provide Atoms object")

        x = computer(atoms).reshape(1, -1)
        model = self.cluster_expansions[prop].model
        predicted = model.predict(x)

        if isinstance(predicted, np.ndarray):
            predicted = predicted.squeeze()

        return predicted

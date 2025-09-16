r"""
this module provides an `ase.calculator.Calculator` class that wraps `tce-lib`
"""


from dataclasses import dataclass
from typing import Optional
from itertools import pairwise
from enum import Enum, auto
import warnings

from ase.calculators.calculator import Calculator
from ase import Atoms
import numpy as np

from .training import ClusterExpansion
from .topology import FeatureComputer, topological_feature_vector_factory


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
r"""set of intensive properties which need to be rescaled"""


@dataclass
class TCECalculator(Calculator):

    """
    ASE calculator wrapper for `tce-lib`.
    """

    cluster_expansions: dict[ASEProperty, ClusterExpansion]
    feature_computer: Optional[FeatureComputer] = None

    def __init__(
        self,
        cluster_expansions: dict[ASEProperty, ClusterExpansion],
        feature_computer: Optional[FeatureComputer] = None,
        **results
    ):
        r"""basic initialization method. ensures that all cluster expansions have the same bases and type maps"""
        warnings.warn(
            f"{self.__class__.__name__} is not well tested. Use with caution",
            UserWarning
        )
        super().__init__()
        self.cluster_expansions = cluster_expansions

        for e1, e2 in pairwise(cluster_expansions.values()):
            assert e1.cluster_basis == e2.cluster_basis
            assert np.all(e1.type_map == e2.type_map)

        if not feature_computer:
            expansion_ids = list(cluster_expansions.keys())
            self.feature_computer = topological_feature_vector_factory(
                basis=cluster_expansions[expansion_ids[0]].cluster_basis,
                type_map=cluster_expansions[expansion_ids[0]].type_map,
            )
        else:
            self.feature_computer = feature_computer
        self.results = {}

        self.results.update(**results)

    def get_property(self, name: str, atoms: Optional[Atoms] = None, allow_calculation: bool = True):

        r"""
        compute property from `ase.Atoms` object

        Args:
            name (str): name of property
            atoms (ase.Atoms): atoms object
            allow_calculation (bool): allow calculation
        """

        prop = STR_TO_PROPERTY[name]

        x = self.feature_computer(atoms).reshape(1, -1)
        model = self.cluster_expansions[prop].model
        predicted = model.predict(x).squeeze()

        if prop in INTENSIVE_PROPERTIES:
            predicted /= len(atoms)
        return predicted

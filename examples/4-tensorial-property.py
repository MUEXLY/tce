from ase import build, Atoms
from ase.calculators.eam import EAM
import numpy as np
from numpy.typing import NDArray

from tce.constants import LatticeStructure, ClusterBasis
from tce.training import train, get_type_map
from tce.structures import Supercell
from tce.topology import topological_feature_vector_factory


def main():

    lattice_parameter = 3.56
    lattice_structure = LatticeStructure.BCC
    species = np.array(["Cu", "Ni"])
    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        name=species[0],
        crystalstructure=lattice_structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))

    num_configurations = 50
    configurations = []
    for _ in range(num_configurations):
        configuration = atoms.copy()
        x_cu = generator.random()
        configuration.symbols = generator.choice(
            a=species,
            p=[x_cu, 1.0 - x_cu],
            size=len(configuration)
        )

        configuration.calc = EAM(potential="Cu_Ni_Fischer_2018.eam.alloy")
        configurations.append(configuration)

    basis = ClusterBasis(
        lattice_structure=lattice_structure,
        lattice_parameter=lattice_parameter,
        max_adjacency_order=3,
        max_triplet_order=2
    )
    type_map = get_type_map(configurations)
    extensive_feature_computer = topological_feature_vector_factory(basis=basis, type_map=type_map)
    def intensive_feature_computer(atoms_: Atoms) -> NDArray:

        return extensive_feature_computer(atoms_) / len(atoms_)

    cluster_expansion = train(
        configurations,
        basis=basis,
        feature_computer=intensive_feature_computer,
        target_property_computer=lambda atoms_: atoms_.get_stress()
    )

    # predict a larger stress

    larger_system = build.bulk(
        name=species[0],
        crystalstructure=lattice_structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat((10, 10, 10))
    larger_system.symbols = generator.choice(type_map, p=[0.7, 0.3], size=len(larger_system))
    feature_vector = intensive_feature_computer(larger_system)
    print(cluster_expansion.model.predict(feature_vector))


if __name__ == "__main__":

    main()

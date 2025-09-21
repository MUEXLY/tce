from pathlib import Path

import numpy as np
from ase import build
from ase.calculators.eam import EAM
from ase.calculators.singlepoint import SinglePointCalculator
from ase import io
from ase.units import GPa
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tce.training import train, get_type_map
from tce.constants import LatticeStructure, ClusterBasis
from tce.calculator import TCECalculator, ASEProperty
from tce.topology import topological_feature_vector_factory

DATASET_DIR = Path("copper-nickel-many-features")
CLUSTER_BASIS = ClusterBasis(
    lattice_parameter=3.56,
    lattice_structure=LatticeStructure.BCC,
    max_adjacency_order=3,
    max_triplet_order=2
)


def generate_and_save_dataset():

    potential = "Cu_Ni_Fischer_2018.eam.alloy"
    species = np.array(["Cu", "Ni"])
    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        name=species[0],
        crystalstructure=CLUSTER_BASIS.lattice_structure.name.lower(),
        a=CLUSTER_BASIS.lattice_parameter,
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

        configuration.calc = EAM(potential=potential)
        stress = configuration.get_stress()
        energy = configuration.get_potential_energy()
        configuration.calc = SinglePointCalculator(
            atoms=configuration,
            energy=energy,
            stress=stress,
        )
        configurations.append(configuration)

    DATASET_DIR.mkdir(exist_ok=True)
    for i, configuration in enumerate(configurations):

        io.write(DATASET_DIR / f"{i}.xyz", configuration)


def main():

    if not DATASET_DIR.exists():
        generate_and_save_dataset()

    configurations = [io.read(p) for p in DATASET_DIR.glob("*.xyz")]
    train_config, test_config = train_test_split(configurations, test_size=0.2, random_state=0)

    type_map = get_type_map(configurations)
    extensive_feature_computer = topological_feature_vector_factory(
        basis=CLUSTER_BASIS,
        type_map=type_map
    )

    stress_ce = train(
        train_config,
        basis=CLUSTER_BASIS,
        model=RidgeCV(),
        feature_computer=lambda atoms: extensive_feature_computer(atoms) / len(atoms),
        target_property_computer=lambda atoms: atoms.get_stress()
    )

    energy_ce = train(train_config, basis=CLUSTER_BASIS, model=LassoCV())

    actual_stress_xx = []
    predicted_stress_xx = []
    actual_stress_xy = []
    predicted_stress_xy = []
    actual_energies = []
    predicted_energies = []
    for config in test_config:
        actual_stress = config.get_stress() / GPa
        actual_energy = config.get_potential_energy()
        config.calc = TCECalculator(
            cluster_expansions={
                ASEProperty.ENERGY: energy_ce,
                ASEProperty.STRESS: stress_ce
            }
        )
        predicted_stress = config.get_stress() / GPa
        predicted_energy = config.get_potential_energy()

        actual_stress_xx.append(actual_stress[0])
        actual_stress_xy.append(actual_stress[5])
        actual_energies.append(actual_energy)

        predicted_stress_xx.append(predicted_stress[0])
        predicted_stress_xy.append(predicted_stress[5])
        predicted_energies.append(predicted_energy)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    scatter_kwargs = {"facecolor": "green", "edgecolor": "black", "alpha": 0.7, "zorder": 7}
    axs[0].scatter(actual_stress_xx, predicted_stress_xx, **scatter_kwargs)
    axs[1].scatter(actual_stress_xy, predicted_stress_xy, **scatter_kwargs)
    axs[2].scatter(actual_energies, predicted_energies, **scatter_kwargs)

    axs[0].set_title(r"$\sigma_{xx}$ (GPa)")
    axs[1].set_title(r"$\sigma_{xy}$ (GPa)")
    axs[2].set_title(r"$E_\text{tot}$ (eV)")

    for ax in axs:
        ax.plot(ax.get_xlim(), ax.get_xlim(), color="black", linestyle="--", label="model = truth")
        ax.grid()

    axs[0].legend()
    fig.tight_layout()
    fig.supylabel("predicted")
    fig.supxlabel("actual")
    fig.tight_layout()
    fig.savefig("calculator-interface.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()

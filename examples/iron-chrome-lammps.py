from typing import Callable

from ase import build
from ase.calculators.calculator import Calculator
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
from scipy.spatial import KDTree
from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    lattice_structure: LatticeStructure = LatticeStructure.BCC
    lattice_parameter: float = 2.87
    size: tuple[int, int, int] = (4, 4, 4)
    two_body_order: int = 2
    three_body_order: int = 1
    type_map: dict[int, str] = {0: "Fe", 1: "Cr"}

    # we need to define a constructor here, not a Calculator object, because each configuration
    # needs its own Calculator instance
    def calculator_constructor() -> Callable[[], Calculator]:
        return LAMMPSlib(
            lmpcmds=[
                "pair_style hybrid/overlay eam/alloy eam/fs",
                "pair_coeff * * eam/alloy Fe_Cr_Eich_2015_TBM_lammps.eam.alloy Fe Cr",
                "pair_coeff * * eam/fs Fe_Cr_Eich_2015_TBM_lammps.eam.fs Fe Cr",
                "neighbor 2.0 bin",
                "neigh_modify delay 10",
                "fix 1 all box/relax iso 0.0 vmax 0.001",
                "min_style cg",
                "minimize 0.0 1.0e-9 100000 10000000"
            ]
        )
    num_samples: int = 50
    rng: np.random.Generator = np.random.default_rng(seed=0)

    species = list(type_map.values())
    inverse_type_map = {v: k for k, v in type_map.items()}

    atoms = build.bulk(
        "Fe",
        crystalstructure=lattice_structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat(size)
    adjacency_tensors = get_adjacency_tensors(
        tree=KDTree(data=atoms.positions, boxsize=np.diag(atoms.cell)),
        cutoffs=lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[lattice_structure][:two_body_order]
    )
    three_body_tensors = get_three_body_tensors(
        lattice_structure=lattice_structure,
        adjacency_tensors=adjacency_tensors,
        max_three_body_order=three_body_order
    )

    X = np.zeros((
        num_samples,
        two_body_order * len(species) ** 2 + three_body_order * len(species) ** 3
    ))
    y = np.zeros(num_samples)

    for i in range(num_samples):

        print(f"{(i + 1) / num_samples:.2%}", end="\r")

        new_configuration = atoms.copy()
        chromium_fraction = rng.uniform(low=0.05, high=0.95)
        new_configuration.symbols = rng.choice(
            ["Fe", "Cr"],
            p=[1.0 - chromium_fraction, chromium_fraction],
            size=len(atoms)
        )
        new_configuration.calc = calculator_constructor()

        # get feature vector from the Atoms object
        state_matrix = np.zeros((len(new_configuration), len(species)))
        for site, symbol in enumerate(new_configuration.symbols):
            state_matrix[site, inverse_type_map[symbol]] = 1.0

        X[i, :] = get_feature_vector(
            adjacency_tensors=adjacency_tensors,
            three_body_tensors=three_body_tensors,
            state_matrix=state_matrix
        )

        # compute energy
        y[i] = new_configuration.get_potential_energy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    eci = np.linalg.pinv(X_train) @ y_train
    training_residuals = X_train @ eci - y_train
    testing_residuals = X_test @ eci - y_test

    plt.hist(
        training_residuals / len(atoms),
        zorder=7,
        label="training",
        linewidth=1.0,
        edgecolor="black",
        color="skyblue"
    )
    plt.hist(
        testing_residuals / len(atoms),
        zorder=8,
        label="testing",
        linewidth=1.0,
        edgecolor="black",
        color="sandybrown"
    )
    plt.legend()
    plt.xlabel("prediction error (eV / atom)")
    plt.ylabel("counts")
    plt.grid()
    plt.tight_layout()
    plt.savefig("cross-val.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":
    main()

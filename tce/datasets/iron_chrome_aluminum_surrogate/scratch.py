from pathlib import Path

from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np


def main():

    paths = list(Path("/home/jwjeffr/steel-entry/configurations").iterdir())

    for i, p in enumerate(paths):
        atoms = io.read(p / "configuration.xyz")
        energy = np.loadtxt(p / "energy.txt")

        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        io.write(f"configuration_{i:.0f}.xyz", atoms)


if __name__ == "__main__":

    main()

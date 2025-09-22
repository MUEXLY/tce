r"""
This module defines some convenience wrappers for running a Monte Carlo simulation from a fitted cluster expansion
model.
"""


from typing import Optional, Callable, TypeAlias
import logging
from functools import wraps

import numpy as np
from numpy.typing import NDArray
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from tce.structures import Supercell
from tce.training import ClusterExpansion


LOGGER = logging.getLogger(__name__)
"""@private"""


MCStep: TypeAlias = Callable[[NDArray[np.floating]], NDArray[np.floating]]
r"""
Type alias defining what a step in a monte carlo simulation looks like. In general, a step should look like a function 
that takes in a state matrix $\mathbf{X}$, and returns a new one.
"""


def two_particle_swap_factory(generator: np.random.Generator) -> MCStep:

    r"""
    Factory to create a sensible MC step, which is to swap two particles.

    Args:
        generator (np.random.Generator): Random number generator to be used to sample a new MC step
    """

    @wraps(two_particle_swap_factory)
    def wrapper(state_matrix: NDArray) -> NDArray[np.floating]:

        new_state_matrix = state_matrix.copy()
        i, j = generator.integers(len(state_matrix), size=2)
        new_state_matrix[i], new_state_matrix[j] = state_matrix[j], state_matrix[i]
        return new_state_matrix

    return wrapper


EnergyModifier: TypeAlias = Callable[[NDArray[np.floating], NDArray[np.floating]], float]
r"""
Type alias defining what an energy modifier should look like. In general, a modifier should look like a function that 
takes in two state matrices $\mathbf{X}$ and $\mathbf{X}'$ and returns the term to be added to the energy difference 
$\Delta E$. For example, if you want to simulate a grand canonical ensemble, the Metropolis acceptance criterion is:

$$ \exp\left(-\beta\left(\Delta E - \sum_\alpha \mu_\alpha \Delta N_\alpha\right)\right) = \exp\left(-\beta\left(\Delta E - \boldsymbol{\mu}\cdot\Delta\mathbf{N}\right)\right) > u $$

for a random number $u$ from $\text{Uniform}(0, 1)$. You can implement this strategy by defining an energy modifier:

```py
from typing import Callable
from functools import wraps

import numpy as np

def energy_modifier_factory(
    chemical_potentials: NDArray[np.floating]
) -> Callable[[NDArray[np.floating], NDArray[np.floating]], float]:

    @wraps(energy_modifier_factory)
    def wrapper(
        state_matrix: NDArray[np.floating],
        new_state_matrix: NDArray[np.floating]
    ) -> float:
        change_in_num_types = new_state_matrix.sum(axis=0) - state_matrix.sum(axis=0)
        return -chemical_potentials @ change_in_num_types

    return wrapper
```

You can see a concrete example of the above energy modifier [here](https://github.com/MUEXLY/tce-lib#training-monte-carlo).
"""


def null_energy_modifier(
    state_matrix: NDArray[np.floating],
    new_state_matrix: NDArray[np.floating]
) -> float:

    r"""
    Default energy modifier, which does nothing to the total energy
    """

    return 0.0


def monte_carlo(
    initial_configuration: Atoms,
    cluster_expansion: ClusterExpansion,
    num_steps: int,
    beta: float,
    save_every: int = 1,
    generator: Optional[np.random.Generator] = None,
    mc_step: Optional[MCStep] = None,
    energy_modifier: Optional[EnergyModifier] = None,
    callback: Optional[Callable[[int, int], None]] = None
) -> list[Atoms]:

    r"""
    monte Carlo simulation from on a lattice defined by a Supercell

    Args:
        initial_configuration (Atoms):
            initial atomic configuration to perform MC on
        cluster_expansion (ClusterExpansion):
            Container defining training data. see `tce.training.ClusterExpansion` for more info. this will usually
            be created by `tce.training.train`.
        num_steps (int):
            Number of Monte Carlo steps to perform
        beta (float):
            Thermodynamic $\beta$, defined by $\beta = 1/(k_BT)$, where $k_B$ is the Boltzmann constant and $T$ is
            absolute temperature. ensure that $k_B$ is in proper units such that $\beta$ is in appropriate units. for
            example, if the training data had energy units of eV, then $k_B$ should be defined in units of eV/K.
        save_every (int):
            How many steps to perform before saving the MC frame. this is similar to LAMMPS's `dump_every` argument
            in the `dump` command
        generator (Optional[np.random.Generator]):
            Generator instance drawing random numbers. if not specified, set to `np.random.default_rng(seed=0)`
        mc_step (Optional[MCStep]):
            Monte Carlo simulation step. if not specified, assume that the user wants to swap 2 particles per step.
        energy_modifier (Optional[Callable[[NDArray[np.floating], NDArray[np.floating]], float]]):
            Energy modifier when performing MC run. each acceptance rule looks very similar for different ensembles,
            i.e., if $\exp(-\beta \Delta H) > u$, where $u$ is a random number drawn from $ [0, 1] $, then accept the swap.
            $\Delta H$, generally, is of the form:
            $$ \Delta H = \Delta E + f(\mathbf{X}, \mathbf{X}') $$
            For example, for the [grand canonical ensemble](https://en.wikipedia.org/wiki/Grand_canonical_ensemble):
            $$ f(\mathbf{X}, \mathbf{X}') = -\sum_\alpha \mu_\alpha\Delta N_\alpha $$
            where $\mu_\alpha$ is the chemical potential of type $\alpha$ and $\Delta N_\alpha$ is change in the number
            of $\alpha$ atoms upon swapping. if unspecified, then energy is not modified throughout the run, which
            samples the [canonical ensemble](https://en.wikipedia.org/wiki/Canonical_ensemble).
        callback (Optional[Callable[[int, int], None]]):
            Optional callback function that will be called after each step. will take in the current step and the
            number of overall steps. if not specified, defaults to a call to LOGGER.info

    """

    if not generator:
        generator = np.random.default_rng(seed=0)
    if not mc_step:
        mc_step = two_particle_swap_factory(generator=generator)
    if not energy_modifier:
        energy_modifier = null_energy_modifier
    if not callback:
        def callback(step_: int, num_steps_: int):
            LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")

    num_types = len(cluster_expansion.type_map)

    lattice_structure = cluster_expansion.cluster_basis.lattice_structure
    lattice_parameter = cluster_expansion.cluster_basis.lattice_parameter

    lengths = initial_configuration.get_cell().lengths()

    supercell = Supercell(
        lattice_structure=lattice_structure,
        lattice_parameter=lattice_parameter,
        size=tuple((lengths // lattice_parameter).astype(int))
    )
    LOGGER.debug(f"initialized supercell with size {supercell.size} for MC run")

    inverse_type_map = {v: k for k, v in enumerate(cluster_expansion.type_map)}
    initial_types = np.fromiter((
        inverse_type_map[symbol] for symbol in initial_configuration.get_chemical_symbols()
    ), dtype=int)

    state_matrix = np.zeros((supercell.num_sites, num_types), dtype=int)
    state_matrix[np.arange(supercell.num_sites), initial_types] = 1

    trajectory = []
    energy = cluster_expansion.model.predict(
        supercell.feature_vector(
            state_matrix=state_matrix,
            max_adjacency_order=cluster_expansion.cluster_basis.max_adjacency_order,
            max_triplet_order=cluster_expansion.cluster_basis.max_triplet_order
        )
    )
    LOGGER.debug(f"initial energy is {energy}")
    for step in range(num_steps):

        callback(step, num_steps)

        if not step % save_every:
            _, types = np.where(state_matrix)
            atoms = initial_configuration.copy()
            atoms.set_chemical_symbols(symbols=cluster_expansion.type_map[types])
            atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy)
            trajectory.append(atoms)
            LOGGER.info(f"saved configuration at step {step:.0f}/{num_steps:.0f}")

        new_state_matrix = mc_step(state_matrix)
        feature_diff = supercell.clever_feature_diff(
            state_matrix, new_state_matrix,
            max_adjacency_order=cluster_expansion.cluster_basis.max_adjacency_order,
            max_triplet_order=cluster_expansion.cluster_basis.max_triplet_order
        )
        energy_diff = cluster_expansion.model.predict(feature_diff)
        if not isinstance(energy_diff, float):
            raise ValueError(
                "cluster_expansion.model.predict did not return a float. "
                "Are you sure this model was trained on energies?"
            )
        modified_energy = energy_diff + energy_modifier(state_matrix, new_state_matrix)
        if np.exp(-beta * modified_energy) > 1.0 - generator.random():
            LOGGER.debug(f"move accepted with energy difference {energy_diff:.3f}")
            state_matrix = new_state_matrix
            energy += energy_diff

    return trajectory
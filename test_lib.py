from typing import Callable
import re
from tempfile import TemporaryDirectory
from pathlib import Path
import pickle

import pytest
import numpy as np
from ase import build
from ase.calculators.singlepoint import SinglePointCalculator
import sparse

import tce
from tce.constants import LatticeStructure, get_three_body_labels, _STRUCTURE_TO_THREE_BODY_LABELS
from tce.structures import Supercell
from tce.training import (
    ClusterBasis,
    INCOMPATIBLE_GEOMETRY_MESSAGE,
    NO_POTENTIAL_ENERGY_MESSAGE,
    NON_CUBIC_CELL_MESSAGE,
    LARGE_SYSTEM_MESSAGE,
    LimitingRidge,
    ClusterExpansion
)
from tce.topology import symmetrize
from tce.datasets import available_datasets, Dataset


@pytest.fixture
def get_supercell() -> Callable[[], Supercell]:

    def supercell(lattice_structure: LatticeStructure) -> Supercell:

        size = None
        if lattice_structure == LatticeStructure.SC:
            size = (5, 5, 5)
        if lattice_structure == LatticeStructure.BCC:
            size = (4, 4, 4)
        if lattice_structure == LatticeStructure.FCC:
            size = (3, 3, 3)
        if not size:
            raise ValueError("lattice_structure must be SC, BCC, or FCC")

        return Supercell(lattice_structure, lattice_parameter=1.0, size=size)

    return supercell


@pytest.mark.parametrize(
    "lattice_structure, num_expected_neighbors",
    [
        (LatticeStructure.SC, 6),
        (LatticeStructure.BCC, 8),
        (LatticeStructure.FCC, 12)
    ]
)
def test_num_neighbors(lattice_structure: LatticeStructure, num_expected_neighbors: int, get_supercell):

    supercell = get_supercell(lattice_structure)

    for adj in supercell.adjacency_tensors(max_order=1):
        assert adj.sum(axis=0).todense().mean() == num_expected_neighbors


@pytest.mark.parametrize("lattice_structure", [LatticeStructure.SC, LatticeStructure.BCC, LatticeStructure.FCC])
def test_feature_vector_shortcut(lattice_structure: LatticeStructure, get_supercell):

    rng = np.random.default_rng(seed=0)
    num_types = 3

    supercell = get_supercell(lattice_structure)

    types = rng.integers(num_types, size=supercell.num_sites)

    state_matrix = np.zeros((supercell.num_sites, num_types), dtype=int)
    state_matrix[np.arange(supercell.num_sites), types] = 1

    new_state_matrix = state_matrix.copy()
    first_site, second_site = rng.integers(supercell.num_sites, size=2)
    while types[first_site] == types[second_site]:
        first_site, second_site = rng.integers(supercell.num_sites, size=2)
    new_state_matrix[first_site, :] = state_matrix[second_site, :]
    new_state_matrix[second_site, :] = state_matrix[first_site, :]

    clever_diff = supercell.clever_feature_diff(
        state_matrix, new_state_matrix,
        max_adjacency_order=2, max_triplet_order=2
    )

    feature_vector = supercell.feature_vector(
        state_matrix,
        max_adjacency_order=2,
        max_triplet_order=2
    )
    new_feature_vector = supercell.feature_vector(
        new_state_matrix,
        max_adjacency_order=2,
        max_triplet_order=2
    )
    naive_diff = new_feature_vector - feature_vector

    assert np.all(naive_diff == clever_diff)


def test_noncubic_cell_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=False).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=False).repeat((3, 3, 3))
    ]
    for configuration in configurations:
        configuration.calc = SinglePointCalculator(configuration, energy=-1.0)

    with pytest.raises(ValueError, match=NON_CUBIC_CELL_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_inconsistent_geometry_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        build.bulk("Fe", crystalstructure="fcc", a=2.7, cubic=True).repeat((3, 3, 3))
    ]

    for configuration in configurations:
        configuration.calc = SinglePointCalculator(configuration, energy=-1.0)

    with pytest.raises(ValueError, match=INCOMPATIBLE_GEOMETRY_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_no_energy_computation_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=True).repeat((3, 3, 3))
    ]

    with pytest.raises(ValueError, match=NO_POTENTIAL_ENERGY_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_large_system_in_training(monkeypatch):

    with monkeypatch.context() as m:

        m.setattr("tce.training.LARGE_SYSTEM_THRESHOLD", 10)

        configurations = [
            build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        ]
        configurations[0].calc = SinglePointCalculator(configurations[0], energy=-1.0)

        with pytest.warns(UserWarning, match=re.escape(LARGE_SYSTEM_MESSAGE)):
            _ = tce.training.train(
                configurations=configurations,
                basis=ClusterBasis(
                    lattice_structure=LatticeStructure.BCC,
                    lattice_parameter=2.7,
                    max_adjacency_order=3,
                    max_triplet_order=1
                )
            )


@pytest.mark.parametrize("dataset_str", available_datasets())
def test_can_load_dataset(dataset_str):

    dataset = Dataset.from_dir(dataset_str)
    print(dataset)


def test_symmetrization_no_axes():

    x = sparse.COO.from_numpy(np.array([
        [1, 1],
        [0, 1]
    ]))
    x_symmetrized = sparse.COO.from_numpy([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    assert np.all(symmetrize(x).todense() == x_symmetrized.todense())


def test_limiting_ridge_throws_error():

    lr = LimitingRidge()
    with pytest.raises(ValueError):
        lr.predict(np.zeros(2))


def test_limiting_ridge_fit():

    X = np.array([1, 2, 3]).reshape((-1, 1))
    y = np.array([2, 4, 6])
    lr = LimitingRidge().fit(X, y)
    assert np.all(y == lr.predict(X))


def test_can_write_and_read_model():

    X = np.array([1, 2, 3]).reshape((-1, 1))
    y = np.array([2, 4, 6])
    lr = LimitingRidge().fit(X, y)

    ce = ClusterExpansion(
        model=lr,
        cluster_basis=ClusterBasis(
            lattice_structure=LatticeStructure.BCC,
            lattice_parameter=2.7,
            max_adjacency_order=3,
            max_triplet_order=1
        ),
        type_map=np.array(["Fe", "Cr"])
    )

    with TemporaryDirectory() as directory:
        temp_path = Path(directory) / "model.pkl"
        ce.save(temp_path)
        ce_new = ClusterExpansion.load(temp_path)

    assert ce_new.cluster_basis == ce.cluster_basis
    assert np.all(ce_new.model.coef_ == ce.model.coef_)


def test_bad_pkl_object():

    with TemporaryDirectory() as directory:
        temp_path = Path(directory) / "obj.pkl"
        with temp_path.open("wb") as f:
            pickle.dump(object(), f)
        with pytest.raises(ValueError):
            _ = ClusterExpansion.load(temp_path)


@pytest.mark.parametrize("lattice_structure", LatticeStructure)
def test_computed_labels_equal_cached_labels(lattice_structure: LatticeStructure):

    cached = _STRUCTURE_TO_THREE_BODY_LABELS[lattice_structure]
    loaded = get_three_body_labels(lattice_structure)

    assert np.all(cached == loaded)

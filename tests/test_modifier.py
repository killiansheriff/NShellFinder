import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
import numpy as np
import pytest
from ovito.data import DataCollection, NearestNeighborFinder
from ovito.io import import_file

from NshellFinder import NshellFinder


@pytest.fixture
def import_data():
    pipe = import_file("examples/fcc.dump")
    yield pipe.compute()


def test_fcc(import_data: DataCollection):
    data = import_data

    data.apply(NshellFinder(crystal_structure="fcc", cutoff=18.2))
    neighbor_indices_per_shell = data.attributes["Neighbor indices per shell"]

    first_nn = neighbor_indices_per_shell[0]
    second_nn = neighbor_indices_per_shell[1]

    finder = NearestNeighborFinder(18, data)
    nn_idx, _ = finder.find_all()

    first_nn_true = nn_idx[:, :12]
    second_nn_true = nn_idx[:, 12:]

    assert np.all(np.sort(first_nn, axis=1) == np.sort(first_nn_true, axis=1))
    assert np.all(np.sort(second_nn, axis=1) == np.sort(second_nn_true, axis=1))

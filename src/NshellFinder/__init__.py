import itertools
import os

import numpy as np
from ovito.data import (
    CutoffNeighborFinder,
    DataCollection,
    DataTable,
    ElementType,
    NearestNeighborFinder,
)
from ovito.pipeline import ModifierInterface
from traits.api import Bool, Float, Int, List, String


class NshellFinder(ModifierInterface):
    """ """

    cutoff = Float(default_value=18.2)
    crystal_structure = String(value="fcc")

    def get_cumsum_atom_in_shell(self):
        file_path = os.path.join(
            os.path.dirname(__file__),
            f"shell_informations/{self.crystal_structure}_shell_count.txt",
        )
        number_of_atoms_in_shells = np.loadtxt(file_path)
        cum_sum_atom_in_shell = np.zeros(len(number_of_atoms_in_shells) + 1)
        cum_sum_atom_in_shell[1:] = np.cumsum(number_of_atoms_in_shells)

        return cum_sum_atom_in_shell

    def get_nshell_neighbor_idx(self, nshell, cum_sum_atom_in_shell, neigh_idx):
        lb_count = int(cum_sum_atom_in_shell[nshell])
        up_count = int(cum_sum_atom_in_shell[nshell + 1])

        nn_ind_in_shell = neigh_idx[
            :,
            lb_count:up_count,
        ]
        return nn_ind_in_shell

    def modify(self, data: DataCollection, frame: int, **kwargs):
        finder = CutoffNeighborFinder(self.cutoff, data)
        neigh_idx, _ = finder.find_all(sort_by="distance")

        N = data.particles.count

        starts = np.searchsorted(neigh_idx[:, 0], np.arange(N), side="left")
        ends = np.searchsorted(neigh_idx[:, 0], np.arange(N), side="right")

        neigh_idx = np.array(
            [neigh_idx[starts[i] : ends[i]][:, 1] for i in range(N)]
        )  # (Natoms, Nneigh within cutoff)

        cum_sum_atom_in_shell = self.get_cumsum_atom_in_shell()

        max_shell_given_cutoff = np.argmax(np.where(cum_sum_atom_in_shell <= len(neigh_idx[0])))

        neighbor_indices_per_shell = []

        for nshell in range(max_shell_given_cutoff):
            nshell_nn_indices = self.get_nshell_neighbor_idx(
                nshell, cum_sum_atom_in_shell, neigh_idx
            )
            neighbor_indices_per_shell.append(nshell_nn_indices)

        data.attributes["Neighbor indices per shell"] = neighbor_indices_per_shell


if __name__ == "__main__":
    from ovito.io import import_file

    dump = "/home/ksheriff/PAPERS/first_paper/03_mtp/data/dumps/dumps_mtp_mc/ordered_relaxation_20_1_300K.dump"
    pipeline = import_file(dump)
    mod = MshellFinder(crystal_structure="fcc", cutoff=18.2)
    pipeline.modifiers.append(mod)
    data = pipeline.compute()

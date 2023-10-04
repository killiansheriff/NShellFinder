from ovito.io import import_file

from NshellFinder import NshellFinder

pipeline = import_file("fcc.dump")
mod = NshellFinder(crystal_structure="fcc", cutoff=18.2)
pipeline.modifiers.append(mod)
data = pipeline.compute()

neighbor_indices_per_shell = data.attributes["Neighbor indices per shell"]

first_nn = neighbor_indices_per_shell[0]
second_nn = neighbor_indices_per_shell[1]

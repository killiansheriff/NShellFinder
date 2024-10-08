# NShellFinder
![PyPI Version](https://img.shields.io/pypi/v/nshellfinder.svg) ![PyPI Downloads](https://static.pepy.tech/badge/nshellfinder) ![tests](https://github.com/killiansheriff/NShellFinder/actions/workflows/python-tests.yml/badge.svg)

Ovito Python modifier to find the n-th coordination shell neighbors. 

## Utilisation 
Here is an example on how to find the indices of nearest neighbors -- cluster by shells up to certain cutoff -- for the fcc crystal structure. The scrip can be found in the ``examples/`` folder.  

```python
from ovito.io import import_file
from NshellFinder import NshellFinder

pipeline = import_file("fcc.dump")
mod = NshellFinder(crystal_structure="fcc", cutoff=18.2)
pipeline.modifiers.append(mod)
data = pipeline.compute()

neighbor_indices_per_shell = data.attributes["Neighbor indices per shell"]
# (number of nearest neighbor shells up to cutoff, number of atoms, number of nearest neighbors in the shell)

first_nn = neighbor_indices_per_shell[0] # (number of atoms, 12 first nearest neigbors)
second_nn = neighbor_indices_per_shell[1] # (number of atoms, 6 second nearest neighbors)
```


## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user NshellFinder
```

For *OVITO PRO* built-in Python interpreter, please use:
```bash
ovitos -m pip install --user NshellFinder
```

If you want to install the lastest git commit, please replace ``NshellFinder`` by ``git+https://github.com/killiansheriff/NshellFinder.git``.

## Contact
If any questions, feel free to contact me (ksheriff at mit dot edu).

## References & Citing 
If you use this repository in your work, please cite:

```
@article{sheriffquantifying2024,
	title = {Quantifying chemical short-range order in metallic alloys},
	doi = {10.1073/pnas.2322962121},
	journaltitle = {Proceedings of the National Academy of Sciences},
	author = {Sheriff, Killian and Cao, Yifan and Smidt, Tess and Freitas, Rodrigo},
	date = {2024-06-18},
}
```

and 

```
@article{sheriff2024chemicalmotif,
  title = {Chemical-motif characterization of short-range order with E(3)-equivariant graph neural networks},
  DOI = {10.1038/s41524-024-01393-5},
  journal = {npj Computational Materials},
  author = {Sheriff,  Killian and Cao,  Yifan and Freitas,  Rodrigo},
  year = {2024},
  month = sep,
}
```

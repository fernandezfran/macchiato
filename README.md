# macchiato

[![macchiatos CI](https://github.com/fernandezfran/macchiato/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/macchiato/actions/workflows/CI.yml)

Data-driven nearest-neighbors models to predict physical experiments in
silicon-based lithium-ion battery anodes.

In order to improve certain aspects of the materials, it is necessary to have 
an in-depth understanding of the physics of the processes involved. In the 
intrinsic behavior of silicon anodes for lithium-ion batteries, two 
states of Li-ions are always observed in different experiments that have not yet 
been elucidated. These models attempt to better interpret these peculiarities.


## Requirements

You need Python 3.8+ to run macchiato.


## Installation

You can install the most recent stable release of macchiato with 
[pip](https://pip.pypa.io/en/latest/)

```
pip install macchiato
```

or you can install the development version directly from this
[repo](https://github.com/fernandezfran/macchiato)
```
pip install git+https://github.com/fernandezfran/macchiato
```


## Citation

> F. Fernandez, M. Otero, M. B. Oviedo, D. E. Barraco, S. A. Paz, and E. P. M. Leiva.
> (2023). Prediction of NMR, X-ray and M\"ossbauer experimental results for amorphous 
> Li-Si alloys using a novel DFTB model. _preprint arXiv:2305.11006_.

BibTeX entry:

```bibtex
@misc{fernandez2023prediction,
      title={Prediction of NMR, X-ray and M\"ossbauer experimental results for amorphous Li-Si alloys using a novel DFTB model},
      author={Francisco Fernandez and Manuel Otero and Ma. Belén Oviedo and Daniel E. Barraco and S. Alexis Paz and Ezequiel P. M. Leiva},
      year={2023},
      eprint={2305.11006},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

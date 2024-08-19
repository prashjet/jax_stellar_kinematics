# Stellar Kinematic modelling with JAX

This repository contains tools for spectral modelling to extract the Line-of-Sight-Velocity Distribution (LOSVD) from integrated-light galaxy spectra. Existing tools for this task (e.g. [ppxf](https://pypi.org/project/ppxf/)) are limited to modelling single spectra. A [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)-based implementation will allow us to scale-up to model entire Integral Field Unit (IFU) datacubes, which are spatially-resolved spectroscopic datasets that consist of many thousand spectra. Simultaneously modelling the spectral *and* spatial information content of the datacube is known as 3D modelling. This technique will allow us to leverage physically-motivated prior information in joint position and velocity space, which will lead to improved detection and characterisation of galactic structures. A proof-of-principle of this idea is shown in [Hinterer, Hubmer, Jethwa et al. 2022](https://arxiv.org/abs/2206.03925). Aditionally, JAX and its ecosystem of inference software will improve uncertainty quantification and prior modelling compared to existing tools.

This repository hosts some initial implementations of LOSVD-recovery tools implemented in JAX and [`numpyro`](https://num.pyro.ai/en/latest/index.html#introductory-tutorials). The goal is to expand this into a complete software package, building in spatial-modelling capability with Gaussian Processes using [`tinygp`](https://tinygp.readthedocs.io/en/stable/).

Repository contents:
- `code`
    -  `fit_spectrum.py`: main functions
    -  `gauss_hermite.py`: implementation of the [Gauss Hermite](https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V/abstract) LOSVD model
    - `sanders_evans.py`: implementation of the LOSVD model from [Sanders & Evans 2020](https://arxiv.org/abs/2009.07858)
    - `example_output.ipynb`: notebook illustrating some code output
- `data`:
    -  `spectrum.out`: spectrum from galaxy NGC1023, taken from the [ATLAS 3D](https://www-astro.physics.ox.ac.uk/atlas3d/) survey
    -  `emission_line_mask.npy`
- `output`:
    -  `mcmc_result.out`: the output of `code/fit_spectrum.py`
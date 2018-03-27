# Zeros of the spectrogram of white noise
[![Build Status]()


This companion Python project contains numerical experiments associated to the paper

> Bardenet, R., Flamant, J., & Chainais, P. (2017). On the zeros of the  spectrogram of white noise. arXiv preprint arXiv:1708.00082.

## Project description

This Python package provides several utility functions to reproduce the figures presented in the paper.
Three Jupyter notebooks located in the `notebooks/` folder are available for this purpose:

- Figure 1 and 2: `Spectrograms of real and complex WGN.ipynb`
- Figure 5 and 6: `Rank envelope tests.ipynb`
- Figure 6 `Reconstruction.ipynb`

## Download

### Dependencies

Currently, this code requires the following packages to be installed:
- `matplotlib`
- `numpy`
- `scipy`
- `seaborn`
- `cmocean`

Spatial statistics functions rely on the use of the spatstat toolbox developed in `R`. Thus we also require
- `rpy2` version > 2.8.5
- `R` version > 3.3.2 (https://www.r-project.org/)
- `spatstat` version > 1.48-0 library (http://spatstat.org/)

### Install from sources

Clone this repository

```bash
git clone https://github.com/jflamant/2018-zeros-spectrogram-white-noise.git
cd 2018-zeros-spectrogram-white-noise
```

And execute `setup.py`

```bash
pip install .
```

Of course, if you're in development mode and you want to install also dev packages, documentation and/or tests, you can do as follows:

```bash
pip install -e .
```

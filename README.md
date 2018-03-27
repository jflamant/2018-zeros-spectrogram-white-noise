# Zeros of the spectrogram of white noise
[![Build Status](https://travis-ci.org/CRIStAL-Sigma/mock_tex_paper.svg?branch=master)](https://travis-ci.org/CRIStAL-Sigma/mock_tex_paper)


This companion Python project contains numerical experiments associated to the paper

>Bardenet, R., Flamant, J., & Chainais, P. (2017). On the zeros of the  spectrogram of white noise. arXiv preprint arXiv:1708.00082.

## Project description

This is a library implementing time-travel equations based on incomplete scarce agile data. This project is used to build flux capacitors.  More information about this project can be found in my third book.

## Download

### Dependencies

Currently, this code requires the following packages to be installed:

Also to compute spatial statistics functions we need
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

## Usage examples

You can import [my package] by doing

```python
import my_package as mp
```

The main functions included in this package are `x()` and `z()`. `x` receives A as argument and does X. Here is example of its usage:

```python
x(`hello`, `world`, 27, [])
```

A more detailed documentation can be found in [link].

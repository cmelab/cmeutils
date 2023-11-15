# CME Lab Utilies
[![build](https://github.com/cmelab/cmeutils/actions/workflows/build.yml/badge.svg)](https://github.com/cmelab/cmeutils/actions/workflows/build.yml)
[![pytest](https://github.com/cmelab/cmeutils/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/cmeutils/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/cmeutils/branch/master/graph/badge.svg?token=WPJGJX23I7)](https://codecov.io/gh/cmelab/cmeutils)


Helpful functions used by the [CME Lab](https://www.boisestate.edu/coen-cmelab/).

### Installation
Installation of CME Lab Utilities requires a conda package manager. We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Installing from conda-forge:
1. Create a new conda environment and install `cmeutils`
	````
	conda create -n cmeutils -c conda-forge cmeutils
	conda activate cmeutils
	```

## Installing from source (for development):
1. Clone this repository:
    ```
    git clone git@github.com:cmelab/cmeutils.git
    cd cmeutils
    ```
2. Set up and activate environment:
    ```
    conda env create -f environment.yml
    conda activate cmeutils
    ```
3. Install from source with pip:
    ```
    pip install -e .
    ```

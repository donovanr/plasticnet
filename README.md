# Plastic Net
[![Build Status](https://www.travis-ci.com/donovanr/plasticnet.svg?branch=master)](https://www.travis-ci.com/donovanr/plasticnet)
[![Documentation Status](https://readthedocs.org/projects/plasticnet/badge/?version=latest)](https://plasticnet.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/donovanr/plasticnet/branch/master/graph/badge.svg)](https://codecov.io/gh/donovanr/plasticnet)

## Organization

The main contents of the repository are organized as:

```
.
├── docs
│   └── source
└── plasticnet
    ├── classes
    ├── solvers
    ├── tests
    └── utils
```

## Installation

We use [Conda](https://conda.io/) to manage the development environment, with dependencies listed in [.environment.yml](../master/.environment.yml).
If you don't have `conda` installed and set up for Python 3.x, install it using the instructions on their site, and then install this package with

```
git clone https://github.com/donovanr/plasticnet.git
cd plasticnet
conda env create -f .environment.yml
conda activate plasticnet
pip install -e .
```

## Development
We use:
- [GitHub](https://github.com) for code hosting and issue tracking
- [Travis CI](https://travis-ci.org/) for testing
- [Codecov](https://codecov.io/) for code coverage reporting
- [Read the Docs](https://readthedocs.org/) for auto-generating and publishing the documentation
- [pre-commit hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) for formatting and linting

### Travis
The Travis configuration is in [.travis.yml](../master/.travis.yml), and doesn't have anything fancy set up.

Tests are run automatically when commits are pushed to GitHub. To run the tests locally, issue `pytest` in the main project directory.

### Codecov
Configuration for Codecov is in [.codecov.yml](../master/.codecov.yml).
Travis is set up to run `pytest-cov`, and the resulting report is pushed to Codecov.
The configuration for `pytest-cov` is in [.coveragerc](../master/.coveragerc)

### Read the Docs
Documentation config for [Sphinx](http://www.sphinx-doc.org/) + [autodoc](http://www.sphinx-doc.org/en/master/usage/quickstart.html#autodoc) lives in [docs/source/conf.py](../master/docs/source/conf.py).
Build configs live in [.readthedocs.yml](../master/.readthedocs.yml).
In the Read the Docs admin pane, you should choose the `CPython 3.x` interpreter.

Docs are generated automatically when commits are pushed to GitHub.
To generate the docs locally, from the main project directory, use

```
sphinx-build -b html docs/source docs/build
```

### Pre-Commit Hooks
Pre-commit hooks are configured in [.pre-commit-config.yaml](../master/.pre-commit-config.yaml), and need to be set up (once) with `pre-commit install`.

The pre-commit hooks we use are:
- [Black](https://black.readthedocs.io/en/stable/) for formatting, with the default settings
- [Flake8](http://flake8.pycqa.org/en/latest/) for linting, with configurations in [.flake8](../master/.flake8)

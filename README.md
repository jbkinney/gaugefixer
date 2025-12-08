# GaugeFixer

GaugeFixer is a lightweight Python package that allows interpretation of the parameters of linear models for sequence-function relationships by removing unconstrained degrees of freedom in their values, an operation known as "fixing the gauge".

## Installation

Create a new python environment with conda and activate it

```bash
conda create -n gaugefixer python=3.10
conda activate gauge
```

### Users

Install the latest stable from PyPI using the pip package manager

```bash
pip install gaugefixer
```
Online documentation is available at [readthedocs](https://gaugefixer.readthedocs.io)

### Developers

Download the lastest version from GitHub

```bash
git clone https://github.com/jbkinney/gaugefixer.git
```

and install the package in development model

```bash
pip install -e gaugefixer
```

Run tests with:

```bash
pytest test 
```

To build the documentation of the package:

```bash
cd docs
pip install -r requirements.txt
bash build_docs.sh
```

## License

See the LICENSE file for details.

## Cite GaugeFixer

Carlos Martí-Gómez, David M. McCandlish, Justin B. Kinney (2025).
GaugeFixer: Removing unconstrained degrees of freedom in sequence-function relationships. 
In preparation.

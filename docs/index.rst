========================================================================================
GaugeFixer: Removing Unconstrained Degrees of Freedom in Sequence-Function Relationships
========================================================================================

``GaugeFixer`` is a lightweight Python package designed to remove unconstrained degrees of freedom
in the parameters of linear models for sequence-function relationships, where sequences are encoded
using indicator functions on subsequences.

History and Motivation
======================
Over the last few years, the Kinney and McCandlish labs have developed a theory of gauge freedoms
and proposed a family of linear Gauges that includes commonly used Gauges for representing 
sequence-function relationships.

- Anna Posfai, Juannan Zhou, David M. McCandlish, Justin B. Kinney (2025).
  Gauge fixing for sequence-function relationships. PLOS Computational Biology. 
  `doi.org/10.1371 <https://doi.org/10.1371/journal.pcbi.1012818>`_

- Anna Posfai, David M. McCandlish, Justin B. Kinney (2025).
  Symmetry, gauge freedoms, and the interpretability of sequence-function relationships. Physical Review Research. 
  `doi.org/10.1103 <https://doi.org/10.1103/PhysRevResearch.7.023005>`_

While the theory is well established, computational tools for fixing the gauge for a subset of linear gauges
are only partially present in `MAVE-NN <https://github.com/jbkinney/mavenn/tree/master>`_, 
a software package for quantitative modeling of sequence-function relationships. ``GaugeFixer`` was born
out of the need for a comprehensive, efficient, and user-friendly set of tools for fixing the gauge
in a way that is independent of the specific implementation of the model.

Installation
============

We recommend using a new independent environment with `Python >3.10`, as used during 
development and testing of ``GaugeFixer``, to minimize problems with dependencies. For instance,
one can create and activate a new conda environment as follows: ::

        $ conda create -n gauge python=3.10
        $ conda activate gauge

``GaugeFixer`` is available on PyPI and installable through the ``pip`` package manager: ::

        $ pip install gaugefixer

You can also install the latest or specific versions from `GitHub <https://github.com/jbkinney/gaugefixer>`_ as follows: ::

        $ git clone https://github.com/jbkinney/gaugefixer.git

and install it in the current Python environment: ::
        
        $ cd gaugefixer
        $ pip install .

For developers, tests can be run using ``pytest``: ::

        $ pytest test

Citation
========

- Carlos Martí-Gómez, David M. McCandlish, Justin B. Kinney (2025).
  GaugeFixer: Removing unconstrained degrees of freedom in sequence-function relationships. 
  In preparation.

.. toctree::
        :maxdepth: 1
        :caption: Table of Contents

        usage/tutorial.ipynb
        api

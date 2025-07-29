Welcome to PST's Documentation
==============================

Overview
========
The **Population Synthesis Toolkit (PST)** is a Python library intended for astronomy researchers, particularly those working in extragalactic astrophysics and stellar population studies, who need a flexible and extensible Python-based toolkit for modeling galaxy properties.
PST is designed to provide a user-friendly interface for working with **Simple Stellar Population (SSP)** models and for synthesizing a variety of observable quantities such as spectra, photometry, and equivalent widths.
In particular, PST is conceived to address the following challenges:
- Handling a broad variety of SSP libraries, publicly available in heterogeneous native formats.
- Modeling arbitrarily complex galaxy star formation and chemical evolution histories.
- Enabling the simultaneous and self-consistent analysis of photometric and spectroscopic data from different instruments.

At its core, PST combines individual SSPs to generate **Composite Stellar Populations (CSP)** through the implementation of **Chemical Evolution Models (CEM)**. These models track the evolution of a stellar system, considering both its **Star Formation History (SFH)**—the mass converted into stars over time—and the chemical enrichment of the gas.

PST supports a wide range of SSP models (e.g., **PopStar, XSL, MILES, BC03**) and offers flexible CEMs, allowing users to simulate observable quantities with precision. Additionally, it provides auxiliary tools to include **dust extinction** and **kinematics** effects.

Key Features
============
- Intuitive interface to multiple SSP models (e.g., **PopStar**, **XSL**, **MILES**, **BC03**).
- Implementation of various **Chemical Evolution Models (CEMs)** to simulate **Composite Stellar Populations (CSPs)**.
- Tools to compute **synthetic spectra, photometry**, and **equivalent widths** from both SSPs and CSPs.
- Optional integration of **dust extinction** and **kinematics** effects into the model outputs.
- Modules for additional functionality, such as fitting **Spectral Energy Distributions (SEDs)** to observed data.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   user_guide
   tutorials
   get_ssp_data
   api


Acknowledgement
===============

If you use PST in your research, please cite the following `paper <https://joss.theoj.org/papers/10.21105/joss.08203>`_:

.. code-block:: latex

   @article{PST_2025,
            doi = {10.21105/joss.08203},
            url = {https://doi.org/10.21105/joss.08203},
            year = {2025},
            publisher = {The Open Journal},
            volume = {10},
            number = {111},
            pages = {8203},
            author = {Corcho-Caballero, Pablo and Ascasibar, Yago and Jiménez-López, Daniel},
            title = {The Population Synthesis Toolkit (PST) Python Library},
            journal = {Journal of Open Source Software} }


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Welcome to PST's Documentation
==============================

Overview
========
The **Population Synthesis Toolkit (PST)** is a Python library designed to provide a user-friendly interface for working with **Simple Stellar Population (SSP)** models and synthesizing a variety of observable quantities such as spectra, photometry, and equivalent widths.

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

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

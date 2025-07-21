---
title: 'The Population Synthesis Toolkit (PST) Python Library'
tags:
  - Python
  - astronomy
  - astrophysics
  - galaxies
  - stellar population synthesis
authors:
  - name: Pablo Corcho-Caballero 
    orcid: 0000-0001-6327-7080
    corresponding: true
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Yago Ascasibar 
    orcid: 0000-0003-1577-2479
    corresponding: true
    equal-contrib: true
    affiliation: "2, 3"
  - name: Daniel Jiménez-López
    orcid: 0009-0001-2907-8691
    corresponding: true
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 4 
affiliations:
 - name: Kapteyn Astronomical Institute, University of Groningen, the Netherlands
   index: 1
 - name: Department of Theoretical Physics, Universidad Autónoma de Madrid (UAM), Spain
   index: 2
 - name: Centro de Investigación Avanzada en Física Fundamental (CIAFF-UAM), Spain
   index: 3
 - name: Observatorio Astronómico Nacional (OAN), Spain
   index: 4
date: 18 December 2024
bibliography: paper.bib

---

# Summary

Stellar population synthesis is a crucial methodology in astrophysics, enabling the interpretation of the integrated light of galaxies and stellar clusters. By combining empirical and/or theoretical libraries of the spectral energy distributions emitted by simple stellar populations (SSPs) with star formation history (SFH) and chemical evolution models, population synthesis can help estimate essential properties of galaxies, such as total stellar mass, star formation rate, mass-weighted age, metallicity, and so on.

PST is a Python library that offers a comprehensive and flexible framework for stellar population synthesis. Its main goal is to easily and efficiently compute composite spectra using different galaxy evolution models and SSP libraries. It also incorporates additional effects such as cosmic redshift, dust extinction and attenuation, and computes several observable quantities derived from the spectra, including broadband photometric fluxes and equivalent widths.

# State of the field

A number of software packages have been developed to support stellar population synthesis and modeling of galaxy spectral energy distributions. Tools such as [binary_c-python](https://gitlab.surrey.ac.uk/ri0005/binary_c-python) [@hendrik&Izzard23] and [SPISEA](https://github.com/astropy/SPISEA) [@hosek+20] are designed primarily for generating and analyzing simple stellar populations, often with a focus on individual stars, binaries, or star clusters. Meanwhile, libraries such as [python-FSPS](https://github.com/dfm/python-fsps) [@ben_johnson_2024_12447779], a Python interface to the Flexible Stellar Population Synthesis (FSPS) code [@conroy+09, @conroy&gunn10], and the more recent [DSPS](https://github.com/ArgonneCPAC/dsps) [@hearin+23], implemented using JAX for efficient gradient computation and forward modeling, provide extensive modeling capabilities, although they are sometimes limited to a specific set of SSP models or isochrones.

Other packages put a stronger emphasis on fitting observed data to derive galaxy properties. Examples of these include  Bayesian frameworks such as [CIGALE](https://cigale.lam.fr/) [@boquien+19], [ProSpect](https://github.com/asgr/ProSpect) [@robotham+20] and [Prospector](https://prospect.readthedocs.io/en/v1.0.0/) [@johnson+21], which infer star formation histories and other physical parameters using spectro-photometric data. Alternative frequentist tools such as [PpXF](https://pypi.org/project/ppxf/) [@capellari+04], [Starlight](http://www.starlight.ufsc.br/) [@cid-fernandes+05], or [Pipe3D](https://gitlab.com/pipe3d/pyPipe3D) [@sanchez+16], are commonly used to extract stellar kinematics and stellar population parameters from observed galaxy spectra, often in the context of integral field spectroscopy.

# Statement of need

The user-friendly modular framework of PST is designed to address the following challenges:

- Handle a broad variety of SSP libraries that are publicly available in heterogeneous native formats.
- Model arbitrarily complex galaxy star formation and chemical evolution histories.
- Enable the simultaneous and self-consistent analysis of photometric and spectroscopic data from different instruments.

PST is designed for astronomy researchers, especially those working in extragalactic astrophysics and stellar population synthesis, who require a flexible and extensible Python-based toolkit for modeling galaxy properties. PST is suited for users with intermediate to advanced Python expertise and familiarity with common data formats and concepts in astronomical spectroscopy and photometry.

The primary use cases are data analysis, synthetic model construction, and pipeline integration for studies involving stellar population synthesis (see the examples below). PST is particularly useful in workflows that combine observational data with theoretical models within a Bayesian or forward-modeling framework.

PST is currently a dependency of PyKOALA [@pykoala], another  open-source Python package focused on reducing optical integral-field spectroscopic observations. There, PST is mainly used to derive broadband photometry.
PST is also at the core of the Bayesian Estimator for Stellar Population Analysis [[BESTA](https://https://besta.readthedocs.io/), see also @cc+25], where it is coupled with the [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) [@zuntz+15] Monte Carlo sampling framework to infer the physical properties of galaxies from the observed colors and spectra.

# Features and functionality

PST design is built around three main components.

First, the SSP module enables the consistent use and manipulation of different SSP libraries. This allows for the seamless ingestion of models and data from various literature sources.
The current version includes interfaces to a range of SSP models, including PopStar [@molla+09], Bruzual and Charlot (BC03) [@bc+03], E-MILES [@vazdekis+16], and the X-Shooter Spectral Library (XSL) [@verro+22] SSP models.

For any SSP model integrated into PST, the library provides tools for interpolating across stellar ages, metallicities, and wavelengths. Users can easily compute key SSP quantities, such as the stellar mass-to-light ratio in a given band, colors, and line indices.

Second, the `ChemicalEvolutionModel` classes represent the star formation and chemical enrichment histories required to produce composite spectral energy distributions and additional derived quantities. These classes implement several widely used analytic prescriptions for modeling star formation histories (SFHs), such as exponentially declining or log-normal models. They also implement complex SFH representations, such as table-based SFHs and particle-like data models. These models are particularly suitable for post-processing results from cosmological hydrodynamic simulations.

Third, PST features a dedicated `observables` module that can predict additional quantities from spectra, such as broadband photometric fluxes, colours, and equivalent widths, which are useful for estimating the strength of absorption or emission lines. PST includes automatic integration with the photometric filters provided by the [Spanish Virtual Observatory Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/) [@rodrigo+20] for synthetic photometry calculations, as well as popular line indices such as the Lick system [@worthey+94].

# Documentation and tutorials

To make PST easier to use, we provide a set of comprehensive tutorials in the form of Jupyter notebooks. These tutorials cover the following topics:

- Interacting with SSP models and exploring their fundamental properties.
- Producing composite spectra using:
  - Analytic SFH models;
  - Table-based SFH models;
  - Particle-like data representations.
- Predicting observable quantities for a grid of models.

Full documentation is available [online](https://population-synthesis-toolkit.readthedocs.io/en/latest/).

# Acknowledgements

We acknowledge financial support from the Spanish State Research Agency (AEI/10.13039/501100011033) through grant PID2019-107408GB-C42.

Daniel Jiménez-López was supported by Fondo Europeo de Desarrollo Regional (MINCIN/AEI/10.13039/501100011033/FEDER, UE), through a FPI-contract fellowship in the project PID2022-138560NB.

Our package relies on several widely used open-source Python libraries, including [Numpy](http://www.numpy.org) [@harris2020array], [Matplotlib](https://www.matplotlib.org/) [@hunter:2007] and [Astropy](http://www.astropy.org) [@astropy:2013, @astropy:2018, @astropy:2022].

# References

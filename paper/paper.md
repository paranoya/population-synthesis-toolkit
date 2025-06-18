---
title: 'The Population Synthesis Toolkit (PST) Python library'
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

Stellar population synthesis is a crucial methodology in astrophysics, enabling the interpretation of the integrated light of galaxies and stellar clusters. By combining empirical and/or theoretical libraries of the spectral energy distribution emitted by simple stellar populations (SSPs) with models of the star formation history (SFH) and chemical evolution, population synthesis facilitates the estimation of essential galaxy properties, such as total stellar mass, star formation rate, mass-weighted age and metallicity, etc.

PST is a Python library that offers a comprehensive and flexible framework for stellar population synthesis. Its main goal is to compute composite spectra using different galaxy evolution models and SSP libraries with ease and efficiency. It incorporates additional effects, such as cosmic redshift and dust extinction, and it computes several observable quantities derived from the spectra, including broadband photometric fluxes and equivalent widths.

# State of the field

A number of software packages have been developed to support stellar population synthesis and modeling of galaxy spectral energy distributions. Tools such as [binary_c-python](https://gitlab.surrey.ac.uk/ri0005/binary_c-python) and [SPISEA](https://github.com/astropy/SPISEA) [@hosek+20] are primarily designed for generating and analyzing simple stellar populations, often with a focus on individual stars, binaries, or star clusters. Meanwhile, libraries such as [python-FSPS](https://github.com/dfm/python-fsps), a Python interface to the Flexible Stellar Population Synthesis (FSPS) code [@conroy+09], and the more recent [DSPS](https://github.com/ArgonneCPAC/dsps) [@hearin+23], implemented using JAX for efficient gradient computation and forward modeling, provide extensive modeling capabilities, although they are sometimes limited to a specific set of SSP models or isochrones.

Other packages put a stronger emphasis on fitting observed data to derive galaxy properties. These include Bayesian frameworks like [CIGALE](https://cigale.lam.fr/) [@boquien+19], [ProSpect](https://github.com/asgr/ProSpect) [@robotham+20] or [Prospector](https://prospect.readthedocs.io/en/v1.0.0/) [@johnson+21], which infer star formation histories and other physical parameters using spectro-photometric data. Alternative frequentist tools such as [PpXF](https://pypi.org/project/ppxf/) [@capellari+04], [Starlight](http://www.starlight.ufsc.br/) [@cid-fernandes+05], or [Pipe3D](https://gitlab.com/pipe3d/pyPipe3D) [@sanchez+16], are commonly used to extract stellar kinematics and stellar population parameters from observed galaxy spectra, often in the context of integral field spectroscopy.

# Statement of need

Compared to alternative approaches, the user-friendly modular framework of PST is conceived to address the following challenges:

- To handle a broad variety of SSP libraries, publicly available in heterogeneous native formats.
- To model arbitrarily complex galaxy star formation and chemical evolution histories.
- To enable the simultaneous and self-consistent analysis of photometric and spectroscopic data from different instruments.

PST is designed for astronomy researchers, particularly those working in extragalactic astrophysics and stellar population synthesis, who need a flexible and extensible Python-based toolkit for modeling galaxy properties. It is suited for users with intermediate to advanced expertise in Python and familiarity with common data formats and concepts in astronomical spectroscopy and photometry.

Primary use cases include data analysis, synthetic model construction, and pipeline integration for studies involving stellar population synthesis (see examples below). PST is especially valuable in workflows that combine observational data with theoretical models in a Bayesian or forward-modeling context.

PST is currently a dependency of PyKOALA [@pykoala], another open source Python package focused on the reduction of optical integral-field spectroscopic observations, where it is mainly used to derive broadband photometry.
It is also at the core of the Bayesian Estimator for Stellar Population Analysis [[BESTA](https://https://besta.readthedocs.io/), see also @cc+25], where it is coupled with the [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) [@zuntz+15] Monte Carlo sampling framework to infer the physical properties of galaxies from the observed colours and spectra.

# Features and functionality

PST design is built around three main components:

First, the `SSP` module allows for the uniform use and manipulation of different SSP libraries, enabling seamless ingestion of models and data from various sources in the literature.
The current version includes interfaces to a range of SSP models, including:
- PopStar [@molla+09]
- Bruzual and Charlot (BC03) [@bc+03]
- E-MILES [@vazdekis+16]
- X-Shooter Spectral Library (XSL) [@verro+22] SSP models.

For any SSP model integrated into PST, the library provides tools for interpolating across stellar ages, metallicities, and wavelengths. Users can easily compute key quantities of the SSP, such as the stellar-mass-to-light ratio in a given band, colours, line indices, etc.

Second, the `ChemicalEvolutionModel` classes represent the star formation and chemical enrichment histories required to produce composite spectral energy distributions and additional derived quantities. They implement several widely-used analytic prescriptions for modeling SFHs, such as exponentially declining or log-normal models, as well as complex SFH representations, such as table-based SFHs and particle-like data models, particularly suitable for post-processing the results from cosmological hydrodynamical simulations.

Third, PST features a dedicated `observables` module to predict additional quantities from spectra, such as broadband photometric fluxes, colours, and equivalent widths. PST includes automatic integration with the photometric filters provided by the [Spanish Virtual Observatory Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/) [@rodrigo+20] for synthetic photometry calculations, as well as popular line indices such as the Lick system [@worthey+94].

# Documentation and tutorials

To facilitate the use of PST, we provide a comprehensive set of tutorials in the form of Jupyter notebooks. These tutorials cover the following topics:

- Interacting with SSP models and exploring their fundamental properties.
- Producing composite spectra using:
  - Analytic SFH models.
  - Table-based SFH models.
  - Particle-like data representations.
- Predicting observable quantities for a grid of models.

Full documentation is available [online](https://population-synthesis-toolkit.readthedocs.io/en/latest/).

# Acknowledgements

We acknowledge financial support from the Spanish State Research Agency (AEI/10.13039/501100011033) through grant PID2019-107408GB-C42.

Daniel Jiménez-López was supported by Fondo Europeo de Desarrollo Regional (MINCIN/AEI/10.13039/501100011033/FEDER, UE), through a FPI-contract fellowship in the project PID2022-138560NB.

# References

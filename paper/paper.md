---
title: 'Population Synthesis Toolkit (PST): A Python library for '
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
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Yago Ascasibar 
    orcid: 0000-0003-1577-2479
    corresponding: true
    equal-contrib: true
    affiliation: "2, 3"
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Kapteyn Astronomical Institute, University of Groningen, the Netherlands
   index: 1
   ror: 00hx57361
 - name: Department of Theoretical Physics, Universidad Autónoma de Madrid (UAM), Spain
   index: 2
 - name: Centro de Investigación Avanzada en Física Fundamental (CIAFF-UAM), Spain
   index: 3
date: 18 December 2024
bibliography: paper.bib

---

# Summary

Stellar population synthesis is a crucial methodology in astrophysics, enabling the interpretation of the integrated light of galaxies and stellar clusters. By combining empirical and/or theoretical libraries of the spectral energy distribution emitted by simple stellar populations (SSPs) with models of the star formation history (SFH) and chemical evolution, population synthesis facilitates the estimation of essential galaxy properties, such as total stellar mass, star formation rate, mass-weighted age and metallicity, etc.

PST is a Python library that offers a comprehensive and flexible framework for stellar population synthesis. It supports the synthesis of composite spectra using multiple SSP libraries, incorporates additional effects such as cosmic redshift and dust extinction, and computes observable quantities, including broadband photometric fluxes and equivalent widths, with ease and efficiency.

# Statement of need

Compared to alternative approaches in the literature, the user-friendly modular framework of PST is conceived to address the following challenges:
- To handle a broad variety of SSP libraries, publicly available in heterogeneous native formats.
- To model arbitrarily complex galaxy star formation and chemical evolution histories.
- To model both photometric and spectroscopic data.

# Features and functionality

PST design is built around three core components:

First, the `SSP` module allows for the uniform use and manipulation of different SSP libraries, enabling seamless ingestion of models and data from various sources in the literature.
The current version includes interfaces to a range of SSP models, including:
- PopStar `[@molla+09]`
- Bruzual and Charlote 03 `[@bc+03]`
- E-MILES `[@vazdekis+16]`
- XSL `[@verro+22]` SSP models.
For any SSP model integrated into PST, the library provides tools for interpolating across stellar ages, metallicities, and wavelengths. Users can easily compute key quantities of SSP models, such as the stellar-mass-to-light ratio in a given band, colours, line indices, etc.

Second, the `ChemicalEvolutionModel` classes represent the star formation and chemical enrichment histories required to produce spectral energy distributions and additional derived quantities. They implement several widely-used analytic prescriptions for modeling SFHs, such as exponentially declining or log-normal models, as well as complex SFH representations, such as table-based SFHs and particle-like data models, particularly suitable for post-processing the results from cosmological hydrodynamical simulations.

Third, PST features a dedicated `observables` module to predict additional quantities from spectra, such as broadband photometric fluxes and equivalent widths. PST includes automatic integration with the photometric filters provided by the [Spanish Virtual Observatory Filter Profile Servive](http://svo2.cab.inta-csic.es/theory/fps/) for synthetic photometry calculations, as well as popular line indices such as the Lick system.

# Tutorials

To facilitate the use of PST, we provide a comprehensive set of tutorials in the form of Jupyter notebooks. These tutorials cover the following topics:

- Interacting with SSP models and exploring their fundamental properties.
- Producing composite spectra using:
  - Analytic SFH models.
  - Table-based SFH models.
  - Particle-like data representations.
- Predicting observable quantities for a grid of models.

# Acknowledgements

?

# References

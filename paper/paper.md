---
title: 'Population Synthesis Toolkit (PST): A Python library for '
tags:
  - Python
  - astronomy
  - stellar population synthetis
authors:
  - name: Pablo Corcho-Caballero 
    orcid: 0000-0001-6327-7080
    corresponding: true
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
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
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 18 December 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Stellar population synthesis (SPS) is a crucial methodology in astrophysics, enabling the interpretation of the integrated light of galaxies and stellar clusters. By combining the light emitted by single stellar populations (SSPs) with models of star formation history (SFH) and chemical enrichment, SPS facilitates the estimation of essential galaxy properties such as stellar metallicity, age, and mass. However, the vast variety of SSP models and the diversity of approaches for modeling galaxy SFHs can introduce significant complexity.

To address these challenges, PST is a Python library that offers a comprehensive and flexible framework for stellar population synthesis. PST supports the synthesis of spectra using multiple SSP models, incorporates additional effects such as dust extinction, and computes observable quantities, including photometric fluxes and equivalent widths, with ease and efficiency.

# Statement of need

PST provides a highly flexible framework for synthesizing stellar populations and deriving key observational quantities. Its design is built around two core components:

- SSP Module: This module allows for the uniform use and manipulation of arbitrary SSP models, enabling seamless ingestion of SSP data from various sources in the literature.
- ChemicalEvolutionModel Classes: These classes represent the chemical evolution and SFH models used to produce composite spectra and additional derived quantities. They simplify the creation and implementation of custom prescriptions for chemical enrichment and SFH modeling.

In addition, PST features a dedicated module for computing observables, offering a user-friendly interface to predict additional quantities from spectra, such as photometric fluxes and equivalent widths. PST includes automatic integration with the photometric filters provided by the [Spanish Virtual Observatory Filter Profile Servive](http://svo2.cab.inta-csic.es/theory/fps/) for synthetic photometry calculations.

# Features and functionality

PST includes interfaces to a range of SSP models, including:

- PopStar [@molla+09]
- Bruzual and Charlote 03 [@bc+03]
- E-MILES [@vazdekis+16]
- XSL [@verro+21b] SSP models.

For any SSP model integrated into PST, the library provides tools for straightforward interpolation across stellar ages, metallicities, and wavelengths. Users can easily compare key quantities of SSP models such as the stellar-mass-to-light ratio.

PST also includes several widely-used analytic prescriptions for modeling SFHs, such as exponentially declining or log-normal models. Additionally, it supports complex SFH representations, such as table-based SFHs, and particle-like data models, making it particularly suitable for interpreting data from cosmological hydrodynamical simulations.

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

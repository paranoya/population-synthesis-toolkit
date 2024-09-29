.. _user_guide:

=====================
User Guide
=====================

This guide covers the detailed usage of PST for varius tasks.

Concepts
========
Before using PST, it's important to understand the following key scientific concepts:

- **Simple Stellar Population Model**: Models representing groups of stars with a specific age, metallicity, and initial mass function.
- **Initial Mass Function (IMF)**: The distribution of star masses at birth, critical to synthesizing stellar populations.
- **Star Formation Histories (SFH)**: Describes the rate at which stars form over time in a galaxy or stellar cluster.

Single Stellar Population (SSP) Models
=======================================
An SSP model represents a library of simple stellar populations (i.e. a set of stars formed at the same time with the same metallicity).
The following example demonstrates how to initialize an SSP model:

.. code-block:: python

    from pst.SSP import PopStar
    import matplotlib.pyplot as plt

    # Initialize the SSP model
    ssp_model = PopStar(IMF="cha")

    print("SSP model consisting of: \n",
    f"{ssp_model.ages.size} ages from {ssp_model.ages[0]} to {ssp_model.ages[-1]}",
    "\n",
    f"{ssp_model.metallicities.size} metallicities from {ssp_model.metallicities[0]} to {ssp_model.metallicities[-1]}")

    print("Wavelength range of the SSP model goes from"
          f"{ssp_model.wavelength[0]} to {ssp_model.wavelength[-1]}",
          f"\nSED shape {ssp_model.L_lambda.shape}")

    # Plot a spectrum
    metallicity_idx = 2
    age_idx = 50
    plt.figure()
    plt.plot(ssp_model.wavelength, ssp_model.L_lambda[metallicity_idx, age_idx])
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Luminosity')
    plt.title(f'SSP Spectrum (Z={ssp_model.metallicities[metallicity_idx]}, Age={ssp_model.ages[age_idx]:.1f})')
    plt.show()

Chemical Evolution Models (CEM)
===============================
A CSP represents a population of stars formed over a range of times, following a star formation history (SFH). You can model a CSP with PST as follows:

.. code-block:: python

    from mysps import CSPModel

    # Initialize the CSP model
    csp_model = CSPModel()

    # Define an exponentially decaying star formation history
    sfh = csp_model.exponential_sfh(tau=2.0)  # Tau in Gyr

    # Define metallicity and IMF
    metallicity = 0.02
    imf_type = 'Chabrier'

    # Retrieve the spectrum
    spectrum = csp_model.get_spectrum(sfh, metallicity, imf_type)

    # Plot the spectrum
    plt.plot(spectrum.wavelength, spectrum.flux)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.title('CSP Spectrum with Exponential SFH')
    plt.show()

For more options, refer to the :ref:`api_reference`.

observables
===========

Dust
====


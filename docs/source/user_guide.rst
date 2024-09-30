.. _user_guide:

=====================
User Guide
=====================

This guide covers the detailed usage of PST for varius tasks. For practical cases, see the :ref:`tutorials <tutorials>` section.

Simple Stellar Population (SSP) Models
======================================

**Simple Stellar Population (SSP)** models are an essential tool in understanding the evolution and properties of stellar systems. An SSP represents a group of stars formed at the same time with the same initial chemical composition. By assuming a single age and metallicity, SSPs provide a framework for modeling how these populations evolve over time and allow the synthesis of observables, such as spectra and photometry.

In this section, we'll dive into the SSP models implemented in PST, explore their attributes, and highlight the features that this module provides for astrophysical modeling.

:class:`SSPBase`: Core Class for SSP Models

The class :class:`SSPBase` is the backbone of the SSP models in PST. It provides a flexible and efficient way to represent the spectral energy distributions (SEDs) of stellar populations across a grid of ages and metallicities.

Key Attributes
^^^^^^^^^^^^^^

- **ages** (:class:`astropy.units.Quantity`): An array of SSP ages, representing different stages of stellar evolution.
- **metallicities** (:class:`astropy.units.Quantity`): The metallicity values (fraction of elements heavier than helium) for the SSPs.
- **L_lambda** (:class:`astropy.units.Quantity`): The spectral energy distributions (SEDs) of the SSPs. This 3D array holds data for each combination of metallicity, age, and wavelength.
- **wavelength** (:class:`astropy.units.Quantity`): The wavelength array associated with the SEDs of the SSPs, allowing the user to model the flux over a range of wavelengths.

### Features Provided by PST SSP Models

PST provides a number of features to work with SSP models, allowing users to manipulate and extract various physical and observational quantities:

1. **SSP Interpolation**
   The `get_weights` method provides a 2D interpolation tool that allows you to compute weights for a given set of ages and metallicities. This method is useful for handling multiple stellar populations simultaneously, especially in galaxy synthesis models where stars form over a range of ages and metallicities.

    Example:

    .. code-block:: python

            weights = ssp_model.get_weights(ages=[1e9, 5e9], metallicities=[0.02, 0.03])


2. **Grid Binning and Re-interpolation**
    The `regrid` method allows you to rebin the SSP model to new grids of ages and metallicities. This is useful if you need to match the SSP model grid to other datasets or models.

    Example:

    .. code-block:: python

            new_age_bins = np.logspace(6, 10, 50) * u.Gyr
            new_metal_bins = np.logspace(-2, 0, 20)
            ssp_model.regrid(new_age_bins, new_metal_bins)

3. **Wavelength Handling**
    The `cut_wavelength` method allows users to cut the SED to specific wavelength ranges, which is useful when focusing on particular bands or wavelengths.

    Example:

    .. code-block:: python

            ssp_model.cut_wavelength(wl_min=3000 * u.AA, wl_max=7000 * u.AA)

    Additionally, `interpolate_sed` provides a way to interpolate the SEDs over new wavelength bins using a flux-conserving interpolation scheme.

4. **Mass-to-Light Ratios**
    The get_mass_lum_ratio and get_specific_mass_lum_ratio methods calculate the mass-to-light ratio over a specified wavelength range, providing critical information for stellar population synthesis models.

    Example:

    .. code-block:: python

            mass_lum_ratio = ssp_model.get_mass_lum_ratio(wl_range=np.array([4000, 7000]) * u.angstrom)


5. **Synthetic Photometry**
    One of the most powerful features is the ability to compute synthetic photometry using the `compute_photometry` method. This function calculates the flux observed through a set of photometric filters at a given cosmic time.

    Example:

    .. code-block:: python

            filters = load_photometric_filters(["SLOAN_SDSS.g", "SLOAN_SDSS.r"])
            photometry = ssp_model.compute_photometry(filters, z_obs=0.0)


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

Observables
===========


Dust extinction effects
=======================


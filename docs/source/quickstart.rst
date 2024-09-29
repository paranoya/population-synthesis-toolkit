.. _quickstart:

==================
Quickstart Guide
==================

This guide demonstrates how to quickly get started with PST by modeling a combining the spectra of several simple stellar populations.

Minimal Working Example
========================
Here's how to load a Single Stellar Population (SSP) model and calculate its photometric properties:

.. code-block:: python

    from pst.SSP import PopStar
    from pst.observables import load_photometric_filters
    import numpy as np

    imf_type = 'cha'  # Chabrier 2003 Initial Mass Function
    z_redshift = 0.0 # Redshift at which the SSP are observed
    list_of_filter_names = ["SLOAN_SDSS.g", "SLOAN_SDSS.r"]
    filters = load_photometric_filters(list_of_filter_names)
    # Initialize the model
    ssp_model = PopStar(IMF=imf_type)

    # Calculate photometry of each ssp
    photometry_fluxes = ssp_model.compute_photometry(filter_list=filters,
                                                     z_obs=z_redshift)
    # Convert to AB magnitude
    magnitudes = -2.5 * np.log10(photometry_fluxes.to_value("3631 Jy / Msun"))

    # Compute g - r color
    color = magnitudes[0] - magnitudes[1]
    # Plot the color as function of stellar age and metallicity
    import matplotlib.pyplot as plt

    print(color)
    plt.figure()
    plt.pcolormesh(np.log10(ssp_model.ages.value),
                np.log10(ssp_model.metallicities), color,
                cmap='jet')
    plt.colorbar(label=r'$g-r$')
    plt.show()

This simple example initializes an SSP model and computes the photometry associated to each SSP spectrum in the `g`,  and `r` bands, and plots the color as function of SSP age and metallicity.

For a more detailed guide, see the :ref:`user_guide`.
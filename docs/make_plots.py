#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:40:50 2025

@author: pcorchoc
"""

from pst.SSP import PopStar
from pst.observables import load_photometric_filters
import numpy as np
from matplotlib import pyplot as plt

# Define the IMF and redshift
imf_type = 'cha'  # Chabrier 2003 Initial Mass Function
z_redshift = 0.0  # Redshift at which the SSPs are observed

# Load photometric filters (g and r bands)
list_of_filter_names = ["SLOAN_SDSS.g", "SLOAN_SDSS.r"]
filters = load_photometric_filters(list_of_filter_names)

# Initialize the SSP model
ssp_model = PopStar(IMF=imf_type)

# Compute photometry for the SSPs in the selected filters
photometry_fluxes = ssp_model.compute_photometry(filter_list=filters,
                                                 z_obs=z_redshift)

# Convert fluxes to AB magnitudes
magnitudes = -2.5 * np.log10(photometry_fluxes.to_value("3631 Jy / Msun"))

# Compute g - r color
color = magnitudes[0] - magnitudes[1]

# Plot the color as a function of stellar age and metallicity
plt.figure()
plt.pcolormesh(np.log10(ssp_model.ages.value),
               np.log10(ssp_model.metallicities.value), color,
               cmap='jet')
plt.colorbar(label=r'$g - r$')
plt.xlabel('log(Age / yr)')
plt.ylabel('log(Z / Z_sun)')
plt.title(r'$g - r$ Color as a Function of Age and Metallicity')

plt.savefig("source/_static/images/gr_popstar.png", dpi=250, bbox_inches="tight")
plt.show()

### USER GUIDE

from pst.models import ExponentialDelayedCEM
# Create a model based on a delayed-tau exponential SFH model
# with constant metallicity
cem_model = ExponentialDelayedCEM(mass_today=1.0, today=13.7, tau=3.0,
ism_metallicity_today=0.02)

import numpy as np
from astropy import units as u 
from matplotlib import pyplot as plt

cosmic_time = np.arange(0, 13.7) * u.Gyr
mass_formation_history = cem_model.stellar_mass_formed(cosmic_time)

plt.figure()
plt.plot(cosmic_time, mass_formation_history)
plt.xlabel('Cosmic Time (Gyr)')
plt.ylabel('Stellar Mass Formed (M$_\odot$)')
plt.savefig("source/_static/images/delayed_tau_exponential_sfh.png", dpi=300, bbox_inches='tight')
plt.show()


sed = cem_model.compute_SED(ssp_model, t_obs=13.7 * u.Gyr)

plt.figure()
plt.loglog(ssp_model.wavelength, sed)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('SED (Lsun/Angstrom)')
plt.title('SED of the Delayed Tau Exponential SFH Model')
plt.savefig("source/_static/images/delayed_tau_exponential_sfh_sed.png", dpi=300, bbox_inches='tight')
plt.show()
# Compute the photometry for the SED in the selected filters


from pst.observables import Filter

# Load a JWST filter from the SVO Filter Service
jwst_miri_filter = Filter.from_svo("JWST_MIRI.F2550W")

# Compute a synthetic magnitude from the galaxy SED computed above
jwst_miri_filter.interpolate(ssp_model.wavelength)
print(sed)
ab_mag = jwst_miri_filter.get_ab(sed / 4 / np.pi / (10 * u.pc)**2)
print(ab_mag)
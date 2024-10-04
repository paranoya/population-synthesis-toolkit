from pst.SSP import PopStar
from pst.observables import load_photometric_filters
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# Initialize the model
imf_type = 'cha'  # Chabrier (2003) Initial Mass Function
ssp_model = PopStar(IMF=imf_type)  # PopStar SSPs (Mollá et al. 2009)

print("SSP model consisting of: \n",
    f"{ssp_model.ages.size} ages from {ssp_model.ages[0]} to {ssp_model.ages[-1]}",
    "\n",
    f"{ssp_model.metallicities.size} metallicities from {ssp_model.metallicities[0]} to {ssp_model.metallicities[-1]}")

print(f"Wavelength range of the SSP model goes from {ssp_model.wavelength[0]} to {ssp_model.wavelength[-1]}")
print(f"SED shape: {ssp_model.L_lambda.shape}")

# Plot a spectrum
metallicity_idx = 4
age_indices = [50, 80]
plt.figure()
plt.title(f'SSP Spectrum (Z={ssp_model.metallicities[metallicity_idx]})')
for age_idx in age_indices:
    plt.plot(ssp_model.wavelength,
             (ssp_model.wavelength * ssp_model.L_lambda[metallicity_idx, age_idx]).to_value(u.Lsun / u.Msun),
             label=f'age={ssp_model.ages[age_idx].to_value(u.Myr):.1f} Myr')
plt.legend()
plt.xlabel('Wavelength [Angstrom]')
plt.xscale('log')
plt.ylabel(r'$\lambda L_\lambda$ [L$_\odot$ / M$_\odot$]')
plt.yscale('log')
plt.ylim(3e-5, 3e3)
plt.show()

# Calculate photometry of each SSP
z_redshift = 0.0 # Redshift at which the SSP are observed
list_of_filter_names = ["SLOAN_SDSS.g", "SLOAN_SDSS.r"]
filters = load_photometric_filters(list_of_filter_names)
photometry_fluxes = ssp_model.compute_photometry(filter_list=filters,
                                                    z_obs=z_redshift)
# Convert to AB magnitude
magnitudes = -2.5 * np.log10(photometry_fluxes.to_value("3631 Jy / Msun"))

# Compute g - r color
color = magnitudes[0] - magnitudes[1]
# Plot the color as function of stellar age and metallicity
plt.figure()
plt.title(f'SSP colour as a function of age and metallicity')
plt.pcolormesh(ssp_model.ages.to_value(u.yr),
               ssp_model.metallicities,
               color,
               cmap='jet')
plt.xlabel('age [yr]')
plt.xscale('log')
plt.xlim(6e5, 1.5e10)
plt.ylabel('metallicity Z')
#plt.yscale('log')
#plt.ylim(5e-5, 0.05)
plt.ylim(0, 0.055)
plt.colorbar(label=r'$(g-r)$')
plt.show()

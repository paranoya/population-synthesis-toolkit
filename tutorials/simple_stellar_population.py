from pst.SSP import PopStar
from pst.observables import load_photometric_filters
import numpy as np
import matplotlib.pyplot as plt

imf_type = 'cha'  # Chabrier 2003 Initial Mass Function
z_redshift = 0.0 # Redshift at which the SSP are observed
list_of_filter_names = ["SLOAN_SDSS.g", "SLOAN_SDSS.r"]
filters = load_photometric_filters(list_of_filter_names)
# Initialize the model
ssp_model = PopStar(IMF=imf_type)

print("SSP model consisting of: \n",
    f"{ssp_model.ages.size} ages from {ssp_model.ages[0]} to {ssp_model.ages[-1]}",
    "\n",
    f"{ssp_model.metallicities.size} metallicities from {ssp_model.metallicities[0]} to {ssp_model.metallicities[-1]}")

print("Wavelength range of the SSP model goes from"
        f"{ssp_model.wavelength[0]} to {ssp_model.wavelength[-1]}",
        "\nSED shape {ssp_model.L_lambda.shape}")

# Plot a spectrum
metallicity_idx = 2
age_idx = 50
plt.figure()
plt.plot(ssp_model.wavelength, ssp_model.L_lambda[metallicity_idx, age_idx])
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Luminosity')
plt.title(f'SSP Spectrum (Z={ssp_model.metallicities[metallicity_idx]}, Age={ssp_model.ages[age_idx]:.1f})')
plt.show()

# Calculate photometry of each ssp
photometry_fluxes = ssp_model.compute_photometry(filter_list=filters,
                                                    z_obs=z_redshift)
# Convert to AB magnitude
magnitudes = -2.5 * np.log10(photometry_fluxes.to_value("3631 Jy / Msun"))

# Compute g - r color
color = magnitudes[0] - magnitudes[1]
# Plot the color as function of stellar age and metallicity
print(color)
plt.figure()
plt.pcolormesh(np.log10(ssp_model.ages.value),
               np.log10(ssp_model.metallicities), color,
               cmap='jet')
plt.colorbar(label=r'$g-r$')
plt.show()
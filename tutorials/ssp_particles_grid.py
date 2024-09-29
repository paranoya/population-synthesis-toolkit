import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pst.SSP import PopStar
from pst import models
from astropy import units as u

ssp = PopStar(IMF='cha')

# Create some particle data
np.random.seed(50)
n_particles = 10000
particles_z = 10**(np.random.uniform(-4, 0.3, n_particles))
particles_t_form = np.random.exponential(3, n_particles)
particles_mass = 10**(np.random.uniform(5, 6, n_particles))

sfh_model = models.ParticleGridCEM(time_form=particles_t_form * u.Gyr,
                        metallicities=particles_z * u.dimensionless_unscaled,
                        masses=particles_mass * u.Msun)

cosmic_time = np.geomspace(1e-3, 13.7, 300) * u.Gyr


# plt.figure()
# plt.hist(particles_t_form, weights=particles_mass)


mass_history = sfh_model.stellar_mass_formed(cosmic_time)

plt.figure()
plt.subplot(111)
plt.plot(cosmic_time, mass_history)

ssp_weights = sfh_model.interpolate_ssp_masses(ssp, t_obs=13.7 * u.Gyr)
plt.figure()
plt.pcolormesh(np.log10(ssp.ages.to_value("Gyr")), np.log10(ssp.metallicities), ssp_weights,
               norm=LogNorm())
plt.colorbar()
plt.show()

sed = sfh_model.compute_SED(ssp, t_obs=13.7 * u.Gyr)
plt.figure()
plt.plot(ssp.wavelength, sed)
plt.show()
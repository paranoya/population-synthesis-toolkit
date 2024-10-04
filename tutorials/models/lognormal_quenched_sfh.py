import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PopStar
from pst import models
from astropy import units as u

from cProfile import Profile
from pstats import SortKey, Stats
from time import time as ttime

# Initialise the SSP model
ssp = PopStar(IMF='cha')

# Define the parameters of the Chemical Evolution Model
alpha_powerlaw = -2.0
ism_metallicity_today = 0.02 * u.dimensionless_unscaled
t0 = 7.0 * u.Gyr
scale = 2.0
quenching_time = 10 * u.Gyr

model = models.LogNormalQuenchedCEM(
    alpha_powerlaw=alpha_powerlaw, ism_metallicity_today=ism_metallicity_today,
    mass_today=1234 * u.Msun,
    today=13.7,
    t0=t0, scale=scale, quenching_time=quenching_time)

dummy_t = np.linspace(0, 13.7, 1000) * u.Gyr

# Check the mass formation history of the model
plt.figure()
plt.title("Log-normal SFH with a quenching event")
plt.plot(dummy_t, model.stellar_mass_formed(dummy_t))
plt.axvline(quenching_time.to_value(dummy_t.unit), color='r', label="Quenching event")
plt.legend()
plt.xlabel("Cosmic time")
plt.ylabel("Stellar mass formed")
plt.show()

# Estimate the elapsed time when computing the synthetic spectra
t0 = ttime()
model.compute_SED(ssp, t_obs=13.7 * u.Gyr)
print("Time spent computing the SED ", ttime() - t0)


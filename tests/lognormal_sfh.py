import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar, BaseGM, XSL
from pst import models
from astropy import units as u

from cProfile import Profile
from pstats import SortKey, Stats
from time import time as ttime

ssp = XSL(IMF='Kroupa', ISO='P00')

lbtime = np.geomspace(1e-3, 13.7, 300)
time = 13.7 - lbtime[::-1]
time[-1] = 13.7

dummy_t = np.linspace(0, 13.7, 1000) * u.Gyr

tau = 30.0
alpha = 0.0
z_0 = 0.02
t_0 = 7.0

model = models.Exponential_SFR(
    Z=z_0 << u.dimensionless_unscaled,
    M_inf= 1 << u.Msun,
    tau=3 << u.Gyr)

plt.figure()
plt.plot(dummy_t, model.integral_SFR(dummy_t))
plt.show()

sed = model.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)

plt.figure()
plt.plot(ssp.wavelength, sed)
plt.show()

np.savetxt("test_spectra",
           np.array([ssp.wavelength.to_value("angstrom"), sed.to_value("erg/s/angstrom")]).T,
           header="sfh: Exponential, tau: 3.0 Gyr, Z=0.02\n Wavelength (AA), Luminosity (erg/s/AA)")
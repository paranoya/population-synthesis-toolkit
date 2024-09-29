import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar, BaseGM, XSL
from pst import models
from astropy import units as u

from cProfile import Profile
from pstats import SortKey, Stats
from time import time as ttime

ssp = PyPopStar(IMF='KRO')

lbtime = np.geomspace(1e-3, 13.7, 300)
time = 13.7 - lbtime[::-1]
time[-1] = 13.7

dummy_t = np.linspace(0, 13.7, 1000) * u.Gyr

tau = 30.0
alpha = -2.0
z_0 = 0.02
t_0 = 7.0

def exponential_sfh(time, tau):
    m =  (1 - np.exp(-time / tau)) 
    m /= m[-1]
    return m* u.Msun

def z_star(time, alpha, z_0, t_0):
    z = z_0 * (1 - np.power((time + t_0)/ t_0, alpha))
    return z * u.dimensionless_unscaled

m1 = exponential_sfh(time, tau)
z1 = np.ones_like(m1.value) * 0.02 * u.dimensionless_unscaled
z1 = z_star(time, alpha, z_0, t_0)

model = models.Tabular_MFH(times=time * u.Gyr, masses=m1, Z=z1)
print("PROFILING >>>\n\n")
t0 = ttime()
sed  = model.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
print(sed)
print(ttime() - t0)

with Profile() as profile:
    model.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CALLS)
        .print_stats()
    )



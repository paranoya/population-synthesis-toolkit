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

model = models.LogNormalQuenched_MFH(
    alpha=alpha, z_today=z_0 << u.dimensionless_unscaled,
    lnt0=np.log(7.0), scale=2.0, t_quench=10 << u.Gyr, tau_quench=1 << u.Gyr)

plt.figure()
plt.plot(dummy_t, model.integral_SFR(dummy_t))
plt.show()

n_tries = 1
print("PROFILING >>>\n\n")
t0 = ttime()
model.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
print(ttime() - t0)

with Profile() as profile:
    model.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CALLS)
        .print_stats()
    )

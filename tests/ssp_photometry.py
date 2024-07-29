from pst.observables import Filter
from pst.SSP import PyPopStar, BaseGM
from astropy import units as u
from astropy import constants
import numpy as np
from matplotlib import pyplot as plt
import os
from time import time

ssp = PyPopStar(IMF='KRO')
ssp_original = ssp.copy()

ssp.interpolate_sed(np.arange(900, 11000, 3) * u.angstrom)
#ssp = BaseGM()
t_in = time()
_ = ssp.copy()
t_end = time()
print("time spent copying the ssp: ", t_end - t_in)


def prepare_photometric_filters(filters):
    filters_out = []
    for f in filters:
        if os.path.exists(f):
            filters_out.append(Filter(filter_path=f))
        else:
            filters_out.append(Filter(filter_name=f))
    return filters_out

filters = prepare_photometric_filters(
    ['GALEX_FUV', 'GALEX_NUV', 'Euclid_VIS.vis', 'PANSTARRS_PS1.g',
     'PANSTARRS_PS1.r', 'JPLUS_J0660'])

t_in = time()
ssp.compute_photometry(filters)
t_end = time()
print("time spent generating ssp photometry: ", t_end - t_in)

plt.figure()
original_f_nu = ssp_original.L_lambda[1, 1] * u.Msun / 4 / np.pi / (10*u.pc)**2 * ssp_original.wavelength**2 / constants.c
f_nu = ssp.L_lambda[1, 1] * u.Msun / 4 / np.pi / (10*u.pc)**2 * ssp.wavelength**2 / constants.c
plt.plot(ssp_original.wavelength, original_f_nu.to('Jy'))
plt.plot(ssp.wavelength, f_nu.to('Jy'))
for photo, f in zip(ssp.photometry[:, 1, 1], ssp.photometry_filters):
    plt.plot(f.effective_wavelength(), photo, 'o')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500, 1e4)
plt.ylim(.6e3, 1e3)
plt.show()

#plt.figure()
#plt.imshow(photometry[0] - photometry[1], aspect='auto')
#plt.colorbar()
#plt.show()

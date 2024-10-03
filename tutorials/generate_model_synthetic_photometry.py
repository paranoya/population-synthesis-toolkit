"""
This tutorial shows how to generate a grid of photometric fluxes 
from a given SSP model.

The grid will have four dimensions:
- Observed redshift
- Dust attenuation
- Filter band
- SSP metallicity
- SSP age

The SSP photometric grid can then be combined with a given
star formation history to produce a grid of photometric fluxes
of for different SFHs.

"""

from matplotlib import pyplot as plt
import os
import numpy as np
from time import time
from astropy import units as u

from pst.SSP import PopStar
from pst.dust import DustScreen
from pst.observables import Filter, load_photometric_filters
from pst.models import LogNormalCEM

# Prepare the SSP photometry
filters = load_photometric_filters(
    ['PANSTARRS_PS1.g', 'PANSTARRS_PS1.r', 'PANSTARRS_PS1.i', 'PANSTARRS_PS1.z'])

# ssp = PopStar(IMF='cha')
ssp = PopStar(IMF='cha')
# Interpolate the SED to a coarser grid of wavelengths
ssp.interpolate_sed(np.arange(1000, 2e4, 5.0) * u.angstrom)

# Instanciate a dust model based on Cardelli+89
dust_model = DustScreen("ccm89")

# Create an array of values of A_V 
a_v_array = np.linspace(0, 2, 20)
redshift_array = np.linspace(0, 2.5, 30)

# Let's also estimate the required amount of time to build the grid
tstart = time()
all_photometry = np.zeros(
    (redshift_array.size, a_v_array.size, len(filters),
                           *ssp.L_lambda.shape[:-1]))

for i, z in enumerate(redshift_array):
    for j, av in enumerate(a_v_array):
        # For each value of AV, we create a new SSP model
        red_ssp = dust_model.redden_ssp_model(ssp, a_v=av)
        # Compute the SSP photometry at the observed redhisft z
        all_photometry[i, j] = red_ssp.compute_photometry(filter_list=filters,
                                                          z_obs=z)

tend = time()
print("time spent generating SSP photometry: ", tend - tstart)
print("time spent generating a single SSP model photometry: ",
      (tend - tstart) / (all_photometry.shape[0] * all_photometry.shape[1]))

# Instanciate a chemical evolution model

lnt_array = np.linspace(np.log(0.1), np.log(10), 10)
scale_array = np.linspace(0.1, 5, 20)

all_models_photometry = np.zeros(
    (lnt_array.size, scale_array.size, *all_photometry.shape[:-2]))

model = LogNormalCEM(alpha_powerlaw=0, ism_metallicity_today=0.02, lnt0=3,
                     scale=1, mass_today=1e9 * u.Msun)

# Benchmark the time spent creating the photometric grid
tstart = time()
for i, lnt in enumerate(lnt_array):
    for j, scale in enumerate(scale_array):
        model.lnt0 = lnt
        model.scale = scale

        all_models_photometry[i, j] = model.compute_photometry(
            ssp, t_obs=ssp.ages.max(), photometry=all_photometry)

tend = time()
print("time spent generating Model photometry: ", tend - tstart)

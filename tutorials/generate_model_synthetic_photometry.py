#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:20:31 2024

@author: pcorchoc
"""

from matplotlib import pyplot as plt
import os
import numpy as np
from time import time
from astropy import units as u

from pst.SSP import PopStar, XSL
from pst.dust import DustScreen
from pst.observables import Filter, load_photometric_filters
from pst.models import LogNormal_MFH

# Prepare the SSP photometry
filters = load_photometric_filters(
    ['CFHT_MegaCam.u',
     'PANSTARRS_PS1.g', 'PANSTARRS_PS1.r', 'PANSTARRS_PS1.i', 'PANSTARRS_PS1.z'])

# ssp = PopStar(IMF='cha')
ssp = XSL(IMF='Kroupa', ISO='P00')
ssp.interpolate_sed(np.arange(1000, 2e4, 5.0) * u.angstrom)

dust_model = DustScreen("ccm89")

a_v_array = np.linspace(0, 2, 20)
redshift_array = np.linspace(0, 2.5, 30)

tstart = time()
all_photometry = np.zeros((redshift_array.size, a_v_array.size, len(filters),
                           *ssp.L_lambda.shape[:-1]))

for i, z in enumerate(redshift_array):
    for j, av in enumerate(a_v_array):
        red_ssp = dust_model.redden_ssp_model(ssp, a_v=av)
        all_photometry[i, j] = red_ssp.compute_photometry(filter_list=filters, z_obs=z)

tend = time()
print("time spent generating SSP photometry: ", tend - tstart)
print("time spent generating a single SSP model: ", (tend - tstart) / (all_photometry.shape[0] * all_photometry.shape[1]))

# %%Compute the SFH model photometry
lnt_array = np.linspace(np.log(0.1), np.log(10), 10)
scale_array = np.linspace(0.1, 5, 20)

all_models_photometry = np.zeros(
    (lnt_array.size, scale_array.size, *all_photometry.shape[:-2]))

model = LogNormal_MFH(alpha=0, z_today=0.02, lnt0=3, scale=1, m_today=1e9 * u.Msun)
tstart = time()
for i, lnt in enumerate(lnt_array):
    for j, scale in enumerate(scale_array):
        model.lnt0 = lnt
        model.scale = scale
        masses = model.interpolate_ssp_masses(ssp, t_obs=ssp.ages.max())
        all_models_photometry[i, j] = model.compute_photometry(
            ssp, t_obs=ssp.ages.max(), photometry=all_photometry)

tend = time()
print("time spent generating Model photometry: ", tend - tstart)
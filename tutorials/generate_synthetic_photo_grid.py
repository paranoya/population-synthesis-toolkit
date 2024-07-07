#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:10:23 2024

@author: pcorchoc
"""

from matplotlib import pyplot as plt
import os
import numpy as np
from time import time

from pst.SSP import PopStar
from pst.dust import DustScreen
from pst.observables import Filter, load_photometric_filters


filters = load_photometric_filters(
    ['GALEX_FUV', 'GALEX_NUV', 'Euclid_VIS.vis', 'PANSTARRS_PS1.g',
     'PANSTARRS_PS1.r', 'JPLUS_J0660'])

ssp = PopStar(IMF='cha')
dust_model = DustScreen("ccm89")

a_v_array = np.linspace(0.1, 3, 30)
redshift_array = np.linspace(0, 2.5, 10)


tstart = time()
ssps = [dust_model.redden_ssp_model(ssp, a_v=av) for av in a_v_array]
all_photometry = np.zeros((redshift_array.size, a_v_array.size, len(filters),
                           *ssp.L_lambda.shape[:-1]))
for i, z in enumerate(redshift_array):
    for j, ssp in enumerate(ssps):
        all_photometry[i, j] = ssp.compute_photometry(filter_list=filters, z_obs=z)

tend = time()
print("time spent generating SSP photometry: ", tend - tstart)
print("time spent generating a single SSP model: ", (tend - tstart) / (all_photometry.shape[0] * all_photometry.shape[1]))

met_idx = 3
age_idx = 5
plt.figure()
plt.pcolormesh(a_v_array, redshift_array,
    - 2.5 * np.log10(all_photometry[:, :, 3, met_idx, age_idx] / all_photometry[:, :, 4, met_idx, age_idx]),
    cmap='jet')
plt.colorbar(label=f'PS g - PS r (Z={ssp.metallicities[met_idx]}, age={ssp.ages[age_idx]:.2f})')




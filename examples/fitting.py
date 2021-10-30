#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:34:18 2020

@author: D. Jiménez-López, Y. Ascasibar
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.io import fits

# You must install the pst package in ediatable/development form.
# Go to the pst root folder an type:
# `python -m pip install pst -e .`
#
import pst


# %% System settings

# Input and output direcotries

input_paths = {}
input_paths['Illustris'] = os.path.join('.', 'data', 'Illustris')
for dataset in input_paths:
    path = input_paths[dataset]
    if os.path.isdir(path) is False:
        raise FileNotFoundError('Cannot find input directory "{}"'.format(path))

output_paths = {}
output_paths['root'] = os.path.join('.', 'results')
output_paths['Illustris'] = os.path.join(output_paths['root'], 'Illustris')
for dataset in output_paths:
    path = output_paths[dataset]
    if os.path.isdir(path) is False:
        os.makedirs(path)
        print('Created "{}" directory'.format(path))


# %% Stellar population synthesys

# Observables

t0 = 13.7 * u.Gyr  # time of observation
obs_filters = ['u', 'g', 'r', 'i', 'z']

# SSPs

ssp = pst.SSP.PopStar(IMF="sal_0.15_100")


# %% Primordial polynomia

N_max = len(obs_filters)  # maximum polynomial degree
# negative_corrected = False  # Apply the negative-SFR correction?

poly_fits = []
for N in range(1, N_max+1):  # for every polynomial degree N up to N_max
    poly_fits.append(pst.fit.Polynomial_MFH_fit(N, ssp, obs_filters, t0))


# %% Read Illustris MFHs

real_models = {}

for filename in os.listdir(input_paths['Illustris']):
    if filename.endswith(".fits"):
        with fits.open(os.path.join(input_paths['Illustris'], filename)) as hdul:
            subhalo = filename[:-9]
            print(subhalo)

            lb_time = hdul[1].data['lookback_time'] * u.Gyr
            mass_formed = np.sum(hdul[3].data, axis=1) *u.Msun # sum over metallicities
            t_sorted = (t0-lb_time)[::-1]
            mfh_sorted = np.cumsum(mass_formed[::-1])

            model = pst.models.Tabular_MFH(t_sorted, mfh_sorted)
            real_models[subhalo] = model
    else:
        print('Skipping', filename)


# %% Generate mock observations

observed_lumonosities = {}

for model_name in real_models:
    model = real_models[model_name]
    real_sed = model.compute_SED(ssp, t0)

    L_Lsun = []
    for filter_name in obs_filters:
        photo = pst.observables.luminosity(
            flux=real_sed, wavelength=ssp.wavelength, filter_name=filter_name)
        L_Lsun.append(photo.integral_flux.to_value(u.Lsun))

    observed_lumonosities[model_name] = np.array(L_Lsun)


# %% Polynomial fit

polynomial_fits = {}

for target in observed_lumonosities:
    L_obs_Lsun = observed_lumonosities[target]

    for fit in poly_fits:
        polynomial_fits[target] = fit.fit(L_obs_Lsun)


# %% Plot MFH

            # plt.figure()
            # mean_sfr = (mfh_sorted[-1]-mfh_sorted[:-1]) / (t0-t_sorted[:-1])
            # plt.plot(t0-t_sorted[:-1], mean_sfr.to_value(u.Msun/u.yr), 'k-', label=subhalo)
            # t = t0-np.logspace(-3, np.log10(t0.to_value(u.Gyr)), 101)*u.Gyr
            # # plt.plot(t, model.integral_SFR(t), 'y-', label='table')
            # plt.xlabel('lookback time [Gyr]')
            # # plt.xscale('log')
            # plt.ylabel(r'$\langle\Psi\rangle$ [M$_\odot$/yr]')
            # peak = np.max(mean_sfr.to_value(u.Msun/u.yr))
            # plt.ylim(-.1*peak, 1.1*peak)
            # # plt.ylim(1e-3*peak, 1.1*peak)
            # # plt.yscale('log')
            # # plt.legend()
            # # plt.show()

            # # plt.figure()
            # # plt.plot(lb_time, (mfh_sorted[-1]-mfh_sorted)[::-1], 'k--')
            # # plt.xlabel('lookback time [Gyr]')
            # # plt.xscale('log')
            # # plt.yscale('log')
            # # plt.show()

            # # plt.figure()
            # # plt.plot(lb_time, mass_formed, label=filename)
            # # plt.plot(lb_time[:-1], np.diff(mfh_sorted)[::-1], label='model')
            # # plt.xlabel('lookback time [Gyr]')
            # # plt.xscale('log')
            # # plt.yscale('log')
            # # plt.legend()
            # # plt.show()



            # L_obs_Lsun =  dtype=u.Quantity)

        # mean_sfr_poly = (polynomial_fit.integral_SFR(t0)
                          # -polynomial_fit.integral_SFR(t)) / (t0-t)
        # plt.plot(t0-t, mean_sfr_poly.to_value(u.Msun/u.yr), ls='--',
                  # label='N = {}'.format(len(polynomial_fit.coeffs)-1))

            # plt.legend()
            # plt.show()


# %%                                                    ... Paranoy@ Rulz! ;^D

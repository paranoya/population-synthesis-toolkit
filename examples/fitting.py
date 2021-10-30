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

def output_subdir(parent, key):
    subdir = os.path.join(output_paths[parent], key)
    output_paths[key] = subdir
    if os.path.isdir(subdir) is False:
        os.makedirs(subdir)
        print('Created "{}" directory'.format(subdir))

output_subdir('root', 'Illustris')


# %% Stellar population synthesys


# Observables

t0 = 13.7 * u.Gyr  # time of observation
obs_filters = ['u', 'g', 'r', 'i', 'z']


# SSPs

ssp = pst.SSP.PopStar(IMF="sal_0.15_100")


# %% Primordial polynomia

N_max = len(obs_filters)  # maximum polynomial degree
# negative_corrected = False  # Apply the negative-SFR correction?

polynomial_bases = []
for N in range(1, N_max+1):  # for every polynomial degree N up to N_max
    polynomial_bases.append(pst.fit.Polynomial_MFH_fit(N, ssp, obs_filters, t0))


# %% Read Illustris models

real_models = {}

for filename in os.listdir(input_paths['Illustris']):
    if filename.endswith(".fits"):
        subhalo = filename[:-9]
        print(subhalo)
        real_models[subhalo] = pst.models.Tabular_Illustris(
            os.path.join(input_paths['Illustris'], filename), t0)
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

    polynomial_fits[target] = []
    for basis in polynomial_bases:
        polynomial_fits[target].append(basis.fit(L_obs_Lsun))


# %% Plots

def plot_result(savename, result, ylabel, logy=True, lookback_x=True):
    output_subdir('Illustris', savename)
    for model_name in real_models:
        plt.figure()
        plt.title(model_name)
        model = real_models[model_name]
        poly_fits = polynomial_fits[model_name]

        lookback_time = np.logspace(-3, np.log10(t0.to_value(u.Gyr)), 101)*u.Gyr
        t = t0 - lookback_time
        if lookback_x:
            x = lookback_time
            plt.xlabel('lookback time [Gyr]')
            plt.xscale('log')
        else:
            x = t
            plt.xlabel('t [Gyr]')

        y = result(model, t)
        ymax = y.max().value
        ymin = y.min().value
        plt.plot(x, y, 'k-', label='model')
        for i in range(N_max):
            y = result(poly_fits[i], t)
            plt.plot(x, y,
                     'b', alpha=(i+1.)/N_max, ls=(1+(i&1))*'-',
                     label='N={}'.format(i+1))

        plt.ylabel(ylabel)
        if logy:
            plt.ylim(max(1e-3*ymax, ymin), 1.1*ymax)
            plt.yscale('log')
        else:
            margin = .1*(ymax-ymin)
            plt.ylim(ymin-margin, ymax+margin)
        plt.legend()
        plt.savefig(os.path.join(
            output_paths[savename], '{}.png'.format(model_name)))
        plt.show()
        plt.close()


# %%
plot_result('mass', lambda model, t: model.integral_SFR(t),
            r'M [M$_\odot$]', logy=False, lookback_x=False)

# %%
plot_result('mean_SFR', lambda model, t:
            (model.integral_SFR(t0) - model.integral_SFR(t)) / (t0 -t),
            r'<SFR> [M$_\odot$/Gyr]')

# %%
plot_result('SFR', lambda model, t: model.SFR(t),
            r'SFR [M$_\odot$/Gyr]', logy=False) #, lookback_x=False)

# %%
plot_result('dot_SFR', lambda model, t: model.dot_SFR(t),
            r'd(SFR)/dt [M$_\odot$/Gyr^2]', logy=False) #, lookback_x=False)



# %%                                                    ... Paranoy@ Rulz! ;^D

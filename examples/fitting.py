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
# `python -m pip install -e .`
#
from pst import chemical_evolution as SPiCE_CEM
from pst import SSP as SPiCE_SSP
from pst import observables
from pst.fit import Polynomial_MFH_fit as Fit


# %% System settings

# Input and output direcotries

input_paths = {}
input_paths['Illustris'] = os.path.join('.', 'data', 'Illustris')
for dataset in input_paths:
    path = input_paths[dataset]
    if os.path.isdir(path) is False:
        raise FileNotFoundError('Cannot find input directory "{}"'.format(path))

output_paths = {}
output_paths['Illustris'] = os.path.join('.', 'results', 'Illustris')
for dataset in output_paths:
    path = output_paths[dataset]
    if os.path.isdir(path) is False:
        os.makedirs(path)
        print('Created "{}" directory'.format(path))


# Plots

figures = {}
panels = {}
plot_params = {}
# plot verbosity flags (e.g. show, save, show|save to do both, 0 to do nothing)
flags = {}
show = 1
save = 2

flags['sample_SSP'] = 0  # sample SED for a certain age and Z
flags['SED'] = 0  # model and fitted SED and luminosities
flags['Mass'] = 0  # mass formation histories
flags['mean_SFR'] = 0  # mean star formation rate over a lookback time
flags['SFR'] = 0  # star formation history
flags['dot_SFR'] = 0  # time derivative d(SFR)/dt
flags['ddot_SFR'] = 0  # second time derivative d^2(SFR)/dt^2
flags['distances'] = 0  # fit residuals

plot_params['sample_SSP'] = {'ages': [0, 10, 60, 80, 100], 'metallicities':[4]}


def individual_plots(keys):
    for key in keys:
        if flags[key] & (show|save):
            figures[key] = {}
            panels[key] = {}

individual_plots(['SED', 'Mass', 'mean_SFR', 'SFR', 'dot_SFR', 'ddot_SFR'])


# %% Observables

t0 = 13.7 * u.Gyr  # time of observation
obs_filters = ['u', 'g', 'r', 'i', 'z']


# %% Stellar population synthesys

SSP = SPiCE_SSP.PopStar(IMF="sal_0.15_100")

if flags['sample_SSP'] & (show|save):
    fig, ax = plt.subplots()
    ax.set_title('Some sample SSPs')
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\lambda L_\lambda$ [L$_\odot$/M$_\odot$]')
    ax.set_ylim(1e-6, 1e3)
    ax.set_yscale('log')
    for age_i in plot_params['sample_SSP']['ages']:
        for Z_i in plot_params['sample_SSP']['metallicities']:
            ax.plot(SSP.wavelength,
                    SSP.wavelength*SSP.SED[Z_i][age_i],
                    label='Z={}, {:.4f} Myr'.format(
                        SSP.metallicities[Z_i], SSP.ages[age_i].to_value(u.Myr)))
    ax.legend()
    if (flags['sample_SSP'] & show):
        fig.show()
        print('show')
    if (flags['sample_SSP'] & save):
        path_create(output_path)
        fig.savefig(os.path.join('sample_SSP.png'))
        fig.close()
        print('close')


# %% Primordial polynomia

N_max = len(obs_filters)  # maximum polynomial degree
# negative_corrected = False  # Apply the negative-SFR correction?

poly_fits = []
for N in range(1, N_max+1):  # for every polynomial degree N up to N_max
    poly_fits.append(Fit(N, SSP, obs_filters, t0))


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

            model = SPiCE_CEM.Tabular_MFH(t_sorted, mfh_sorted)
            real_models[subhalo] = model
    else:
        print('Skipping', filename)


# %% Generate mock observations

observed_lumonosities = {}

for model_name in real_models:
    model = real_models[model_name]
    real_sed = model.compute_SED(SSP, t0)

    if flags['SED'] & (show|save):
        fig, ax = plt.subplots()
        figures['SED'] = fig
        panels['SED'] = ax
        ax.xlabel(r'$\lambda$ [$\AA$]')
        ax.xlim(3000, 10000)
        ax.xscale('log')
        ax.ylabel(r'$\lambda L_\lambda\ [L_\odot]$')
        ax.yscale('log')
        ax.plot(SSP.wavelength, SSP.wavelength*real_sed, 'k-', label=model_name)

    L_Lsun = []
    for filter_name in obs_filters:
        photo = observables.luminosity(
            flux=real_sed, wavelength=SSP.wavelength, filter_name=filter_name)
        L_Lsun.append(photo.integral_flux.to_value(u.Lsun))

        if flags['SED'] & (show|save):
            l_eff = photo.effective_wavelength()
            norm = np.trapz(photo.filter_resp, photo.wl_filter)
            ax.plot(l_eff, L[-1]*l_eff/norm, 'ro')

    observed_lumonosities[model_name] = np.array(L_Lsun)


# %% Polynomial fit

polynomial_fits = {}

for target in observed_lumonosities:
    L_obs_Lsun = observed_lumonosities[target]

    for fit in poly_fits:
        polynomial_fits[target] = fit.fit(L_obs_Lsun)


# %%

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:43:07 2021

@author: pablo
"""

import prospect.io.read_results as reader
# import prospect.plotting as plotting
from matplotlib import pyplot as plt
import numpy as np

from prospect.models import transforms
from glob import glob
import h5py


def compute_pdf(param, weights, **kwargs):
    h, xedges = np.histogram(param, weights=weights, **kwargs)
    return h, xedges


def make_chain_plots(res):
    chains = res['chain']
    labels = res['theta_labels']
    n_chains = chains.shape[1]
    n_rows = np.sqrt(n_chains)
    if n_rows - int(n_rows) > 0:
        n_rows = int(n_rows) + 1
    else:
        n_rows = int(n_rows)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows, sharex=True,
                            figsize=(2*n_rows, 2*n_rows))
    for i in range(n_chains):
        ax = axs.flatten()[i]
        chain_i = chains[:, i]
        ax.plot(chain_i)
        if chain_i.max()/chain_i.min() > 1e3:
            ax.set_yscale('log')
        ax.set_ylabel(labels[i])
    plt.show()
    plt.close()


results = glob('prospector_fits/*h5')

table = np.loadtxt('all_TNG_mass_histories')
timesteps = table[:, -1]
lookbacktime = timesteps.max() - timesteps

galaxy_classification = np.loadtxt(
    'IllustrisTNG100_galaxy_bin_classification_subsample_fsps',
    usecols=(0, 1, 2), dtype=str)

prospector_sfhs = []
5
f = h5py.File('prospector_fits_illustris_tng.hdf5', 'w')

for galaxy in results:
    print(galaxy)
    # galaxy = 'prospector_fits/illustris_subhaloID_168390_continuity_sfh.h5'
    res, obs, model = reader.results_from(galaxy)
    entry = obs['entry']
    subhalo_id = obs['SubhaloID']
    best_params = res['bestfit']['parameter']
    lnprobability = res['lnprobability']
    # Values of the free params for each run
    chains = res['chain']
    # Weights corresponding to each run
    weights = res['weights']

    # Age bins used during the fit
    agebins = model.params['agebins']

    grp = f.create_group(str(subhalo_id))
    grp.create_dataset('agebins', data=agebins)

    masses = []
    sfrs = []
    for chain_i in range(chains.shape[0]):
        m = transforms.logsfr_ratios_to_masses(
            logmass=chains[chain_i, 0],
            logsfr_ratios=chains[chain_i, 1:],
            agebins=agebins)
        s = transforms.logsfr_ratios_to_sfrs(logmass=chains[chain_i, 0],
                                             logsfr_ratios=chains[chain_i, 1:],
                                             agebins=agebins)
        masses.append(m)
        sfrs.append(s)

    masses = np.array(masses)
    sfrs = np.array(sfrs)

    log_mass_bin_edges = np.linspace(3, 12, 51)
    log_mass_bins = (log_mass_bin_edges[1:]+log_mass_bin_edges[:-1])/2
    log_sfr_bin_edges = np.linspace(-5, 2, 51)
    log_sfr_bins = (log_sfr_bin_edges[1:]+log_sfr_bin_edges[:-1])/2
    mass_pdfs = []
    sfrs_pdfs = []
    for bin_i in range(masses.shape[1]):
        mass_h, _ = np.histogram(np.log10(masses[:, bin_i]),
                                 bins=log_mass_bin_edges, weights=weights,
                                 density=True)
        mass_pdfs.append(mass_h)
        sfr_h, _ = np.histogram(np.log10(sfrs[:, bin_i]),
                                bins=log_sfr_bin_edges, weights=weights,
                                density=True)
        sfrs_pdfs.append(sfr_h)
    mass_pdfs = np.array(mass_pdfs)
    sfrs_pdfs = np.array(sfrs_pdfs)

    mean_log_mass = np.sum(log_mass_bins[np.newaxis, :] * mass_pdfs *
                           np.diff(log_mass_bin_edges)[np.newaxis, :], axis=1)
    var_log_mass = np.sum((log_mass_bins[np.newaxis, :]
                           - mean_log_mass[:, np.newaxis])**2 * mass_pdfs
                          * np.diff(log_mass_bin_edges)[np.newaxis, :],
                          axis=1)
    sigma_log_mass = np.sqrt(var_log_mass)

    mean_log_sfr = np.sum(log_sfr_bins[np.newaxis, :] * sfrs_pdfs *
                          np.diff(log_sfr_bin_edges)[np.newaxis, :], axis=1)
    var_log_sfr = np.sum((log_sfr_bins[np.newaxis, :]
                          - mean_log_sfr[:, np.newaxis])**2 * sfrs_pdfs
                         * np.diff(log_sfr_bin_edges)[np.newaxis, :],
                         axis=1)
    sigma_log_sfr = np.sqrt(var_log_sfr)

    best_masses = transforms.logsfr_ratios_to_masses(
        logmass=best_params[0],
        logsfr_ratios=best_params[1:],
        agebins=agebins)
    best_sfh = transforms.logsfr_ratios_to_sfrs(
        logmass=best_params[0],
        logsfr_ratios=best_params[1:],
        agebins=agebins)
    best_log_sfr = np.log10(best_sfh)
    best_log_mass = np.log10(best_masses)

    bestfit_tot_mass = res['bestfit']['parameter'][0]
    mass_pdf, mass_binedges = compute_pdf(chains[:, 0], weights,
                                          bins=weights.size//50)
    mass_bins = (mass_binedges[1:]+mass_binedges[:-1])/2
    mean_logm = np.sum(chains[:, 0]*weights)
    sigma_logm = np.sqrt(np.sum((chains[:, 0]-mean_logm)**2 * weights))

    grp.create_dataset('mean_log_total_mass', data=mean_logm)
    grp.create_dataset('std_log_total_mass', data=sigma_logm)
    grp.create_dataset('best_log_total_mass', data=bestfit_tot_mass)
    # Photometry data
    best_photo = res['bestfit']['photometry']
    wl_photo = [obs['filters'][i].wave_effective for i in range(
                len(obs['filters']))]

    grp.create_dataset('mean_log_mass', data=mean_log_mass)
    grp.create_dataset('std_log_mass', data=sigma_log_mass)
    grp.create_dataset('mean_log_sfr', data=mean_log_sfr)
    grp.create_dataset('std_log_sfr', data=sigma_log_sfr)
    grp.create_dataset('best_log_mass', data=best_log_mass)
    grp.create_dataset('best_log_sfr', data=best_log_sfr)
    grp.create_dataset('run_time', data=res['sampling_duration'])
    agebins = 10**agebins / 1e9

    # Illustris data
    class_pos = np.where(galaxy_classification[:, 0] == str(subhalo_id))[0]
    mass_class = galaxy_classification[class_pos, 1].squeeze()
    ssfr_class = galaxy_classification[class_pos, 2].squeeze()

    cum_mass_tng = table[:, entry]

    sfh_tng_ori = np.diff(np.hstack((0, cum_mass_tng)))/np.diff(
        np.hstack((timesteps, 3*timesteps[-1]-2*timesteps[-2])))
    sfh_tng_ori /= 1e9
    log_sfh_tng = np.log10(sfh_tng_ori)

    sfh_tng = (np.interp(agebins[:, 1], lookbacktime, cum_mass_tng
                         ) - np.interp(agebins[:, 0], lookbacktime[::-1],
                                       cum_mass_tng[::-1])
               )/(np.diff(agebins, axis=1).flatten())
    sfh_tng /= 1e9

    cum_mass = np.cumsum(best_masses[::-1])[::-1]
    # Data for plotting
    cum_mass = np.vstack((cum_mass, cum_mass)).T.flatten()
    sfh = np.vstack((best_sfh, best_sfh)).T.flatten()
    mean_sfh = np.vstack((mean_log_sfr, mean_log_sfr)).T.flatten()
    std_sfh = np.vstack((sigma_log_sfr, sigma_log_sfr)).T.flatten()
    best_log_sfr = np.vstack((best_log_sfr, best_log_sfr)).T.flatten()

    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False)
    ax = axs[0]
    ax.set_title('{}, Mass class:{}, sSFR class:{}'.format(subhalo_id,
                                                           mass_class,
                                                           ssfr_class))
    ax.axvline(np.log10(cum_mass_tng[-1]), c='k', label='True value')
    ax.plot(mass_bins, mass_pdf, c='r')
    ax.set_ylabel(r'$\frac{dp(\log_{10}(M_*)}{d\log_{10}(M_*)}$')
    ax.set_xlabel(r'Present $\log_{10}(M_*/M_\odot)$', labelpad=1)
    ax.legend()
    # -------------------------------------------------------------------------
    ax = axs[1]
    ax.plot(np.log10(lookbacktime*1e9), log_sfh_tng, c='k',
            label='True value')
    # ax.plot(np.log10(agebins.flatten()*1e9), np.log10(sfh), c='r')
    ax.plot(np.log10(agebins.flatten()*1e9), mean_sfh, c='r')
    ax.plot(np.log10(agebins.flatten()*1e9), best_log_sfr, c='b')
    ax.fill_between(np.log10(agebins.flatten()*1e9),
                    mean_sfh-std_sfh, mean_sfh+std_sfh,
                    color='red', alpha=0.2)
    # ax.set_ylim(np.clip(np.log10(np.nanmin(sfh_tng_ori))*1.1, a_min=-2,
    #                     a_max=-1), np.log10(sfh_tng_ori.max())*1.1)
    ax.set_ylim(-3, 2)
    ax.set_xlim(6, 10.15)
    ax.set_xlabel('Lookback time (Gyr)')
    ax.set_ylabel(r'$SFR~(M_\odot/yr)$')
    ax.legend()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig('prospector_fits/plots/{}.png'.format(subhalo_id))
    plt.close()

f.close()

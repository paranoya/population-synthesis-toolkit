#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:43:07 2021

@author: pablo
"""

import prospect.io.read_results as reader
import prospect.plotting as plotting
from matplotlib import pyplot as plt
import numpy as np

from prospect.models import transforms
from glob import glob

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
    
galaxy_classification = np.loadtxt('IllustrisTNG100_galaxy_bin_classification_subsample_fsps',
                                   usecols=(0,1,2), dtype=str)
for galaxy in results:
    res, obs, model = reader.results_from(galaxy)
    
    # # Trace plots
    # tfig = reader.traceplot(res)
    # # Corner figure of posterior PDFs
    
    # labels_to_logify = res['theta_labels'][:7]
    
    # labels_to_logify.append('total_mass')
    
    entry = obs['entry']
    subhalo_id = obs['SubhaloID']
    best_params = res['bestfit']['parameter']
    agebins = model.params['agebins']
    masses = transforms.logsfr_ratios_to_masses(logmass=best_params[0],
                                                logsfr_ratios=best_params[1:],
                                                agebins=agebins)
    sfh = transforms.logsfr_ratios_to_sfrs(logmass=best_params[0],
                                           logsfr_ratios=best_params[1:],
                                           agebins=agebins)

    agebins = 10**agebins / 1e9
    # Illustris data
    class_pos = np.where(galaxy_classification[:, 0] == str(subhalo_id))[0]
    mass_class = galaxy_classification[class_pos, 1].squeeze()
    ssfr_class = galaxy_classification[class_pos, 2].squeeze()
    
    cum_mass_tng = table[:, entry]
    
    sfh_tng_ori = np.diff(np.hstack((0, cum_mass_tng)))/np.diff(
        np.hstack((timesteps, 3*timesteps[-1]-2*timesteps[-2])))
    sfh_tng_ori /= 1e9
    
    sfh_tng = (np.interp(agebins[:, 1], lookbacktime, cum_mass_tng
                         ) - np.interp(agebins[:, 0], lookbacktime[::-1],
                                       cum_mass_tng[::-1])
               )/(np.diff(agebins, axis=1).flatten())
    sfh_tng /= 1e9
    cum_mass = np.cumsum(masses[::-1])[::-1]
    cum_mass = np.vstack((cum_mass, cum_mass)).T.flatten()
    sfh = np.vstack((sfh, sfh)).T.flatten()


    best_photo = res['bestfit']['photometry']
    wl_photo = [obs['filters'][i].wave_effective for i in range(
                len(obs['filters']))]

    bestfit_masses = res['bestfit']['parameter'][:7]
    lnprobability = res['lnprobability']
    chains = res['chain']
    weights = res['weights']

    mass_pdf, mass_binedges = compute_pdf(chains[:, 0], weights,
                                          bins=weights.size//50)
    mass_bins = (mass_binedges[1:]+mass_binedges[:-1])/2
    mean_logm = np.sum(chains[:, 0]*weights)
    sigma_logm = np.sqrt(np.sum((chains[:, 0]-mean_logm)**2 * weights))

    bestfit_tot_mass = res['bestfit']['parameter'][-1]

    log_masses = np.sum(np.log10(chains[:, :7])*weights[:, np.newaxis], axis=0)
    np.log10(np.sum(10**log_masses))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False)
    ax = axs[0]
    ax.set_title('{}, Mass class:{}, sSFR class:{}'.format(subhalo_id, mass_class, ssfr_class))
    ax.axvline(np.log10(cum_mass_tng[-1]), c='k', label='True value')
    ax.plot(mass_bins, mass_pdf, c='r')
    ax.set_ylabel(r'$\frac{dp(\log_{10}(M_*)}{d\log_{10}(M_*)}$')
    ax.set_xlabel(r'Present $\log_{10}(M_*/M_\odot)$', labelpad=1)
    ax.legend()
    ax = axs[1]
    # ax.plot(np.mean(agebins, axis=1), sfh_tng, c='k')
    ax.plot(lookbacktime, sfh_tng_ori, c='k', label='True value')
    ax.plot(agebins.flatten(), sfh, c='r')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 2e1)
    ax.set_xlabel('Lookback time (Gyr)')
    ax.set_ylabel(r'$SFR~(M_\odot/yr)$')
    ax.legend()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig('prospector_fits/plots/{}.png'.format(subhalo_id))
    plt.close()
    # cfig = reader.subcorner(res)
    # plt.savefig('corner.pdf')

    # %%
    # plt.figure()
    # plt.plot(wl_photo, obs['maggies'], 'ko-')
    # plt.plot(wl_photo, best_photo, 'ro-')
    # plt.yscale('log')
    # plt.xscale('log')
    # labels = res['theta_labels']
    
    # bestparams = res['bestfit']['parameter']
        
    # make_chain_plots(res)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 20:38:43 2021

@author: pablo
"""

from astropy.io import fits
from astropy.table import Table
import numpy as np
from glob import glob


import sys
from pst import SSP
from astropy import units as u
import pyphot
lib = pyphot.get_library()
import random

filters = [lib[filter_i] for filter_i in ['SDSS_u', 'SDSS_g', 'SDSS_r',
                                          'SDSS_i', 'SDSS_z']]

# ssp_model = SSP.PopStar(IMF='kro_0.15_100')
ssp_model = SSP.FSPS()
model_wl = ssp_model.L_lambda[0, 0].spectral_axis
# %%
tng = '/home/pablo/ageing_diagram_models/IllustrisTNG-Galaxies/TNG100-1/SFHs/*'

files = glob(tng)

# Bins corresponding to QUENCHED, GREEN VALLEY and MAIN SEQUENCE
# It would be desirable to include starburst galaxies.
log_ssfr_bin_edges = np.array([-12.0, -11.0, -10.0, -9.0])
# Bins accounting for low-mass, MW-like and massive galaxies
logm_bin_edges = np.array([9, 10, 11, 12])

galaxy_class = {'Mass': {0: 'LOW', 1: 'MW', 2: 'MASSIVE'},
                'sSFR': {0: 'QG', 1: 'GV', 2: 'MS'}}

# Storing
sdss_photometry = []
sfhs = []
galaxy_classification = []
subhalos_ID = []

for file in files:
    subhalo_id = file[file.find('subhalo_')+8:file.find('_sfh')]
    subhalo_id = int(subhalo_id)
    print('SubhaloID: ', subhalo_id)

    hdul = fits.open(file)
    cat_info = hdul[0].header

    lookback_time = hdul[1].data['lookback_time']
    met_bins = hdul[2].data['metallicity']
    mass_formed = hdul[3].data
    try:
        mass = np.sum(mass_formed, axis=1)
    except:
        continue
    # Total stellar mass within two effective radii (Msun)
    stellar_mass = cat_info['massinrad_stars'] * 1e10

    if stellar_mass < 1e9:
        continue
    subhalos_ID.append(subhalo_id)
    # Total SFR within two effective radii (Msun/yr)
    sfr = cat_info['sfrinrad']

    ssfr = sfr/stellar_mass
    # Queched galaxies in illustris have ssfr=0 --> Set to last bin
    log_ssfr = np.log10(np.clip(ssfr, a_min=5e-12, a_max=1e-9))

    ssfr_pos = np.searchsorted(log_ssfr_bin_edges, log_ssfr)
    stellar_mass_pos = np.searchsorted(logm_bin_edges, np.log10(stellar_mass))

    subhalo_class = (galaxy_class['Mass'][stellar_mass_pos-1],
                     galaxy_class['sSFR'][ssfr_pos-1])
    galaxy_classification.append(subhalo_class)
    # Illustris provides some photometry but I do not know how it was computed
    # u_m = cat_info['stellarphotometrics_u']
    # g_m = cat_info['stellarphotometrics_g']
    # r_m = cat_info['stellarphotometrics_r']
    # i_m = cat_info['stellarphotometrics_i']
    # z_m = cat_info['stellarphotometrics_z']
    # sdss_photometry.append([subhalo_id, u_m, g_m, r_m, i_m, z_m])

    # Population synthesis

    mean_met = np.sum(mass_formed*met_bins[np.newaxis, :], axis=1)/mass
    mean_met[np.isnan(mean_met)] = 0
    mean_met[:] = 0.02
    cum_mass = np.cumsum(mass[::-1])

    sfhs.append(cum_mass)

    timesteps = (lookback_time.max()-lookback_time)*u.Gyr
    timesteps = timesteps[::-1]

    sed, weights = ssp_model.compute_SED(
        time=timesteps.to('yr'),
        mass=cum_mass*u.Msun,
        metallicity=mean_met)
    illustris_mags = []

    dist = 4 * np.pi * (10*u.pc)**2
    
    for _filter in filters:
        luminosity = _filter.get_flux(model_wl.value,
                                      sed.to(u.erg/u.s/u.angstrom).value).value
        # Absolute magnitude
        flux = luminosity * u.erg/(u.s * u.angstrom) / dist
        flux = flux.to(u.erg/(u.s * u.angstrom * u.cm**2))
        mag = -2.5 * np.log10(flux.value/_filter.AB_zero_flux.value)
        illustris_mags.append(mag)
    sdss_photometry.append(illustris_mags)

sdss_photometry = np.array(sdss_photometry)
sfhs.append(timesteps)
sfhs = np.array(sfhs)
galaxy_classification = np.array(galaxy_classification)
subhalos_ID = np.array(subhalos_ID, dtype=str)

# Save fotometry

data = np.vstack((subhalos_ID, galaxy_classification[:, 0],
                  galaxy_classification[:, 1],
                  ))
for band in range(sdss_photometry.shape[1]):
    data = np.vstack((data, sdss_photometry[:, band]))

np.savetxt('IllustrisTNG100_galaxy_bin_classification_all_fsps',
           data.T,
           header='SubhaloID   MASS_BIN_CLASS   SSFR_BIN_CLASS' +
           '   u_mag   g_mag   r_mag   i_mag   z_mag',
           fmt=('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s'))

# SELECTED SUBSAMPLE
selected_subsample_idx = np.array([], dtype=int)


def draw_random(array, n_samples=3):
    n_samples = min(len(array), n_samples)
    return np.array(random.choices(array, k=n_samples), dtype=int)


pos = np.where((data[1, :] == 'LOW') & (data[2, :] == 'QG'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'LOW') & (data[2, :] == 'GV'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'LOW') & (data[2, :] == 'MS'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))

pos = np.where((data[1, :] == 'MW') & (data[2, :] == 'QG'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'MW') & (data[2, :] == 'GV'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'MW') & (data[2, :] == 'MS'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))

pos = np.where((data[1, :] == 'MASSIVE') & (data[2, :] == 'QG'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'MASSIVE') & (data[2, :] == 'GV'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))
pos = np.where((data[1, :] == 'MASSIVE') & (data[2, :] == 'MS'))[0]
selected_subsample_idx = np.concatenate((selected_subsample_idx,
                                         draw_random(pos)))

selected_subsample_idx = np.array(selected_subsample_idx)
np.savetxt('IllustrisTNG100_galaxy_bin_classification_subsample_fsps',
           data[:, selected_subsample_idx].T,
           header='SubhaloID   MASS_BIN_CLASS   SSFR_BIN_CLASS' +
           '   u_mag   g_mag   r_mag   i_mag   z_mag',
           fmt=('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s'))

# Save sfhs

all_entries = np.hstack((np.array(subhalos_ID,
                                  dtype=str), 'time'))
header = ''
for i in range(all_entries.size):
    header += ' '+all_entries[i]
np.savetxt('all_TNG_mass_histories',
           np.vstack((sfhs[:], sfhs[-1, :])).T,
           header=header)

all_entries = np.hstack((np.array(subhalos_ID[selected_subsample_idx],
                                  dtype=str), 'time'))
header = ''
for i in range(all_entries.size):
    header += ' '+all_entries[i]
np.savetxt('subsample_TNG_mass_histories',
           np.vstack((sfhs[selected_subsample_idx, :], sfhs[-1, :])).T,
           header=header)


# np.savetxt('data/IllustrisTNG100_sdss_photo', sdss_photometry,
#            fmt=['%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f'],
#            header='SubhaloID u g r i z')
# np.savetxt('data/IllustrisTNG100_sfhs', sfhs,
#            header='Cumulative SFH (Msun)' +
#            '\n Last column corresponds to the lookback time bins')

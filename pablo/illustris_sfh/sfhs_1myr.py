#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:58:45 2021

@author: pablo
"""

import os
import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo

all_scale_factor = np.linspace(0.001, 1, 500)

all_redshift = 1/all_scale_factor -1
all_ages = []
today = cosmo.age(0).value
for z_i in all_redshift:
    all_ages.append(cosmo.age(z_i).value) #Gyr

f_ages = interp1d(all_scale_factor, today-all_ages)


basePath = '/media/pablo/Elements/TNG100-1/output'
fields = ['SubhaloMassInRadType','SubhaloSFRinRad', 'SubhaloLenType',
          'SubhaloCM', 'SubhaloPos', 'SubhaloFlag', 'SubhaloHalfmassRadType']

subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)
# il.snapshot.loadSubhalo(basePath, 99, 0, 0, fields=['Masses'])

masses_type = subhalos['SubhaloMassInRadType']
halfmass_rad_stars = subhalos['SubhaloHalfmassRadType'][:, 4]
n_part_type = subhalos['SubhaloLenType']
subhalos_cm = subhalos['SubhaloCM']
subhalos_pos = subhalos['SubhaloPos']
flags = subhalos['SubhaloFlag']
stellar_mass = masses_type[:, 4] * 1e10 / 0.70

selected_subhalos = np.where(stellar_mass > 1e9)[0]
# %%
plt.figure()
plt.hist(np.log10(stellar_mass), bins='auto', range=[6, 12], log=True)

plt.figure()
plt.scatter(np.log10(stellar_mass), n_part_type[:, 4])
plt.yscale('symlog')
plt.grid(b=True)
plt.show()
plt.close()
# %%
lookback_time_bin_edges = np.arange(0, 13701, 1)
subhalo_fields = ['GFM_InitialMass', 'GFM_Metallicity',
                  'GFM_StellarFormationTime', 'Coordinates']
f = h5py.File('IllustrisTNG100-1_SFH_1myr.hdf5', 'w')
# grp = f.create_group('age')
f.create_dataset('age', data=np.arange(0, 13700, 1))
for sub_id_i in selected_subhalos:
    print(sub_id_i)
    grp = f.create_group('sub_id_{}'.format(sub_id_i))
    try:
        subhalo = il.snapshot.loadSubhalo(basePath, snapNum=99,
                                          id=sub_id_i,
                                          partType=4, fields=subhalo_fields)
    except:
        continue
    subhalo_cm = subhalos_cm[sub_id_i]
    subhalo_pos = subhalos_pos[sub_id_i]
    subhalo_halfrad = halfmass_rad_stars[sub_id_i]
    subhalo_flag = flags[sub_id_i]
    coords = subhalo['Coordinates'] - subhalo_pos[np.newaxis, :]
    dist_to_cm = np.sum(coords**2, axis=1)
    intworad = dist_to_cm < (2*subhalo_halfrad)**2
    formation_time = subhalo['GFM_StellarFormationTime']
    not_wind = formation_time > 0
    intworad = intworad & not_wind
    masses = subhalo['GFM_InitialMass'][intworad]
    lookbacktime = f_ages(formation_time[intworad])
    metallicity = subhalo['GFM_Metallicity'][intworad]
    
    mass_history, _ = np.histogram(lookbacktime * 1e3,
                                bins=lookback_time_bin_edges,
                                weights=masses * 1e10/0.7)
    sfr_history = mass_history / 1e6
    # met_history = np.ones_like(sfr_history) 
    grp.create_dataset('flag', data=subhalo_flag)
    grp.create_dataset('mass_history', data=mass_history)
    grp.create_dataset('sfh', data=sfr_history)
    # grp.create_dataset('metallicity_history', data=met_history)
f.close()   

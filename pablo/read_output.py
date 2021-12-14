#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:23:24 2021

@author: pablo
"""

import h5py

data = h5py.File('prospector_fits_illustris_tng.hdf5', 'r')

subhalos_list = list(data.keys())
print('Subhalos ', subhalos_list)

# For each galaxy
galaxy_data = data[subhalos_list[0]]

print('Galaxy data: ', galaxy_data.keys())

# Getting the data
mean_log_sfr = galaxy_data['mean_log_sfr'][()]

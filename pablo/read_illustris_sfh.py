#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 19:32:00 2021

@author: pablo
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt

file = h5py.File('/home/pablo/population-synthesis-toolkit/pablo/illustris_sfh/IllustrisTNG100-1_SFH_1myr.hdf5', 'r')

galaxies = list(file.keys())
galaxies.pop(0)
ages = file['age'][()]
for galaxy in galaxies[:20]:
    sfh = file[galaxy]['sfh'][()]
    np.savetxt('illustris_sfh/individual_sfhs/' + galaxy, np.array([ages, sfh]).T)
    print(np.log10(np.sum(sfh * 1e6)))

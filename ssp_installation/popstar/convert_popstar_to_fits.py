#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:03 2024

@author: pcorchoc
"""

from astropy.io import fits
from astropy.table import Table
from astropy import units as u

import os
import numpy as np

# Path to the original files root directory (i.e. ROOT/IMF/SED/files...)
path = "."
# To ingest alternative IMF models, replace "cha" by the desired value e.g. "kro", "sal"
imf = "cha"

# Ages are read from one of the available "results table"
path_to_table = f"ta-{imf}-z0.0001.txt"
results_table = Table.read(path_to_table, format="ascii")

logage = results_table["logage"].value
ages = 10**logage
metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02,
                                       0.05])

wavelength = np.loadtxt(os.path.join(
            path, IMF, 'SED', f'spneb_{IMF}_0.15_100_z0500_t9.95'), dtype=float,
            skiprows=0, usecols=(0,), unpack=True) * u.Angstrom

primary = fits.PrimaryHDU()
hdul_list = [primary]
sed_table = Table(data=[wavelength], names=["wavelength"])
hdul_list.append(fits.BinTableHDU(sed_table, name="wavelength"))

for i, Z in enumerate(metallicities):
    sed_table = Table()
    for j, age in enumerate(logage):
        file = os.path.join(
            path, IMF, 'SED',
            'spneb_{0}_0.15_100_z{1:04.0f}_t{2:.2f}'.format(IMF, Z*1e4, age))
        star, total = np.loadtxt(
            file, dtype=float, skiprows=0, usecols=(1, 3),
            unpack=True)  # Lsun/Angstrom/Msun
        sed_table["logage_yr_{:02.2f}".format(age)
                  + "_stellar"] = star * u.Lsun / u.angstrom / u.Msun
        sed_table["logage_yr_{:02.2f}".format(age)
                  + "_total"] = total * u.Lsun / u.angstrom / u.Msun
    hdul_list.append(fits.BinTableHDU(sed_table,
                                      name="sed_z_0.{:04.0f}".format(Z*1e4))
                     )

# sed_table.write("popstar_cha_0.15_100.fits", overwrite=True)

hdul = fits.HDUList(hdul_list)
hdul.writeto("popstar_cha_0.15_100.fits.gz", overwrite=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:14:11 2024

@author: pcorchoc
"""

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import argparse


# KEY : [Z, X, Y, [Fe/H]] --> TODO: UPDATE X, Y and Fe/H
BC03_METALLICITIES = {
    # Padova1994
    'm22': [0.0001, 0.7696, 0.2303, -2.2490],
    'm32': [0.0004, 0.7696, 0.2303, -2.2490],
    'm42': [0.004, 0.7696, 0.2303, -2.2490],
    'm52': [0.008, 0.7696, 0.2303, -2.2490],
    'm62': [0.02, 0.7696, 0.2303, -2.2490],  # Zsun
    'm72': [0.05, 0.7696, 0.2303, -2.2490],
    'm82': [0.1, 0.7696, 0.2303, -2.2490],
    # Padova2000
    'm122': [0.0004, 0.7696, 0.2303, -2.2490],
    'm132': [0.001, 0.7696, 0.2303, -2.2490],
    'm142': [0.004, 0.7696, 0.2303, -2.2490],
    'm152': [0.008, 0.7696, 0.2303, -2.2490],
    'm162': [0.019, 0.7696, 0.2303, -2.2490],  # Zsun
    'm172': [0.03, 0.7696, 0.2303, -2.2490],
                      }

def load_ages(filename):
    color_table = Table.read(filename.replace('.all', '.1color'), format='ascii')
    return color_table['col1'].data 

def make_table(filename):
    # Get the array of ages
    log_ages_yr = load_ages(filename)
    # Parse the metallicity
    metal_kw = os.path.basename(filename).split("_")[-3]
    metallicity = BC03_METALLICITIES[metal_kw]
    # print(metal_kw)
    # Get the data
    data = np.loadtxt(filename)
    wavelength = data[:, 0]
    table = Table(data=[wavelength], names=['wavelength'])

    for ith, lage in enumerate(log_ages_yr):
        table[f'log_age_yr_{lage:2.4f}'] = data[:, ith + 1]
    # print(table)
    
    primary = fits.PrimaryHDU()
    primary.header['Z'] = metallicity[0]
    hdul = fits.HDUList([primary, fits.BinTableHDU(table)])
    hdul_filename = filename.replace('.all', '.fits')
    print(f"Saving data as HDUL file at {hdul_filename}")
    hdul.writeto(hdul_filename, overwrite=True)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('filename')           # positional argument
    # parser.add_argument('output')           # positional argument

    args = parser.parse_args()
    # filename = os.path.join(args.output, args.filename)
    filename = args.filename
    filename = filename.replace('ised', 'all')
    print(f"Converting {args.filename} to FITS file")
    make_table(filename)

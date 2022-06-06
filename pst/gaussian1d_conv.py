#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:18:39 2022

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt

def gaussian1d_conv(f, sigma, deltax):
    """blah..."""
    sigma_pixels = sigma / deltax
    pix_range = np.arange(0, f.size, 1)
    XX = pix_range[:, np.newaxis] - pix_range[np.newaxis, :]
    g = np.exp(- (XX)**2 / 2 / sigma_pixels[np.newaxis, :]**2) / (
               sigma_pixels[np.newaxis, :] * np.sqrt(2 * np.pi))
    g /= g.sum(axis=1)[:, np.newaxis]

    f_convolved = np.sum(f[np.newaxis, :] * g, axis=1)
    return f_convolved


if __name__ == '__main__':
    x = np.linspace(1, 100, 100)
    old_sigma = 1
    f = 1 / old_sigma / np.sqrt(2 * np.pi) * np.exp(- (x - 30)**2
                                                    / 2 / old_sigma**2)
    sigma = 3 * np.ones(f.size)
    # sigma = np.linspace(1, 30, f.size)
    new_sigma = np.sqrt(3**2 + old_sigma**2)
    new_gauss = 1 / new_sigma / np.sqrt(2 * np.pi) * np.exp(- (x - 30)**2
                                                            / 2 / new_sigma**2)
    conv_f = gaussian1d_conv(f, sigma, deltax=1)
    plt.figure()
    plt.plot(f)
    plt.plot(new_gauss, lw=1, c='k')
    plt.plot(conv_f, '--')
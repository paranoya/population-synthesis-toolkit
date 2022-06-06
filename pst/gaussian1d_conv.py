#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:18:39 2022

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt


def gaussian1d_conv(f, sigma, deltax):
    """Apply a gaussian convolution to a 1D array f(x).

    params
    ------
    - f: (array) 1D array containing the data to be convolved with.
    - sigma (array) 1D array containing the values of sigma at each value of x
    - deltax: (float) Step size of x in "physical" units.
    """
    sigma_pixels = sigma / deltax
    pix_range = np.arange(0, f.size, 1)
    if len(pix_range) < 2e4:
        XX = pix_range[:, np.newaxis] - pix_range[np.newaxis, :]
        g = np.exp(- (XX)**2 / 2 / sigma_pixels[np.newaxis, :]**2) / (
                   sigma_pixels[np.newaxis, :] * np.sqrt(2 * np.pi))
        g /= g.sum(axis=1)[:, np.newaxis]
        f_convolved = np.sum(f[np.newaxis, :] * g, axis=1)
    else:
        print(' WARNING: TOO LARGE ARRAY --- APPLYING SLOW CONVOLUTION METHOD ---')
        f_convolved = np.zeros_like(f)
        for pixel in pix_range:
            XX = pixel - pix_range
            g = np.exp(- (XX)**2 / 2 / sigma_pixels**2) / (
                       sigma_pixels * np.sqrt(2 * np.pi))
            g /= g.sum()
            f_convolved[pixel] = np.sum(f * g)
    return f_convolved


if __name__ == '__main__':
    # Small test
    from ppxf.ppxf_util import gaussian_filter1d

    x = np.linspace(1, 100, 10000)
    old_sigma = 1
    f = 1 / old_sigma / np.sqrt(2 * np.pi) * np.exp(- (x - 30)**2
                                                    / 2 / old_sigma**2)
    sigma = 3 * np.ones(f.size)
    # sigma = np.linspace(1, 30, f.size)
    new_sigma = np.sqrt(3**2 + old_sigma**2)
    new_gauss = 1 / new_sigma / np.sqrt(2 * np.pi) * np.exp(- (x - 30)**2
                                                            / 2 / new_sigma**2)
    conv_f = gaussian1d_conv(f, sigma, deltax=1)
    ppxf_conv_f = gaussian_filter1d(f, sig=sigma)
    plt.figure()
    plt.plot(f, c='r')
    plt.plot(new_gauss, lw=1, c='k')
    plt.plot(conv_f, '--', c='b')
    plt.plot(ppxf_conv_f, '--', c='g')
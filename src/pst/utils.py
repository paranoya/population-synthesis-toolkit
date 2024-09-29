"""
Module containing generic utility functions
"""

import numpy as np
from astropy import units as u


SQRT_2 = np.sqrt(2)

def flux_conserving_interpolation(new_wave, wave, spectra):
    """Interpolate a spectra to a new grid of wavelengths preserving the flux density.
    
    Parameters
    ----------
    new_wave : np.ndarray
        New grid of wavelengths
    wave : np.ndarray
        Original grid of wavelengths
    spectra : np.ndarray
        Spectra associated to `wave`.
    
    Returns
    -------
    interp_spectra : np.ndarray
        Interpolated spectra to `new_wave`
    """
    wave_limits = 1.5 * wave[[0, -1]] - 0.5 * wave[[1, -2]]
    wave_edges = np.hstack([wave_limits[0], (wave[1:] + wave[:-1])/2, wave_limits[1]])

    new_wave_limits = 1.5 * new_wave[[0, -1]] - 0.5 * new_wave[[1, -2]]
    new_wave_edges = np.hstack([new_wave_limits[0], (new_wave[1:] + new_wave[:-1])/2, new_wave_limits[1]])
    cumulative_spectra = np.cumsum(spectra * np.diff(wave_edges))
    cumulative_spectra = np.insert(cumulative_spectra, 0, 0)
    new_cumulative_spectra = np.interp(new_wave_edges, wave_edges, cumulative_spectra)
    interp_spectra = np.diff(new_cumulative_spectra) / np.diff(new_wave_edges)
    return interp_spectra

def gaussian1d_conv(f, sigma, deltax):
    """Apply a gaussian convolution to a 1D array f(x).

    params
    ------
    f : np.array
        1D array containing the data to be convolved with.
    sigma : np.array
        1D array containing the values of sigma at each value of x
    deltax : float
        Step size of x in "physical" units.
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

def check_unit(quantity, default_unit=None):
    """Check the units of an input quantity.
    
    Parameters
    ----------
    quantity : np.ndarray or astropy.units.Quantity
        Input quantity.
    default_unit : astropy.units.Quantity, default=None
        If `quantity` has not units, it corresponds to the unit assigned to it.
        Otherwise, it is used to check the equivalency with `quantity`.
    """
    isq = isinstance(quantity, u.Quantity)
    if isq and default_unit is not None:
        if not quantity.unit.is_equivalent(default_unit):
            raise u.UnitTypeError(
                "Input quantity does not have the appropriate units")
        else:
            return quantity
    elif not isq and default_unit is not None:
        return quantity * default_unit
    elif not isq and default_unit is None:
        raise ValueError("Input value must be a astropy.units.Quantity")
    else:
        return quantity


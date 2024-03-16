#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:56:46 2018

@author: pablo
"""

import numpy as np
import os
from astropy import units as u
from astropy import constants

from scipy import interpolate

from specutils.manipulation import FluxConservingResampler

# =============================================================================
class Observable(object):
# =============================================================================
    pass

# =============================================================================
class Spectrum(Observable):
# =============================================================================
    """
    
    """
    def __init__(self, central_wavelenghts):
        self.wavelengths = central_wavelenghts
        # self.wavelength_bins = np.array([
        #     (3*self.wavelengths[0]-self.wavelengths[1])/2,
        #     (self.wavelengths[1:]+self.wavelengths[:-1])/2,
        #     (3*self.wavelengths[-1]-self.wavelengths[-2])/2
        #     ])
        #TODO: Different resamplers
        self.resampler = FluxConservingResampler()
    
    def from_Spectrum1D(self, spectrum):
        return self.resampler(spectrum, self.wavelengths)        
        
# Helper functions for finding photometric filters.

def list_of_available_filters():
    filter_dir = os.path.join(os.path.dirname(__file__),
                              "data", "filters")
    print(f"Checking filters available at {filter_dir}")
    return os.listdir(filter_dir)

def find_filt_from_name(name):
    filters = list_of_available_filters()
    for f in filters:
        if name.lower() in f.strip(".dat").lower():
            return os.path.join(os.path.dirname(__file__),
                              "data", "filters", f)
    return None


class Filter(object):
    """This class represent a photometric filter
    
    Attributes
    ----------
    - trans_curve: (np.ndarray)
        Photometric transmission curve.
    - wavelength: (np.ndarray)
        Wavelength vector used for interpolating the filter.
    - filter_wavelength: (np.ndarray)
        Original wavelength vector of the filter.
    - filter_resp: (np.ndarray)
        Original filter response function.
    """

    def __init__(self, **kwargs):

        """This class provides a filter (SDSS, WISE, GALEX, 2MASS photometry) with the same
        number of points as the given wavelength array.

        The wavelength UNITS are by default expressed in AA"""
        print("Initialising Filter variables")

        self.wavelength = kwargs.get('wavelength', None)
        if self.wavelength is not None:
            if not hasattr(self.wavelength, "unit"):
                print("Assuming that input wavelength array is in angstrom")
                self.wavelength *= u.angstrom
            if not (self.wavelength[1:] > self.wavelength[:-1]).all():
                raise NameError('Wavelength array must be crescent')

        self.filter_wavelength = kwargs.get('filter_wavelength', None)
        if self.filter_wavelength is not None and not hasattr(
            self.filter_wavelength, "unit"):
            print("Assuming that input filter wavelength array is in angstrom")
            self.filter_wavelength *= u.angstrom

        self.filter_resp = kwargs.get('filter_resp', None)
        
        self.filter_path = kwargs.get('filter_path', None)
        self.filter_name = kwargs.get('filter_name', None)
        if self.filter_path is not None:
            self.load_filter(path=self.filter_path)
        elif self.filter_name is not None:
            self.load_filter(name=self.filter_name)

        if self.wavelength is not None:
            self.interpolate(self.wavelength)

    def load_filter(self, path=None, name=None):
        """Load a filter from a text file.
        
        Parameters
        ----------
        - path: (str)
            Path to text file.
        - name: (str)
            Name of the filter to be matched using the set of filters available.
            #TODO: where?
        """
        if path is not None:
            self.filter_wavelength, self.filter_resp  = np.loadtxt(
                path, usecols=(0, 1), unpack=True)
            self.filter_path = path
        elif name is not None:
            path = find_filt_from_name(name)
            # Update the name
            self.filter_name = name
            self.filter_path = path
            if path is not None:
                self.filter_wavelength, self.filter_resp = np.loadtxt(
                    path, usecols=(0, 1), unpack=True)
            else:
                raise NameError(f"No filter found with input name {name}")
        else:
            raise NameError("No path, nor name provided")
        print(f"Filter loaded from: {path}")
        self.filter_wavelength *= u.Angstrom
        return self.filter_wavelength, self.filter_resp 

    def effective_wavelength(self):
        return np.sum(self.filter_wavelength*self.filter_resp)/np.sum(self.filter_resp)

    def effective_bandwidth(self):
        return np.sqrt(8*np.log(2)*(
            np.sum(self.filter_wavelength**2*self.filter_resp)/np.sum(self.filter_resp)
            - self.effective_wavelength()**2))

    def effective_transmission(self):
        return np.sum(self.filter_resp**2)/np.sum(self.filter_resp)

    def interpolate(self, wavelength=None):
        """Interpolate a filter response curve to an input wavelength."""
        if not hasattr(wavelength, "unit"):
            print("Assuming that input wavelength array is in angstrom")
            wavelength *= u.angstrom

        delta_wl = wavelength[1:] - wavelength[:-1]
        wavelength_edges = np.zeros((wavelength.size + 1)
                                    ) * wavelength.unit
        wavelength_edges[1:-1] = wavelength[1:] - delta_wl / 2
        wavelength_edges[0] = wavelength[0] - delta_wl[0] / 2
        wavelength_edges[-1] = wavelength[-1] + delta_wl[-1] / 2
        
        cumulative_trans_curve = np.cumsum(self.filter_resp)
        interp_cum_trans_curve = np.interp(wavelength_edges,
                                           self.filter_wavelength,
                                           cumulative_trans_curve)

        self.response = np.diff(interp_cum_trans_curve)
        self.wavelength= wavelength
        print("Filter transmission curve interpolated to input wavelength array")
        return self.response


class TopHatFilter(Filter):
    """Top hat photometric filter"""
    def __init__(self, central_wave, width, **kwargs):
        if not hasattr(central_wave, 'unit'):
            print("Assuming that input central wavelength is expressed in angstrom")
            central_wave *= u.Angstrom
        if not hasattr(width, 'unit'):
            print("Assuming that input width is expressed in angstrom")
            width *= u.Angstrom

        self.wavelength = kwargs.get('wavelength', None)
        if not hasattr(self.wavelength, "unit"):
            print("Assuming that input wavelength array is in angstrom")
            self.wavelength *= u.angstrom
        if not (self.wavelength[1:] > self.wavelength[:-1]).all():
            raise NameError('Wavelength array must be crescent')
        
        if self.wavelength is None:
            self.filter_wavelength = np.linspace(central_wave - width,
                                    central_wave + width,
                                    50)
        else:
            self.filter_wavelength = self.wavelength.copy()

        self.filter_resp = np.ones(self.filter_wavelength.size)
        self.filter_resp[self.filter_wavelength < central_wave - width / 2] = 0
        self.filter_resp[self.filter_wavelength > central_wave + width / 2] = 0
        if self.wavelength is None:
            self.response = self.filter_resp.copy()


# =============================================================================
class Luminosity(Observable, Filter):
# =============================================================================
    """This module computes the photmetric luminosity (power) on a given band
    for a given specific flux (per unit wavelength)."""

    def __init__(self, **kwargs):
        '''
        Filter.__init__(self, **kwargs)
        ## Flux counts
        self.flux = kwargs['flux']
        self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   # flux at 10 pc.
        self.nu = u.c/( self.wavelength)

        diff_nu = - np.ediff1d(np.insert(self.nu, 0, 2*self.nu[0]-self.nu[1]))

        self.integral_flux = np.nansum((self.flux/self.nu * self.filter * diff_nu) )
        '''

        Filter.__init__(self, **kwargs)
        self.flux = kwargs['flux']

        self.integral_flux = np.trapz(self.flux*self.filter, self.wavelength)
        # self.integral_flux = np.trapz(self.flux*self.filter*self.wavelength,
        #                               np.log(self.wavelength/u.Angstrom))


# =============================================================================
class Magnitude(Observable, Filter):
# =============================================================================
    """Class to compute synthetic photometry from spectra."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.wavelength is not None:
            self.nu = constants.c /( self.wavelength)

    def check_spectra(self, spectra):
        if not hasattr(spectra, "unit"):
            print("Assuming that input spectra is expressed in Lsun/AA")
            spectra =  spectra.copy() * u.Lsun / u.angstrom
        else:
            return spectra

    def get_photons(self, spectra, spectra_err=None):
        spectra = self.check_spectra(spectra)
        spectra_err = self.check_spectra(spectra_err)

        photon_flux = np.trapz(
            spectra / (constants.h * constants.c / self.wavelength
                                   ) * self.response,
            x=self.wavelength)
        if spectra_err is not None:
            photon_flux_err = np.trapz(
                spectra_err / (constants.h * constants.c / self.wavelength
                                       ) * self.response,
                x=self.wavelength)
        else:
            photon_flux_err = None
        return photon_flux, photon_flux_err
    
    def get_ab(self, spectra, spectra_err=None):
        n_photons, n_photons_err = self.get_photons(spectra, spectra_err)
        norm_photons, _ = self.get_photons(
            3631 * u.Jy * np.ones(spectra.size) * constants.c / self.wave**2)
        m_ab = - 2.5 * np.log10(n_photons / norm_photons)
        if n_photons_err is None:
            m_ab_err = None
        else:
            m_ab_err = - 2.5 * np.log10(n_photons_err / norm_photons)
        return m_ab, m_ab_err

    def get_flux(self, spectra, spectra_err):
        m_ab, m_ab_err = self.get_AB(spectra, spectra_err)
        f_nu = 10**(-0.4 * m_ab) * 3631 * u.Jy
        f_nu_err = 10**(-0.4 * m_ab_err) * u.Jy
        nu_f_nu = f_nu * constants.c / self.effective_wavelength()
        nu_f_nu_err = f_nu_err * constants.c / self.effective_wavelength()
        return nu_f_nu, nu_f_nu_err

if __name__ == '__main__':
    Magnitude(filter_name='r')
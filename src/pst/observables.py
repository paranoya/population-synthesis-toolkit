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

from matplotlib import pyplot as plt

from . import utils

def list_of_available_filters():
    filter_dir = os.path.join(os.path.dirname(__file__),
                              "data", "filters")
#    print(f"Checking filters available at {filter_dir}")
    return os.listdir(filter_dir)

def find_filt_from_name(name):
    filters = list_of_available_filters()
    for f in filters:
        if name.lower() in f.strip(".dat").lower():
            return os.path.join(os.path.dirname(__file__),
                              "data", "filters", f)
    return None

def load_photometric_filters(filters):
    """Convenience function for constructing a list of photometric filters.
    
    Parameters
    ----------
    - filters: list of str
        List of filters to load. The list might contain the absolute path to
        a filter response file, or just the filter name following the SVO
        convention.
    
    Returns
    -------
    - filters_out: list of pst.observables.Filter
        List of filters.
    """
    filters_out = []
    for f in filters:
        if os.path.exists(f):
            filters_out.append(Filter(filter_path=f))
        else:
            filters_out.append(Filter(filter_name=f))
    return filters_out


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
#        print("Initialising Filter variables")

        self.wavelength = kwargs.get('wavelength', None)
        if self.wavelength is not None:
            if not hasattr(self.wavelength, "unit"):
#                print("Assuming that input wavelength array is in angstrom")
                self.wavelength *= u.angstrom
            if not (self.wavelength[1:] > self.wavelength[:-1]).all():
                raise NameError('Wavelength array must be crescent')

        self.filter_wavelength = kwargs.get('filter_wavelength', None)
        if self.filter_wavelength is not None and not hasattr(
            self.filter_wavelength, "unit"):
#            print("Assuming that input filter wavelength array is in angstrom")
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

        if self.wavelength is not None:
            self.nu = constants.c /( self.wavelength)

    def load_filter(self, path=None, name=None, wl_unit=u.angstrom):
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
#        print(f"Filter loaded from: {path}")
        self.filter_wavelength *= wl_unit
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
            wavelength = wavelength << u.angstrom
        self.response = utils.flux_conserving_interpolation(
            wavelength, self.filter_wavelength, self.filter_resp)
        self.wavelength= wavelength
        return self.response

    def check_spectra(self, spectra):
        if spectra is not None and not hasattr(spectra, "unit"):
#            print("Assuming that input spectra is expressed in Lsun/cm2/AA")
            spectra =  spectra.copy() * u.Lsun / u.angstrom / u.cm**2
        return spectra

    def get_photons(self, spectra, spectra_err=None, mask_nan=True):
        spectra = self.check_spectra(spectra)
        spectra_err = self.check_spectra(spectra_err)
        if mask_nan:
            mask = np.isfinite(spectra)
            photon_flux = np.trapz(
                spectra[mask] / (constants.h * constants.c / self.wavelength[mask]
                                   ) * self.response[mask],
                x=self.wavelength[mask])
        else:
            photon_flux = np.trapz(
                spectra / (constants.h * constants.c / self.wavelength
                                   ) * self.response,
                x=self.wavelength)

        if spectra_err is not None:
            if mask_nan:
                mask = mask & np.isfinite(spectra_err)
            else:
                mask = np.ones_like(spectra_err, dtype=bool)

            photon_flux_err = np.trapz(
                spectra_err[mask] / (constants.h * constants.c / self.wavelength[mask]
                                       ) * self.response[mask],
                x=self.wavelength[mask])
        else:
            photon_flux_err = None
        return photon_flux, photon_flux_err
    
    def get_ab(self, spectra, spectra_err=None, mask_nan=True):
        n_photons, n_photons_err = self.get_photons(spectra, spectra_err, mask_nan=mask_nan)
        norm_photons, _ = self.get_photons(
            3630.781 * u.Jy * np.ones(spectra.shape) * constants.c / self.wavelength**2,
            mask_nan=False)
        m_ab = - 2.5 * np.log10(n_photons / norm_photons)
        if n_photons_err is None:
            m_ab_err = None
        else:
            m_ab_err = 2.5 / np.log(10) * n_photons_err / n_photons
        return m_ab, m_ab_err

    def get_fnu(self, spectra, spectra_err=None, mask_nan=True):
        """Compute the  specific flux per frequency unit from a spectra."""
        n_photons, n_photons_err = self.get_photons(spectra, spectra_err, mask_nan=mask_nan)
        norm_photons, _ = self.get_photons(
            3630.781 * u.Jy * np.ones(spectra.shape) * constants.c / self.wavelength**2,
            mask_nan=False)
        f_nu = n_photons / norm_photons * 3630.781 * u.Jy
        # f_nu = f_nu.to('Jy')
        if spectra_err is None:
            f_nu_err = None
        else:
            f_nu_err = n_photons_err / norm_photons * 3630.781 * u.Jy
            # f_nu_err = f_nu_err.to('Jy')
        return f_nu, f_nu_err

    def get_flambda_vegamag(self, spectra, spectra_err=None, mask_nan=True):
        spectra = self.check_spectra(spectra)
        spectra_err = self.check_spectra(spectra_err)
        if mask_nan:
            mask = np.isfinite(spectra)
        else:
            mask = np.ones_like(spectra, dtype=bool)

        av_f_lambda = np.trapz(spectra[mask] * self.wavelength[mask] * self.response[mask], x=self.wavelength[mask]
                               ) / np.trapz(self.response[mask] * self.wavelength[mask], x=self.wavelength[mask])

        if spectra_err is not None:
            if mask_nan:
                mask = mask & np.isfinite(spectra_err)
            else:
                mask = np.ones_like(spectra_err, dtype=bool)

            av_f_lambda_err = np.trapz(
            spectra_err[mask] / (constants.h * constants.c / self.wavelength[mask]
                                   ) * self.response[mask],
            x=self.wavelength[mask])
        else:
            av_f_lambda_err = None

        return av_f_lambda, av_f_lambda_err

    def plot(self):
        """Plot the filter response curve."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(self.filter_wavelength, self.filter_resp, label='Original',
                color='k', where="mid")
        ax.plot(self.filter_wavelength, self.filter_resp, '.', color='k')
        ax.set_xlabel(f"Wavelength ({self.filter_wavelength.unit})")
        ax.set_ylabel("Filter response")
        if self.wavelength is not None:
            ax.step(self.wavelength, self.response, label='Interpolated',
                      color='r', where="mid")
        ax.legend()
        return fig

class TopHatFilter(Filter):
    """Top hat photometric filter"""
    def __init__(self, central_wave, width, **kwargs):
        if not hasattr(central_wave, 'unit'):
#            print("Assuming that input central wavelength is expressed in angstrom")
            central_wave *= u.Angstrom
        if not hasattr(width, 'unit'):
#            print("Assuming that input width is expressed in angstrom")
            width *= u.Angstrom

        self.wavelength = kwargs.get('wavelength', None)
        if not hasattr(self.wavelength, "unit"):
#            print("Assuming that input wavelength array is in angstrom")
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

if __name__ == '__main__':
    from pst.SSP import BaseGM
    import matplotlib.pyplot as plt
    ssp = BaseGM()
    
    photometry = Filter(filter_name='r', wavelength=ssp.wavelength)

    for sed in ssp.L_lambda.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2])):
        sed = 1 * u.Msun * sed / 4 / np.pi / (10 * u.pc)**2
        mag, mag_err = photometry.get_ab(sed)
        

#        print("SSP absolute magnitude: ", mag)
    plt.figure()
    plt.plot(ssp.wavelength, photometry.response)
    plt.plot(ssp.wavelength, sed / np.mean(sed))
    plt.yscale('log')
    plt.show()

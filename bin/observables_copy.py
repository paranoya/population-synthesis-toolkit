#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:56:46 2018

@author: pablo
"""

import numpy as np
import os
from pst import root_path

from astropy import units as u
from astropy import constants
#from specutils.manipulation import FluxConservingResampler


class Observable(object):
    pass


# class Spectrum(Observable):
#     """
#     This class computes a spectrum given the spectral axis points.
#     """

#     def __init__(self, central_wavelenghts):
#         self.wavelengths = central_wavelenghts
#         # self.wavelength_bins = np.array([
#         #     (3*self.wavelengths[0]-self.wavelengths[1])/2,
#         #     (self.wavelengths[1:]+self.wavelengths[:-1])/2,
#         #     (3*self.wavelengths[-1]-self.wavelengths[-2])/2
#         #     ])
#         # TODO: Different resamplers
#  #       self.resampler = FluxConservingResampler()

#     def from_Spectrum1D(self, spectrum):
#         return self.resampler(spectrum, self.wavelengths)


# class Luminosity(Observable):

#     """This module computes the photometric luminosity (power) on a given band
#     for a given specific flux (per unit wavelength)."""

#     def __init__(self, **kwargs):

#   #      band = Filter(**kwargs)
#         self.flux = kwargs['flux']
#         self.integral_flux = np.trapz(self.flux*band.filter, self.wavelength)
#         self.central_wavelength = band.effective_wavelength()

# # =============================================================================
# class Magnitude(Observable):
# # =============================================================================
#     pass

# # =============================================================================
# class Photons(Observable):
# # =============================================================================
#     pass

def list_of_available_filters():
    filter_dir = os.path.join(os.path.dirname(__file__),
                              "data", "Filters")
    print(f"Checking filters available at {filter_dir}")
    return os.listdir(filter_dir)

def find_filt_from_name(name):
    filters = list_of_available_filters()
    for f in filters:
        if name.lower() in f.lower():
            return f
    return None

    

class PhotometricFilter(object):
    """This class represent a photometric filter
    
    Attributes
    ----------
    - trans_curve: (np.ndarray)
        Photometric transmission curve.
    - wave: (np.ndarray)
        Wavelength vector of the transmission curve.
    """
    wave = None
    trans_curve = None
    
    def __init__(self, wave=None, trans_curve=None, path=None, name=None):
        if wave is not None and trans_curve is not None:
            self.wave, self.trans_curve = wave, trans_curve
            if not hasattr(wave, 'unit'):
                print("Assuming that input wavelength is in angstrom")
                wave *= u.Angstrom
        elif path is not None:
            self.load_filter(path=path)
        elif name is not None:
            self.load_filter(name=name)

    def load_filter(self, path=None, name=None):
        print("Path", path, "Name ", name)
        if path is not None:
            self.wave, self.trans_curve  = np.loadtxt(path, usecols=(0, 1), unpack=True)
        elif name is not None:
            path = find_filt_from_name(name)
            if path is not None:
                self.wave, self.trans_curve = np.loadtxt(path, usecols=(0, 1), unpack=True)
            else:
                raise NameError(f"No filter found with input name {name}")
        else:
            raise NameError("No path, nor name provided")
        self.wave *= u.Angstrom
        return self.wave, self.trans_curve 

    def effective_wavelength(self):
        return np.sum(self.wave*self.trans_curve)/np.sum(self.trans_curve)

    def effective_bandwidth(self):
        return np.sqrt(8*np.log(2)*(
            np.sum(self.wave**2*self.trans_curve)/np.sum(self.trans_curve)
            - self.effective_wavelength()**2))

    def effective_transmission(self):
        return np.sum(self.trans_curve**2)/np.sum(self.trans_curve)

    def interpolate(self, wavelength_edges):
        """TODO"""

        self.original_trans_curve = self.trans_curve.copy()
        self.original_wave = self.wave
        
        cumulative_trans_curve = np.cumsum(self.trans_curve)
        interp_cum_trans_curve = np.interp(wavelength_edges, self.wave,
                                           cumulative_trans_curve)
        
        self.trans_curve = np.diff(interp_cum_trans_curve)
        self.wave = (wavelength_edges[:-1] + wavelength_edges[1:]) / 2
        print("Filter transmission curve interpolated to input wavelength array")


    def get_photons(self, spectra, spectra_err=None):
        photon_flux = np.trapz(
            spectra / (constants.h * constants.c / self.wave
                                   ) * self.trans_curve,
            x=self.wave)
        if spectra_err is not None:
            photon_flux_err = np.trapz(
                spectra_err / (constants.h * constants.c / self.wave
                                       ) * self.trans_curve,
                x=self.wave)
        else:
            photon_flux_err = None
        return photon_flux, photon_flux_err
    
    def get_AB(self, spectra, spectra_err=None):
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
        f_nu = 10**(-0.4 * m_ab) * 3631
        f_nu_err = 10**(-0.4 * m_ab_err)
        nu_f_nu = f_nu * c / self.effective_wavelength()
        nu_f_nu_err = f_nu_err * c / self.effective_wavelength()
        return nu_f_nu, nu_f_nu_err

class SquareFilter(PhotometricFilter):
    def __init__(self, central_wave, width, wave=None):
        if not hasattr(central_wave, 'unit'):
            print("Assuming that input central wavelength is expressed in angstrom")
            central_wave *= u.Angstrom
        if not hasattr(width, 'unit'):
            print("Assuming that input width is expressed in angstrom")
            width *= u.Angstrom

        if wave is None:
            self.wave = np.linspace(central_wave - width,
                                    central_wave + width,
                                    50)
        else:
            self.wave = wave
            if not hasattr(self.wave, 'unit'):
                print("Assuming that input central wavelength and width are expressed in angstrom")
                self.wave *= u.Angstrom
        self.trans_curve = np.ones(self.wave.size)
        self.trans_curve[self.wave < central_wave - width / 2] = 0
        self.trans_curve[self.wave > central_wave + width / 2] = 0

    


# # =============================================================================
# class magnitude(Filter):
# # =============================================================================
#     """This module computes the photmetric magnitude on a given band
#     for a given flux with UNITS expressed in erg/s.s"""

#     def __init__(self, absolute=False, **kwargs):

#         Filter.__init__(self, **kwargs)
#         self.nu = u.c/( self.wavelength)
#         self.flux = kwargs['flux']
#         self.absolute = absolute
#         photometric_system = kwargs['photometric_system']

#         if photometric_system=='AB':
#             self.magnitude = magnitude.AB(self)
#         if photometric_system=='Vega':
#             self.magnitude = magnitude.Vega(self)

#     def AB(self):   #photometric system  used with SDSS filters
#         """ This function computes the magnitude in the AB system of a given spectrum. The spectrum units must be in erg/s for absolute
#          magnitude computation or in erg/s/cm2 for apparent magnitude. """

#          ## Flux counts
#         if self.absolute==True:
#             self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   # flux at 10 pc.


#         diff_nu = - np.ediff1d(np.insert(self.nu, 0, 2*self.nu[0]-self.nu[1]))

#         integral_flux = np.nansum((self.flux/self.nu * self.filter * diff_nu) )

#         integral_R = np.nansum(self.filter*diff_nu)

#         mag =-2.5*np.log10(integral_flux/integral_R) - 48.60

# #        ## Photon counts
# #        if self.absolute==True:
# #            self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   # flux at 10 pc.
# #
# #        F_nu = self.flux/self.nu
# #
# #
# #        integral_flux = np.trapz(F_nu*self.filter, np.log10(self.nu))
# #
# #        integral_R = np.trapz(self.filter, np.log10(self.nu))
# #
# #        mag =-2.5*np.log10(integral_flux/integral_R) - 48.60


#         return mag

#     def Vega(self): #photometric system  used usually with Jonhson/Bessel/Coussin filters

#         diff_wl = np.ediff1d(np.insert((self.wavelength),0,0))

#         wl_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=0)
#         diff_wl_vega = np.ediff1d(np.insert(wl_vega,0,0))
#         flux_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=2)

#         if self.absolute == True:
#             flux_vega=( flux_vega * 4*np.pi*(25.30*u.ly/u.cm)**2 ) / (4*np.pi* (10*u.pc/u.cm)**2)
#             self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   #flux at 10 pc


#         vega_filter=Filter.new_filter(self.wl_filter, self.filter_resp , wl_vega)
#         integral_flux_vega=np.nansum(flux_vega * vega_filter * diff_wl_vega)
#         integral_flux = np.nansum(self.flux * self.filter * diff_wl )

#         m=-2.5* np.log10(integral_flux/integral_flux_vega) +0.58

#         return m

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt

    photo_filter = SquareFilter(central_wave=6000, width=500)
    

    wavelength = np.linspace(3000, 9000) * u.Angstrom
    flux = np.ones(wavelength.size - 1) * 1e-16 * u.erg / u.s / u.Angstrom / u.cm**2
    
    photo_filter.interpolate(wavelength)
    
    n_photons = photo_filter.get_AB(flux)

    plt.figure()
    plt.plot(photo_filter.wave, photo_filter.trans_curve)

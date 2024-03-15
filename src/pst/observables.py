#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:56:46 2018

@author: pablo
"""

import numpy as np
import os
from astropy import units as u
from scipy import interpolate
from pst import root_path

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
        
# =============================================================================
class Luminosity(Observable):
# =============================================================================
    pass

# =============================================================================
class Magnitude(Observable):
# =============================================================================
    pass

# =============================================================================
class Photons(Observable):
# =============================================================================
    pass

# =============================================================================
class Filter(object):
# =============================================================================

    def __init__(self, **kwargs):

        """This class provides a filter (SDSS, WISE, GALEX, 2MASS photometry) with the same
        number of points as the given wavelength array.

        The wavelength UNITS are by default expressed in AA"""
        self.wavelength = kwargs['wavelength']
        filter_name = kwargs['filter_name']

        if self.wavelength[5]>self.wavelength[6]:
            raise NameError('Wavelength array must be crescent')


        self.filter_resp, self.wl_filter = Filter.get_filt(filter_name)
        self.filter = Filter.new_filter(self.wl_filter,
                                        self.filter_resp,
                                        self.wavelength)


    def get_filt(filter_name):
        absolute_path = os.path.join(root_path, 'data', 'Filters')
        filters_path = {'u':os.path.join(absolute_path, 'SDSS','u.dat'),
                       'g':os.path.join(absolute_path, 'SDSS','g.dat'),
                       'r':os.path.join(absolute_path, 'SDSS','r.dat'),
                       'i':os.path.join(absolute_path, 'SDSS','i.dat'),
                       'z':os.path.join(absolute_path, 'SDSS','z.dat'),
                       'W1':os.path.join(absolute_path, 'WISE','W1.dat'),
                       'W2':os.path.join(absolute_path, 'WISE','W2.dat'),
                       'W3':os.path.join(absolute_path, 'WISE','W3.dat'),
                       'W4':os.path.join(absolute_path, 'WISE','W4.dat'),
                   'GFUV':os.path.join(absolute_path, 'GALEX','GALEX_FUV.dat'),
                   'GNUV':os.path.join(absolute_path, 'GALEX','GALEX_NUV.dat'),
                   '2MASS_J':os.path.join(absolute_path, '2MASS','2MASS_J.dat'),
                   '2MASS_H':os.path.join(absolute_path, '2MASS','2MASS_H.dat'),
                   '2MASS_Ks':os.path.join(absolute_path, '2MASS','2MASS_Ks.dat')}
        w_l, filt= np.loadtxt(filters_path[filter_name],
                              usecols=(0, 1), unpack=True)
        w_l = w_l*u.Angstrom
        return filt, w_l


    def effective_wavelength(self):
        return np.sum(self.wl_filter*self.filter_resp)/np.sum(self.filter_resp)

    def effective_bandwidth(self):
        return np.sqrt(8*np.log(2)*(
            np.sum(self.wl_filter**2*self.filter_resp)/np.sum(self.filter_resp)
            - self.effective_wavelength()**2))

    def effective_transmission(self):
        return np.sum(self.filter_resp**2)/np.sum(self.filter_resp)


    def new_filter( wl, fil, new_wl,*name, save=False):
        """ This function recieve the filter response and wavelength extension in order to interpolate it to a new set
         wavelengths.  First, it is checked if the filter starts or ends on the edges of the data,
         if this occurs an array of zeros is added to limit the effective area.
         Then, the filter response is differenciated seeking the limits of the curve to prevent wrong extrapolation. """

        f=interpolate.interp1d( wl, fil , fill_value= 'extrapolate' )

        new_filt=f(new_wl)

        bad_filter = False

        if  len(np.where(fil[0:5]>0.05)[0]):
            fil = np.concatenate((np.zeros(100),fil))
            bad_filter = True
        elif len(np.where(fil[-5:-1]>0.05)[0]):
            fil = np.concatenate((fil, np.zeros(100)))
            bad_filter = True

        band_init_pos = np.where(fil>0.01)[0][0]
        band_end_pos = np.where(fil[::]>0.01)[0][0]

        wl_init_pos = wl[band_init_pos]
        wl_end_pos = wl[-band_end_pos]



        new_band_init_pos = (np.abs(new_wl-wl_init_pos)).argmin()
        new_band_end_pos = (np.abs(new_wl-wl_end_pos)).argmin()

        # To smooth the limits of the band, first the band width is computed (number of points inside) and then a
        # called 'tails' to avoid erase any value close to te edge. If the filter starts at one corner of the distribution
        # obviously band_width_pos > new_band_init_pos, so the 'tails' could introduce negative positions. In order to avoid
        # this effect it is better to use the own initial position to delimitate the 'tail' of the band. But also, another
        # problem is the possible lack of points and then the tail would be underestimated. For this reason, is estimated
        # the number of points out of the new distribution and the tail is enlarged proportionally.

        band_width_pos =  new_band_end_pos - new_band_init_pos

        band_tails_right_pos = int(band_width_pos*0.1)

        band_tails_left_pos  = band_tails_right_pos

        if band_width_pos>new_band_init_pos:
            missing_points = 0
            if new_band_init_pos==0:
                delta_wl =  np.mean(np.ediff1d(new_wl[0:100]))
                missing_points = (new_wl[0] - wl_init_pos )/delta_wl

            band_tails_left_pos = int(new_band_init_pos*0.1)
            band_tails_right_pos = int(band_width_pos*0.1)+int(missing_points*0.1)

        elif band_width_pos > len(new_wl)-new_band_end_pos:
            missing_points = 0
            if new_band_end_pos==(len(new_wl)-1):
                delta_wl =  np.mean(np.ediff1d(new_wl[-100:-1]))
                missing_points = (wl_end_pos -new_wl[0] )/delta_wl

            band_tails_left_pos = int(band_width_pos*0.1)
            band_tails_right_pos = int((len(new_wl)-new_band_end_pos)*0.1)+int(missing_points*0.1)


        new_filt[0:new_band_init_pos-band_tails_left_pos] = np.clip(new_filt[0:(new_band_init_pos-band_tails_left_pos)],0,0)
        new_filt[(new_band_end_pos+band_tails_right_pos):-1] = np.clip(new_filt[(new_band_end_pos+band_tails_right_pos):-1],0,0)

        new_filt[-1]=0     # Sometimes it is the only point which gives problems

        # Furthermore, the worst case is when the original filter also starst at one corner of the distrib, so probably wrong
        # values appear close to the real curve. More drastically, all the sorrounding points are set to zero.

        if bad_filter == True:
            new_filt[new_band_end_pos:-1]=0
            new_filt[0:new_band_init_pos]=0

        if save==True:
            new_filt_zip=zip(new_wl,new_filt)

            with open('Filters/'+str(name)+'.txt', 'w' ) as f:
                for i,j in new_filt_zip:
                    f.write('{:.4} {:.4}\n'.format(i,j))
            print('Filter'+str(name)+'saved succesfully ')

        return new_filt


    def square_filt(wl,l_i,l_e):

         s_filt=np.ones(len(wl))

         for i,j in enumerate(wl):
             if j<l_i:

                 s_filt[i]=0

             elif j>l_i and j<l_e:

                 s_filt[i]=1

             elif j>l_e:

                 s_filt[i]=0
         return s_filt


# =============================================================================
class luminosity(Filter):
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
class magnitude(Filter):
# =============================================================================
    """This module computes the photmetric magnitude on a given band
    for a given flux with UNITS expressed in erg/s.s"""

    def __init__(self, absolute=False, **kwargs):

        Filter.__init__(self, **kwargs)
        self.nu = 3e8*u.m/u.s/( self.wavelength)
        self.flux = kwargs['flux']
        self.absolute = absolute
        photometric_system = kwargs['photometric_system']

        if photometric_system=='AB':
            self.magnitude = magnitude.AB(self)
        if photometric_system=='Vega':
            self.magnitude = magnitude.Vega(self)

    def AB(self):   #photometric system  used with SDSS filters
        """ This function computes the magnitude in the AB system of a given spectrum. The spectrum units must be in erg/s for absolute
         magnitude computation or in erg/s/cm2 for apparent magnitude. """

         ## Flux counts
        if self.absolute==True:
            self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   # flux at 10 pc.


        diff_nu = - np.ediff1d(np.insert(self.nu, 0, 2*self.nu[0]-self.nu[1]))

        integral_flux = np.nansum((self.flux/self.nu * self.filter * diff_nu) )

        integral_R = np.nansum(self.filter*diff_nu)
        print(integral_flux.to_value)
        print(integral_R)
        mag =-2.5*np.log10(integral_flux/integral_R) - 48.60

#        ## Photon counts
#        if self.absolute==True:
#            self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   # flux at 10 pc.
#
#        F_nu = self.flux/self.nu
#
#
#        integral_flux = np.trapz(F_nu*self.filter, np.log10(self.nu))
#
#        integral_R = np.trapz(self.filter, np.log10(self.nu))
#
#        mag =-2.5*np.log10(integral_flux/integral_R) - 48.60


        return mag

    def Vega(self): #photometric system  used usually with Jonhson/Bessel/Coussin filters

        diff_wl = np.ediff1d(np.insert((self.wavelength),0,0))

        wl_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=0)
        diff_wl_vega = np.ediff1d(np.insert(wl_vega,0,0))
        flux_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=2)

        if self.absolute == True:
            flux_vega=( flux_vega * 4*np.pi*(25.30*u.ly/u.cm)**2 ) / (4*np.pi* (10*u.pc/u.cm)**2)
            self.flux = self.flux/(4*np.pi* (10*u.pc/u.cm)**2)   #flux at 10 pc


        vega_filter=Filter.new_filter(self.wl_filter, self.filter_resp , wl_vega)
        integral_flux_vega=np.nansum(flux_vega * vega_filter * diff_wl_vega)
        integral_flux = np.nansum(self.flux * self.filter * diff_wl )

        m=-2.5* np.log10(integral_flux/integral_flux_vega) +0.58

        return m

if __name__ == '__main__':
    from astropy.modeling import models
    import astropy.constants as c
    from astropy.modeling.physical_models import BlackBody
    from specutils import Spectrum1D

    bb = models.BlackBody(temperature=5700*u.K)
    wl = np.logspace(1, 5, 141) * u.angstrom    
    spec = Spectrum1D(bb(wl), wl)    
    
    blue_wavelength = np.linspace(4000, 6000, 10)*u.angstrom
    
    
    blue_arm = Spectrum(blue_wavelength)
    resampled_spec = blue_arm.from_Spectrum1D(spec)
    
    from matplotlib import pyplot as plt
    
    plt.figure()
    plt.plot(spec.spectral_axis, spec.flux*c.c/wl**2, '-+')
    plt.plot(resampled_spec.spectral_axis, resampled_spec.flux*c.c/blue_wavelength**2, '-o')
    plt.xlim(3000, 10000)    
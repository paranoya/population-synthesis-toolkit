# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:10:48 2024

@author: Propietario
"""
import os
import pst
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import csv
from astropy.io import fits
import random
import extinction

t0 = 13.7*u.Gyr
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr

Z_i = 0.02
A_V_model = 0
R_V = 3.1
ssp = pst.SSP.PopStar(IMF="sal")
ssp.cut_wavelength(1000, 1600000)
obs_filters_wl = []
obs_filters = ['u', 'g', 'r', 'i', 'z']
for name in obs_filters:
    photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = name)
    obs_filters_wl.append(photo.effective_wavelength().to_value())
obs_filters_wl = np.array(obs_filters_wl)   

model_A_lambda = extinction.ccm89(obs_filters_wl, A_V_model, R_V)
dust_extinction = np.power(10, -model_A_lambda/2.5)

def get_flux_densities(model, ssp, obs_filters, Z_i, t, **kwargs):
    fnu = []
#    fnu_error = []
    sed = model.compute_SED(SSP = ssp, t_obs = t0)

    for i, filter_name in enumerate(obs_filters):
        photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = filter_name)
        spectra_flambda = ( sed/(4*np.pi*(10*u.pc.to('cm'))*u.cm**2) )
        fnu_Jy, fnu_Jy_err = photo.get_fnu(spectra_flambda, spectra_err=None)
        fnu.append( fnu_Jy )
#        fnu_error.append( fnu_Jy_error )
        fnu_Jy
    return u.Quantity(fnu)

M_stars = 1 #Msun
tb = 3 #Gyr
c = 0.01 #Gyr
gaussian_fit = pst.models.Gaussian_burst(Z=0.02, M_stars = M_stars, t = tb, c = c)

Fnu_model = get_flux_densities(gaussian_fit, ssp, obs_filters, Z_i, t)*dust_extinction
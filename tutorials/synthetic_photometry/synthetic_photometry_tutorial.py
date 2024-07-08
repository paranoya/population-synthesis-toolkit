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
obs_filters = ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i', 'SLOAN/SDSS.z']
for name in obs_filters:
    photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = name)
    obs_filters_wl.append(photo.effective_wavelength().to_value())
obs_filters_wl = np.array(obs_filters_wl)   

model_A_lambda = extinction.ccm89(obs_filters_wl, A_V_model, R_V)
dust_extinction = np.power(10, -model_A_lambda/2.5)

def get_flux_densities(model, ssp, obs_filters, Z_i, t, **kwargs):
    fnu = []
    sed = model.compute_SED(SSP = ssp, t_obs = t0)
    for i, filter_name in enumerate(obs_filters):
        photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = filter_name)
        spectra_flambda = ( sed/(4*np.pi*(10*u.pc)**2) )
        fnu_Jy, fnu_Jy_err = photo.get_fnu(spectra_flambda, spectra_err=None)
        fnu.append( fnu_Jy )
    return u.Quantity(fnu)

#%%
# M_stars = 1 #Msun
# log_tb = 6.1
# tb = 10**log_tb*1e-9 #Gyr
# c = 0.001 #Gyr
# gaussian_fit = pst.models.Gaussian_burst(Z=0.02, M_stars = M_stars, t = tb, c = c)

# Fnu_model = get_flux_densities(gaussian_fit, ssp, obs_filters, Z_i, t)*dust_extinction
# print('Fnu ', Fnu_model)
#%%
#1 burst
M_stars = 1 #Msun
Z_i = 0.02
log_tb1 = 8

age = 10**log_tb1*1e-9 #Gyr
tb = t0.to_value(u.Gyr) - age #Gyr

single_burst_fit_1 = pst.models.Single_burst(Z = Z_i, M_stars = M_stars, t_burst = tb)

Fnu_model = get_flux_densities(single_burst_fit_1, ssp, obs_filters, Z_i, t)*dust_extinction

c_speed = 3e8*u.m/u.s
L_lambda = (Fnu_model*c_speed/(obs_filters_wl*u.Angstrom)**2*4*np.pi*(10*u.pc)**2).to(u.Lsun/u.Angstrom)
print('L_lambda = ', L_lambda)

f = Fnu_model.to_value() #Jy
f0 = np.array([3767, 3631, 3631, 3631, 3565]) #Jy
b = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]) #Jy

mag = (-2.5/np.log(10))*(np.arcsinh((f/f0)/2*b)+np.log(b))
AB_mag = -2.5*np.log10(f/3631)

print('Fnu ', Fnu_model)
print('SDSS ', mag)
print('AB ', AB_mag)

#%%
#Sum o 3 bursts
log_tb2 =  np.log10(13*1e9)
log_tb3 = 8.1

single_burst_fit_2 = pst.models.Single_burst(Z = Z_i, M_stars = M_stars, t_burst = 10**log_tb2*1e-9)
single_burst_fit_3 = pst.models.Single_burst(Z = Z_i, M_stars = M_stars, t_burst = 10**log_tb3*1e-9)

sum_of_mass = single_burst_fit_1.integral_SFR(t) + single_burst_fit_2.integral_SFR(t) + single_burst_fit_3.integral_SFR(t)

sum_of_bursts_model = pst.models.Tabular_MFH(t, 
                            sum_of_mass, 
                            Z = np.ones(len(t))*Z_i) #Generating test-model


Fnu_total = get_flux_densities(sum_of_bursts_model, ssp, obs_filters, Z_i, t)*dust_extinction

L_lambda = (Fnu_model*c_speed/(obs_filters_wl*u.Angstrom)**2*4*np.pi*(10*u.pc)**2).to(u.Lsun/u.Angstrom)
f = Fnu_model.to_value() #Jy
mag = (-2.5/np.log(10))*(np.arcsinh((f/f0)/2*b)+np.log(b))
AB_mag = -2.5*np.log10(f/3631)







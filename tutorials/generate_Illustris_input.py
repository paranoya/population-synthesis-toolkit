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
n_models = 900 #Models computed per run

ssp = pst.SSP.PopStar(IMF="sal")
ssp.cut_wavelength(1000, 1600000)
obs_filters_wl = []
obs_filters = ['u', 'g', 'r', 'i', 'z']
for name in obs_filters:
    photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = name)
    obs_filters_wl.append(photo.effective_wavelength().to_value())
obs_filters_wl = np.array(obs_filters_wl)   


def get_flux_densities(model, ssp, obs_filters, Z_i, t, **kwargs):
    fnu = []
#    fnu_error = []
    z_array = Z_i*np.ones(len(t))
    sed = model.compute_SED(SSP = ssp, t_obs = t0)

    for i, filter_name in enumerate(obs_filters):
        photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = filter_name)
        spectra_flambda = ( sed/(4*np.pi*(10*u.pc.to('cm'))*u.cm**2) )
        fnu_Jy, fnu_Jy_err = photo.get_fnu(spectra_flambda, spectra_err=None)
        fnu.append( fnu_Jy )
#        fnu_error.append( fnu_Jy_error )
        fnu_Jy
    return u.Quantity(fnu)
    
read_path = os.getcwd()
f = open(os.path.join(read_path, 'Illustris_input.txt'), 'w+')

real_models = {}
Illustris_data_path = 'F:/population-synthesis-toolkit-pst_2/pst/examples/data/Illustris'

#Reading shuffled IDs and dividing them       
id_list = []
with open(os.path.join(os.getcwd(), 'Illustris_shuffled_IDs.txt'), 'r+') as Illustris_IDs_file:
    id_list = Illustris_IDs_file.read().splitlines()
        
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

Illustris_sample = list(id_list)[:n_models]

for i, subhalo in enumerate(Illustris_sample):
    if i%50==0:
        print('Model #{}'.format(i))
    filename = 'subhalo_{}_sfh.fits'.format(subhalo)      
    subhalo = 'subhalo_{}'.format(subhalo)
    real_models[subhalo] = {}
    
    if not os.path.exists(os.path.join(Illustris_data_path, filename)):
        print('not found')
        continue
    else:
        with fits.open(os.path.join(Illustris_data_path, filename)) as hdul:
            lb_time = hdul[1].data['lookback_time']*u.Gyr
            real_models[subhalo]['time'] = (t0-lb_time)[::-1]
            mass_formed = np.sum(hdul[3].data, axis=1)*u.Msun # sum over metallicities
            real_models[subhalo]['mass_formed'] = np.cumsum(mass_formed[::-1])
            real_models[subhalo]['metallicity'] = hdul[0].header['HIERARCH starmetallicity']                                   
            real_models[subhalo]['A_V'] = float(np.array([round(random.uniform(0, 1.2), 1)]))
            
            model_test = pst.models.Tabular_MFH(real_models[subhalo]['time'], 
                                    real_models[subhalo]['mass_formed'], 
                                    Z = np.ones(len(real_models[subhalo]['time']))*real_models[subhalo]['metallicity']*u.dimensionless_unscaled)
            Fnu_target = get_flux_densities(model_test, ssp, obs_filters, np.ones(len(t))*real_models[subhalo]['metallicity']*u.dimensionless_unscaled, t)
            R_V = 3.1
            target_A_lambda = extinction.ccm89(obs_filters_wl, real_models[subhalo]['A_V'], R_V)
            dust_extinction_target = np.power(10, -target_A_lambda/2.5)

            Fnu_obs = Fnu_target*dust_extinction_target #Applying extinction      
            error_Fnu_obs = Fnu_obs*np.array([2e-2, 1e-2, 1e-2, 1e-2, 1e-2]) #Luminosity errors: 2% for u, 1% for griz (SLOAN)
            real_models[subhalo]['Fnu_obs'] = Fnu_obs
            real_models[subhalo]['error_Fnu_obs'] = error_Fnu_obs
            f.write('{} {} {}'.format(subhalo[8:], " ".join(str(x) for x in Fnu_obs.to_value()), " ".join(str(x) for x in error_Fnu_obs.to_value()))+'\n')
f.close()
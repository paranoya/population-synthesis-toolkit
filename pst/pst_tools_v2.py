# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:12:11 2024

@author: Propietario
"""

import numpy as np
from astropy import units as u
import pst
from matplotlib import pyplot as plt
import time as tic
import os
from astropy.io import fits
from scipy import special
import extinction
import random

def generate_luminosities(model, ssp, obs_filters, **kwargs):
    t0 = kwargs.get('t0', 13.7*u.Gyr)
    real_sed = model.compute_SED(ssp, t0)

    L_Lsun = []
    for filter_name in obs_filters:
        photo = pst.observables.luminosity(
            flux=real_sed, wavelength=ssp.wavelength, filter_name=filter_name)
        L_Lsun.append(photo.integral_flux.to_value(u.Lsun))
        
    return np.array(L_Lsun)

def synthetic_mfh(t, t0, **kwargs):
    t_hat = 1/t/t0
    mfh_type = kwargs.get('mfh_type', None)
    if mfh_type == None:
        print('MISSING MFH TYPE')
    if mfh_type == 'square':
        t_start = kwargs.get('mfh_t_start', 0.)
        t_end = kwargs.get('mfh_t_end', 0.)
        
        t_hat_end = 1-t_end/t0
        t_hat_start = 1-t_start/t0
        mfh = (1-(t_hat.clip(t_hat_start, t_hat_end)-t_hat_start)/(t_hat_end - t_hat_start))
        return mfh*u.Msun
    if mfh_type == 'gaussian':
        t_peak = kwargs.get('t_peak')
        fwhm = kwargs.get('fwhm')
        return 1/2*( -special.erf((-t_peak)/(np.sqrt(2)*fwhm)) +  special.erf((t-t_peak)/(np.sqrt(2)*fwhm) ))*u.Msun
    if mfh_type == 'tau':
        hat_tau_target = kwargs.get('hat_tau_target')
        return (1-np.exp((t_hat-1)/hat_tau_target))*u.Msun

def get_real_models(ssp, obs_filters, obs_filters_wl, n_models, n_runs, current_run, target_mfh, **kwargs):

    t0 = kwargs.get('t0', 13.7*u.Gyr)
    if target_mfh == 'Illustris':


        real_models = {}
        Illustris_data_path = os.path.join(os.getcwd(), 'examples', 'data', 'Illustris')
           
        #Reading shuffled IDs and dividing them       
        id_list = []
        with open(os.path.join(os.getcwd(), 'Illustris_shuffled_IDs.txt'), 'r+') as Illustris_IDs_file:
            id_list = Illustris_IDs_file.read().splitlines()
                
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
        divided_chunks = list(split(id_list[:n_models*n_runs], n_runs))[0]
        Illustris_sample = divided_chunks[current_run]
    
        for subhalo in Illustris_sample:
            
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
                    real_models[subhalo]['A_V'] = np.array([round(random.uniform(0, 1.2), 1)])
                    
                    model_test = pst.models.Tabular_MFH(real_models[subhalo]['time'], 
                                            real_models[subhalo]['mass_formed'], 
                                            Z = real_models[subhalo]['metallicity'])
                    L_target = generate_luminosities(model_test, ssp, obs_filters) #Generating the luminosities of the test
                    R_V = 3.1
                    target_A_lambda = extinction.ccm89(obs_filters_wl, real_models[subhalo]['A_V'], R_V)
                    dust_extinction_target = np.power(10, -target_A_lambda/2.5)
                    L_obs = L_target*dust_extinction_target #Applying extinction      
                    L_err = np.array([2e-2, 1e-2, 1e-2, 1e-2, 1e-2]) #Errors on the Luminosity       
                    error_L_obs = L_obs*L_err #Luminosity errors: 2% for u, 1% for griz (SLOAN)
                    real_models[subhalo]['L_obs'] = L_obs
                    real_models[subhalo]['error_L_obs'] = error_L_obs
                    
        return real_models
    
    elif target_mfh == 'input_luminosities':
        real_models = {}
        
        luminosities_path = 'C:/Users/Propietario/Downloads/table_dani_PabloCorcho.fit'
        hdul_p = fits.open(luminosities_path)
        target_list = hdul_p[1].data[n_models*current_run:n_models*(current_run+1)]
        divided_chunks = target_list[0]
        for target in hdul_p[1].data[n_models*current_run:n_models*(current_run+1)]:
            subhalo = target['objID']
            real_models[subhalo] = {}
            real_models[subhalo]['objID'] = target['objID']
            real_models[subhalo]['metallicity'] = target['Z']    
            real_models[subhalo]['metallicity_error'] = target['Zerr']    
            real_models[subhalo]['PetroMag'] = np.array([target['petroMag_u'], target['petroMag_g'], target['petroMag_r'], target['petroMag_i'], target['petroMag_z']])
            real_models[subhalo]['PetroMagErr'] = np.array([target['petroMagErr_u'], target['petroMagErr_g'], target['petroMagErr_r'], target['petroMagErr_i'], target['petroMagErr_z']])
            real_models[subhalo]['ModelMag'] = np.array([target['modelMag_u'], target['modelMag_g'], target['modelMag_r'], target['modelMag_i'], target['modelMag_z']])
            real_models[subhalo]['ModelMagErr'] = np.array([target['modelMagErr_u'], target['modelMagErr_g'], target['modelMagErr_r'], target['modelMagErr_i'], target['modelMagErr_z']])
            real_models[subhalo]['lgm_tot_p16'] = target['lgm_tot_p16']
            real_models[subhalo]['lgm_tot_p50'] = target['lgm_tot_p50']
            real_models[subhalo]['lgm_tot_p84'] = target['lgm_tot_p84']
            real_models[subhalo]['specsfr_tot_p16'] = target['specsfr_tot_p16']
            real_models[subhalo]['specsfr_tot_p50'] = target['specsfr_tot_p50']
            real_models[subhalo]['specsfr_tot_p84'] = target['specsfr_tot_p84']
        return real_models
    
    if target_mfh == 'salim':
    
        ugriz_wl = []
        for obs in obs_filters:
            filter_resp, wl_filter = pst.observables.Filter.get_filt(obs)
            wl = np.sum(wl_filter*filter_resp)/np.sum(filter_resp)
            ugriz_wl.append(wl.to_value())
        ugriz_wl = np.array(ugriz_wl)*u.Angstrom
        
        f = 'C:/Users/Propietario/Downloads/nsa_v0_1_2.fits'
        hdul = fits.open(os.path.join(f))    
        
        f = 'C:/Users/Propietario/Downloads/GSWLC-D2.dat.gz'
        input_salim = []
        for column in range(0, 24):
            input_salim.append( [x.split( )[column] for x in open(f).readlines()])
        
                         
        real_models = {}
        data_nsa = hdul[1].data
        H0=70*u.km/u.s/u.Mpc #km/s/Mpc
        c = 3e8*u.m/u.s #m/s
        obs_filters = ['GFUV', 'GNUV', 'u', 'g', 'r', 'i', 'z']
        obs_filters_wl = []
        for obs in obs_filters:
            filter_resp, wl_filter = pst.observables.Filter.get_filt(obs)
            wl = np.sum(wl_filter*filter_resp)/np.sum(filter_resp)
            obs_filters_wl.append(wl.to_value())
        obs_filters_wl = np.array(obs_filters_wl)
        wl = obs_filters_wl*u.Angstrom
        with open('C:/Users/Propietario/coincidences_salim_nsa.txt', 'r+') as f:
            f_salim = []
            for line in f:
                f_salim.append([int(num) for num in line.split()])
            for i, j in f_salim[:1]:
    
                model_id = np.array(input_salim).T[i][0]
                
                real_models[model_id] = {}
                redshift = data_nsa['Z'][j]
                
                D = (redshift*c/H0).to(u.pc) #pc
                F_nanomagg = data_nsa['NMGY'][j] #Flux nMagg
                F_jy = 3631*u.Jy*1e-9*F_nanomagg #Jy
#                m = -2.5*np.log10(F_jy/(3631*u.Jy))
#    #            m = -2.5*np.log10(F_petro)+22.5
#                M_abs=m-5*np.log10(D/(10*u.pc))
#    #            M_abs = data_nsa['ABSMAG'][j]
#                L0 = 1*u.Lsun
#                L_Lsun = L0*10**(-.4*M_abs)
                Lnu = (F_jy*4*np.pi*D**2).to(u.Lsun/u.Hz)
                nuLnu = (Lnu*c/wl).to(u.Lsun)
                
                F_nanomagg_error = 1/np.sqrt(data_nsa['NMGY_IVAR'][j]) #nMagg
                F_jy_error = 3631*u.Jy*1e-9*F_nanomagg_error #Jy
#                m_error = -2.5*np.log10(F_jy_error/(3631*u.Jy))
#    #            m_error = -2.5*np.log10(F_petro_error)+22.5
#                M_abs_error = m_error-5*np.log10(D/(10*u.pc))
#    #            M_abs_error = 1/data_nsa['AMIVAR'][j]**(1/2)
#                L_Lsun_error = L0*10**(-.4*M_abs_error)
                Lnu_error = (F_jy_error*4*np.pi*D**2).to(u.Lsun/u.Hz)
                nuLnu_error = (Lnu_error*c/wl).to(u.Lsun)
                                                 
#                real_models[model_id]['L_obs'] = np.array(L_Lsun)
#                real_models[model_id]['error_L_obs'] = np.array(L_Lsun_error)
                real_models[model_id]['L_obs'] = np.array(nuLnu)
                real_models[model_id]['error_L_obs'] = np.array(nuLnu_error)

                real_models[model_id]['metallicity'] = data_nsa['METS'][j]
                real_models[model_id]['stellar_mass'] = data_nsa['MASS'][j]
                real_models[model_id]['extinction'] = data_nsa['EXTINCTION'][j]
                real_models[model_id]['IAUNAME'] = data_nsa['IAUNAME'][j]
                
                real_models[model_id]['logM'] = np.array(input_salim).T[i][9]
                real_models[model_id]['logM_sigma'] = np.array(input_salim).T[i][10]
                real_models[model_id]['logSFR'] = np.array(input_salim).T[i][11]
                real_models[model_id]['logSFR_sigma'] = np.array(input_salim).T[i][12]
                real_models[model_id]['A_V'] = np.array(input_salim).T[i][17]
                real_models[model_id]['A_V_sigma'] = np.array(input_salim).T[i][18]
        return real_models
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:32:14 2024

@author: Propietario
"""
import os
import pst
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import csv
import re

t0 = 13.7 * u.Gyr  # time of observation 
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
lookback_time = t0*t_hat
test_lookback_time = lookback_time[::-1]

# t_end_values = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
# model_A_V_grid = [0, 0.5, 1.0, 1.2]
# model_z_grid = np.array([0.004, 0.008, 0.02, 0.05])

ti_grid = [0]*u.Gyr
# tf_grid = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
tf_grid = [10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
z_grid = np.array([0.004, 0.008, 0.02, 0.05])
av_grid = [0, 0.5, 1.0, 1.2]

#Simplified parameters
# tf_grid = [13.7]*u.Gyr
# z_grid = np.array([0.02])
# av_grid = [0]

N_range = np.arange(1, 4)

test_subject = 'salim_100'
input_file = os.path.join(os.getcwd(), '{}_input.csv'.format(test_subject))
output_file = os.path.join(os.getcwd(), '{}_output_full.csv'.format(test_subject))
real_data_file = os.path.join(os.getcwd(), '{}_real_data.csv'.format(test_subject))


obs_filters = ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV', 'SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i', 'SLOAN/SDSS.z']

#%%
pst.fitting_module.compute_polynomial_models(input_file, output_file, obs_filters,
                                              ti_grid = ti_grid, tf_grid = tf_grid,
                                              z_grid = z_grid, av_grid = av_grid,
                                              N_range = N_range)

#%%
ssp = pst.SSP.PopStar(IMF="sal")
ssp.cut_wavelength(1000, 1600000)
obs_filters_wl = []
for name in obs_filters:
    photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = name)
    obs_filters_wl.append(photo.effective_wavelength().to_value())
obs_filters_wl = np.array(obs_filters_wl)

#%% SALIM
   
real_data_content = csv.reader(open(real_data_file, 'r'))

target_id = []
met_salim = []
ssfr_salim = []
ssfr_salim_error = []
ssfr_salim_sn = []

av_salim = []
av_salim_error = []
for i, row in enumerate(real_data_content):
    target_id.append(int(row[0])) 
    met_salim.append(float(row[1]))
    m_salim=10**float(row[2])
    m_salim_error = 10**float(row[3])
    sfr_salim=10**float(row[4])
    sfr_salim_error = 10**float(row[5])     
    av_salim.append(float(row[6]))
    av_salim_error.append(float(row[7]))
    
    ssfr = sfr_salim/m_salim
    ssfr_error = ssfr*(m_salim_error/m_salim + sfr_salim_error/sfr_salim)
    
    ssfr_salim.append(ssfr)
    ssfr_salim_error.append(ssfr_error)
    ssfr_salim_sn.append(ssfr/ssfr_error)
    
av_salim = np.array(av_salim)
#poly ssfr of salim 
poly_ssfr = []
bad_id = []
bad_id_salim_ssfr = []
bad_id_salim_ssfr_error = []
bad_id_salim_ssfr_sn = []
bad_id_poly_ssfr = []
bad_id_met_salim = []
bad_id_av_salim = []
bad_id_av_salim_error = []

dusty_ssfr_salim = []
dusty_ssfr_poly = []
av_lim = .3
   
output_content = csv.reader(open(output_file, "r"))

dust_polynomial = []
dust_polynomial_error = []

for i, row in enumerate(output_content):
    dust_polynomial.append(np.array([float(i) for i in re.findall(r'\S+', row[6])]))
    dust_polynomial_error.append(np.array([float(i) for i in re.findall(r'\S+', row[7])]))
    
    poly_mass_fraction = np.array([float(i) for i in re.findall(r'\S+', row[3])])
    lbt= .3
    lbt_index = abs(test_lookback_time.to_value()-lbt).argmin()
    p = poly_mass_fraction[lbt_index]
    poly_ssfr_i = p/lbt/1e9
    poly_ssfr.append( poly_ssfr_i)
    
    #Selecting bad ids and their ssfr
    if poly_ssfr_i <= 4e-12 and ssfr_salim[i]>1e-11:
        bad_id_poly_ssfr.append(poly_ssfr_i)
        bad_id_salim_ssfr.append(ssfr_salim[i])
        bad_id.append(target_id[i])
        
        bad_id_salim_ssfr_error.append(ssfr_salim_error[i])
        bad_id_salim_ssfr_sn.append(ssfr_salim[i]/ssfr_salim_error[i])
        
        bad_id_met_salim.append(met_salim[i])
        bad_id_av_salim.append(av_salim[i])
        bad_id_av_salim_error.append(av_salim_error[i])
        
        
        if av_salim[i]>av_lim:
            dusty_ssfr_salim.append(ssfr_salim[i])
            dusty_ssfr_poly.append(poly_ssfr_i)
                                        
#%%                   
plt.figure()
plt.xlabel('sSFR poly')
plt.ylabel('sSFR salim')

av_lim = 199
plt.scatter(np.array(poly_ssfr)[av_salim<av_lim], np.array(ssfr_salim)[av_salim<av_lim], c='blue', alpha=.7, s=1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-13, 1e-9)
plt.ylim(1e-13, 1e-9)
plt.legend()
plt.plot(np.array([1e-13, 1e-8]), np.array([1e-13, 1e-8]), 'k-', alpha=.1)
plt.show()
        
#%%
fnu_list = []
fnu_error_list = []
sn_fnu = []
sn_fnu_min = []

ab_mags = []
input_data = csv.reader(open(input_file, 'r'))

for i, row in enumerate(input_data):
    target_id = int(row[0])
    fnu = np.array([float(i) for i in re.findall(r'\S+', row[1])])
    fnu_error = np.array([float(i) for i in re.findall(r'\S+', row[2])])
    fnu_list.append(fnu)
    fnu_error_list.append(fnu_error)
    sn_fnu.append(np.array(fnu)/np.array(fnu_error))
    sn_fnu_min.append(min(np.array(fnu)/np.array(fnu_error)))
    
    AB_mag = -2.5*np.log10(fnu/3631)
    ab_mags.append(AB_mag)
    
u_poly = np.array([m[0] for m in ab_mags])
g_poly = np.array([m[1] for m in ab_mags])
r_poly = np.array([m[2] for m in ab_mags])

plt.figure()
plt.title('Polynomial')
plt.scatter( u_poly-g_poly, g_poly-r_poly, s=2)
plt.ylabel('g-r')
plt.xlabel('u-g')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

#%%
fnu_input = []
fnu_input_error = []
ab_input = []
content = csv.reader(open(input_file, "r"))
for i, row in enumerate(content):
    r = np.array([float(k) for k in row[1].split()])
    
    # target_ID = int(r[0])
    fnu_input.append(r[1:6])
    fnu_input_error.append(r[6:])
    
    f = r[1:6]
    ab_input.append(-2.5*np.log10(f/3631))

    
u_input = np.array([m[0] for m in ab_input])
g_input = np.array([m[1] for m in ab_input])
r_input = np.array([m[2] for m in ab_input])

plt.figure()
plt.title('Input')
plt.scatter( u_input-g_input, g_input-r_input, s=2)
plt.ylabel('g-r')
plt.xlabel('u-g')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

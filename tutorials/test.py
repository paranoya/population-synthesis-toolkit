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

t0 = 13.7 * u.Gyr  # time of observation 
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
lookback_time = t0*t_hat
test_lookback_time = lookback_time[::-1]

# t_end_values = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
# model_A_V_grid = [0, 0.5, 1.0, 1.2]
# model_z_grid = np.array([0.004, 0.008, 0.02, 0.05])

ti_grid = [0]*u.Gyr

tf_grid = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
z_grid = np.array([0.004, 0.008, 0.02, 0.05])

# tf_grid = [13.7]*u.Gyr
# z_grid = np.array([0.02])

av_grid = [0, 0.5, 1.0, 1.2]
N_range = np.arange(1, 4)

test_subject = 'salim'
input_file = os.path.join(os.getcwd(), '{}_input.txt'.format(test_subject))
output_file = os.path.join(os.getcwd(), '{}_output.csv'.format(test_subject))
obs_filters = ['GALEX_FUV', 'GALEX_NUV', 'u', 'g', 'r', 'i', 'z']

#%%
pst.fitting_module.compute_polynomial_models(input_file, output_file, obs_filters,
                                             ti_grid = ti_grid, tf_grid = tf_grid,
                                             z_grid = z_grid, av_grid = av_grid,
                                             N_range = N_range)

#%% get real ssfr of delay_tau

# #[met, logm, log_sigma, logsfr, logsfr_sigma, av, av_sigma]

# tau=.6*u.Gyr
# m_delay = ( 1 - np.exp(-t/tau)*(tau+t)/tau)
# sfr_delay = (t/tau**2)*np.exp(-t/tau)
# ssfr_delay = sfr_delay/m_delay

#%% get real ssfr of salim data
f = open(input_file, 'r+')
input_data = f.read().splitlines()   
input_ids = []

for line in input_data:
    data = line.split(' ')
    input_ids.append(data[0]) #IDs from the input
    
path = 'C:/Users/Propietario/Desktop/pack_read'
filename = 'salim_real_data.txt' #[met, logm, logmerr, logsfr, logsfrerr, av, averr]
f = open(os.path.join(path, filename), 'r+')
content = f.read().splitlines()                    

target_id = []
met_salim = []
ssfr_salim = []
ssfr_salim_error = []
ssfr_salim_sn = []

av_salim = []
av_salim_error = []
for line in content:
    data = line.split(' ')
    if data[0] in input_ids:
        target_id.append(data[0]) #IDs from the original data
        met_salim.append(data[1])
        m_salim=10**float(data[2])
        m_salim_error = 10**float(data[3])
        sfr_salim=10**float(data[4])
        sfr_salim_error = 10**float(data[5])     
        av_salim.append(float(data[6]))
        av_salim_error.append(float(data[7]))
        
        ssfr = sfr_salim/m_salim
        ssfr_error = ssfr*(m_salim_error/m_salim + sfr_salim_error/sfr_salim)
        
        ssfr_salim.append(ssfr)
        ssfr_salim_error.append(ssfr_error)
        ssfr_salim_sn.append(ssfr/ssfr_error)
#%% poly ssfr of salim 
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

csvfile = open(output_file, 'r')
csv_reader = csv.reader(csvfile)

dust_polynomial = []
dust_polynomial_error = []
for i, row in enumerate(csv_reader):
    dust_polynomial.append(np.array([float(i) for i in row[-3].split(' ')]))
    dust_polynomial_error.append(np.array([float(i) for i in row[-4].split(' ')]))
    
    poly_mass_fraction = np.array([float(i) for i in row[0].split(' ')])
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
                            
#%% Bad ids fnu
fnu_ = []
fnu_error_ = []
sn_fnu = []
sn_fnu_min = []

for line in input_data:
    data = line.split(' ')
    subhalo = data[0]
    if subhalo in bad_id:
        fnu = [float(string) for string in data[1:len(obs_filters)+1]]
        fnu_error = [float(string) for string in data[-len(obs_filters):]]
        fnu_.append(fnu)
        fnu_error_.append(fnu_error)
        sn_fnu.append(np.array(fnu)/np.array(fnu_error))
        sn_fnu_min.append(min(np.array(fnu)/np.array(fnu_error)))
        
 #%%                           
plt.figure()
plt.xlabel('sSFR poly')
plt.ylabel('sSFR salim')

plt.scatter(np.array(poly_ssfr)[av_salim<av_lim], np.array(ssfr_salim)[av_salim<av_lim], c='blue', alpha=.7, s=1)
# plt.scatter(dusty_ssfr_poly, dusty_ssfr_salim, c='red', alpha=1, s=1, label='AV>{}'.format(av_lim))

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-12, 1e-9)
plt.ylim(1e-12, 1e-9)
plt.legend()
plt.plot(np.array([1e-13, 1e-8]), np.array([1e-13, 1e-8]), 'k-', alpha=.1)
plt.show()
        
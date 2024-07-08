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
from astropy.io import fits
import extinction

t0 = 13.7 * u.Gyr  # time of observation 
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
lookback_time = t0*t_hat
test_lookback_time = lookback_time[::-1]

obs_filters = ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i', 'SLOAN/SDSS.z']
N_range = np.arange(1, 4)

ti_grid = [0]*u.Gyr
# tf_grid = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
# z_grid = np.array([0.004, 0.008, 0.02, 0.05])
# av_grid = [0, 0.5, 1.0, 1.2]

#Simplified parameters
tf_grid = [13.7]*u.Gyr
z_grid = np.array([0.02])
av_grid = [0]

test_subject = 'micro_Illustris'
input_file = os.path.join(os.getcwd(), '{}_input.txt'.format(test_subject))
output_file = os.path.join(os.getcwd(), '{}_output.csv'.format(test_subject))

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


#%%
#Illustris    
poly_ssfr = []
AV_poly = []
illustris_ssfr = []

illustris_data_path = 'F:/population-synthesis-toolkit-pst_2/pst/examples/data/Illustris'
# illustris_data_path = '/media/danieljl/SSD/population-synthesis-toolkit-pst_2/pst/examples/data/Illustris'
dust_polynomial = []
dust_polynomial_error = []
total_mass = []
av_poly = []
av_poly_error = []

output_data = csv.reader(open(output_file, 'r'))
for i, row in enumerate(output_data):
    #Illustris input sSFR
    ID = int(row[0])
    with fits.open(os.path.join(illustris_data_path, 'subhalo_{}_sfh.fits'.format(ID))) as hdul:
        lb_time = hdul[1].data['lookback_time']*u.Gyr
        Illustris_time = (t0-lb_time)[::-1]
        mass_formed = np.sum(hdul[3].data, axis=1)*u.Msun # sum over metallicities
        Illustris_mass_formed = np.cumsum(mass_formed[::-1])
        Illustris_Z = hdul[0].header['HIERARCH starmetallicity']
        total_star_mass = hdul[0].header['HIERARCH mass_stars']*10**10
        number_stars = hdul[0].header['HIERARCH len_stars']
    
    model_test = pst.models.Tabular_MFH(Illustris_time, 
                                Illustris_mass_formed, 
                                Z = np.ones(len(Illustris_time))*Illustris_Z) #Generating test-model
    fraction = np.logspace(-3, 0, len(t))
    real_fraction = 1- model_test.integral_SFR((t0-test_lookback_time))/model_test.integral_SFR(t0)
    real_age_of_fraction = np.interp(fraction, real_fraction, test_lookback_time) 
    
    lbt= .3
    lbt_index = abs(test_lookback_time.to_value()-lbt).argmin()
    r = real_fraction[lbt_index]
    
    
    #Polynomial output sSFR
    poly_dust = np.array([float(k) for k in row[6].split()])
    dust_polynomial.append(poly_dust)
    dust_polynomial_error.append(np.array([float(k) for k in row[7].split()]))
    
    A_lambda_obs = np.log10(poly_dust[4])/(-0.4)
    A_V_grid = np.linspace(0, 3, 301)
    R_V = 3.1        
    A_g = []
    for j in A_V_grid:
    	A_g.append( extinction.ccm89(obs_filters_wl, j, R_V)[4])       
    
    poly_mass_fraction = np.array([float(k) for k in row[3].split()])

    lbt_index = abs(test_lookback_time.to_value()-lbt).argmin()
    if len(poly_mass_fraction)>1000:
        p = poly_mass_fraction[lbt_index]
        # poly_mass_fraction_error = np.array([float(k) for k in row[4].split()])
        # p_error = poly_mass_fraction_error[lbt_index]
        poly_ssfr_i = p/lbt/1e9
        # poly_ssfr_error_i = p_error/lbt/1e9
        
        poly_ssfr.append( poly_ssfr_i)
        illustris_ssfr.append( r/lbt/1e9 )
        total_mass.append(mass_formed)
        if poly_ssfr_i<3e-12 and r/lbt/1e9>2e-10:
            bad_index = i
        AV_poly.append(np.interp(A_lambda_obs, A_g, A_V_grid) )
#%%
plt.figure()
plt.xlabel('sSFR poly')
plt.ylabel('sSFR Illustris')

AV_poly = np.array(AV_poly)
poly_ssfr = np.array(poly_ssfr)
illustris_ssfr = np.array(illustris_ssfr)

av_lim = 0
plt.scatter(poly_ssfr[AV_poly>=av_lim], illustris_ssfr[AV_poly>=av_lim], c='blue', alpha=.5, s=1)
plt.scatter(poly_ssfr[AV_poly<av_lim], illustris_ssfr[AV_poly<av_lim], c='red', alpha=.5, s=1)
# plt.scatter(dusty_ssfr_poly, dusty_ssfr_salim, c='red', alpha=1, s=1, label='AV>{}'.format(av_lim))

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-12, 1e-9)
plt.ylim(1e-12, 1e-9)
# plt.legend()
plt.plot(np.array([1e-13, 1e-8]), np.array([1e-13, 1e-8]), 'k-', alpha=.1)
plt.show()

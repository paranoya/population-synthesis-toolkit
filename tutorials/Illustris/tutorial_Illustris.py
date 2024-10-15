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
import extinction

t0 = 13.7 * u.Gyr  # time of observation 
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
lookback_time = t0*t_hat
test_lookback_time = lookback_time[::-1]

obs_filters = ['SLOAN_SDSS.u', 'SLOAN_SDSS.g', 'SLOAN_SDSS.r', 'SLOAN_SDSS.i', 'SLOAN_SDSS.z']
N_range = np.arange(1, 4)

ti_grid = [0]*u.Gyr
# tf_grid = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
# z_grid = np.array([0.004, 0.008, 0.02, 0.05])
# av_grid = [0, 0.5, 1.0, 1.2]

#Simplified parameters
tf_grid = [13.7]*u.Gyr
z_grid = np.array([0.02])
av_grid = [0]

test_subject = 'Illustris'
input_file = os.path.join(os.getcwd(), '{}_input.csv'.format(test_subject))
output_file = os.path.join(os.getcwd(), '{}_output.csv'.format(test_subject))

def get_flux_densities(model, ssp, obs_filters, Z_i, t, **kwargs):
    fnu = []
    sed = model.compute_SED(ssp, t_obs=13.7 * u.Gyr)
    for filter_name in obs_filters:
        photo = pst.observables.Filter.from_svo(filter_name)
        photo.interpolate(ssp.wavelength)
        spectra_flambda = ( sed/(4*np.pi*(10*u.pc)**2) )
        fnu_Jy, fnu_Jy_err = photo.get_fnu(spectra_flambda, spectra_err=None)
        fnu.append( fnu_Jy )
    return u.Quantity(fnu)

#%%
pst.fitting_module.compute_polynomial_models(input_file, output_file, obs_filters,
                                            ti_grid = ti_grid, tf_grid = tf_grid,
                                            z_grid = z_grid, av_grid = av_grid,
                                            N_range = N_range)

#%%
ssp = pst.SSP.PopStar(IMF="cha")
ssp.cut_wavelength(1000, 1600000)
obs_filters_wl  = []
for filter_name in obs_filters:
    photo = pst.observables.Filter.from_svo(filter_name)
    obs_filters_wl.append(photo.effective_wavelength().to_value())


#%%
#Illustris    
poly_ssfr = []
AV_poly = []
illustris_ssfr = []

illustris_data_path = 'F:/population-synthesis-toolkit-pst_2/pst/examples/data/Illustris'
# illustris_data_path = '/media/danieljl/SSD/population-synthesis-toolkit-pst_2/pst/examples/data/Illustris'

#Order written in old method:
    
        # f_output.write(mass_fraction +',')
        # f_output.write(target_ID +',')
        # f_output.write(mass_fraction_error +',')
        # f_output.write(total_mass +',')
        # f_output.write(age_of_fraction +',')
        # f_output.write(age_of_fraction_error +',')
        # f_output.write('{}'.format(av_model +','))
        # f_output.write('{}'.format(av_model_error +','))

f_illustris_data = csv.reader(open('Illustris_mfh_data.csv', 'r'))
Illustris_data_ID = []
Illustris_data_mass_formed = []
Illustris_data_total_mass = []
Illustris_data_metallicity = []

for i, row in enumerate(f_illustris_data):
    ID = int(row[0])   
    Illustris_data_ID.append(ID)
    Illustris_time = np.array([float(k) for k in row[1].split()])
    Illustris_data_mass_formed.append(np.array([float(k) for k in row[2].split()]))
    Illustris_data_total_mass.append(float(row[3]))
    Illustris_data_metallicity.append(float(row[4]))
    
dust_polynomial = []
dust_polynomial_error = []
av_poly = []
av_poly_error = []
z_poly = []
z_poly_error = []
ab_mags = []
id_list = []    
output_data = csv.reader(open(output_file, 'r'))
for i, row in enumerate(output_data):
    
    # if len(row) == 10: #Temporal patch to avoid corrupted inputs
    #Illustris input sSFR
    poly_mass_fraction = np.array([float(k) for k in row[2].split()])
    ID = int(row[0])   
    id_list.append(ID)
    poly_mass_fraction_error = np.array([float(k) for k in row[3].split()])
    poly_dust = np.array([float(k) for k in row[6].split()])
    dust_polynomial.append(poly_dust)
    dust_polynomial_error.append(np.array([float(k) for k in row[7].split()]))
    poly_met = np.array([float(k) for k in row[8].split()])
    z_poly.append(poly_met)
    z_poly_error.append(np.array([float(k) for k in row[9].split()]))
    
    masses = (1-poly_mass_fraction)
    model = pst.models.TabularCEM(times = t0-test_lookback_time, 
                                masses = masses*u.Msun, 
                                metallicities = np.full(masses.size, fill_value=poly_met))
    
    
    Fnu_total = get_flux_densities(model, ssp, obs_filters, poly_met, t)
    f = Fnu_total.to_value() #Jy
    AB_mag = -2.5*np.log10(f/3631)
    ab_mags.append(AB_mag)
    
    index = int(np.where(np.array(Illustris_data_ID) == ID)[0])

    Illustris_mass_formed = Illustris_data_mass_formed[index]
    Illustris_Z = Illustris_data_metallicity[index]
    total_star_mass = Illustris_data_total_mass[index]
    model_test = pst.models.TabularCEM(times = Illustris_time*u.Gyr, 
                                masses = Illustris_mass_formed*u.Msun, 
                                metallicities=np.full(Illustris_mass_formed.size, fill_value=Illustris_Z))
                                
    fraction = np.logspace(-3, 0, len(t))
    real_fraction = 1- model_test.stellar_mass_formed((t0-test_lookback_time))/model_test.stellar_mass_formed(t0)
    real_age_of_fraction = np.interp(fraction, real_fraction, test_lookback_time) 
    
    lbt= .3
    lbt_index = abs(test_lookback_time.to_value()-lbt).argmin()
    r = real_fraction[lbt_index]
    
    #Polynomial output sSFR
    
    A_lambda_obs = np.log10(poly_dust[4])/(-0.4)
    A_V_grid = np.linspace(0, 3, 301)
    R_V = 3.1        
    A_g = []
    for j in A_V_grid:
    	A_g.append( extinction.ccm89(np.array(obs_filters_wl), j, R_V)[4])       
    

    lbt_index = abs(test_lookback_time.to_value()-lbt).argmin()
    if len(poly_mass_fraction)>1000:
        p = poly_mass_fraction[lbt_index]
        # poly_mass_fraction_error = np.array([float(k) for k in row[4].split()])
        # p_error = poly_mass_fraction_error[lbt_index]
        poly_ssfr_i = p/lbt/1e9
        # poly_ssfr_error_i = p_error/lbt/1e9
        
        poly_ssfr.append( poly_ssfr_i)
        illustris_ssfr.append( r/lbt/1e9 )
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

#%%
u_poly = np.array([m[0] for m in ab_mags])
g_poly = np.array([m[1] for m in ab_mags])
r_poly = np.array([m[2] for m in ab_mags])

plt.figure()
plt.title('Polynomial')
plt.scatter( u_poly-g_poly, g_poly-r_poly, s=2)
plt.ylabel('g-r')
plt.xlabel('u-g')
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
plt.show()
    
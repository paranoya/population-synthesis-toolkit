# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:17:01 2022

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
from pst import pst_tools_v2 as pst_tools_v2

start_time = tic.time()

# Observables
t0 = 13.7 * u.Gyr  # time of observation 
t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
lookback_time = (t0-t)[::-1] #t0 --> 0

#L_err = np.array([2e-2, 1e-2, 1e-2, 1e-2, 1e-2]) #Errors on the Luminosity
#ugriz_wl = np.array([3543., 4770., 6231., 7625., 9134.])
R_V = 3.1

# SSPs
ssp_pop = pst.SSP.PopStar(IMF="sal_0.15_100")
#ssp_pypop = pst.SSP.PyPopStar(IMF="KRO")
ssp = ssp_pop

script_path = os.getcwd()
#%%    
#MODEL PARAMETERS
obs_errors = True #Adds observational errors
allow_negative_sfr = False #Allows negative SFRs
free_time = True #ti and tf as variables
run_MC_loops = False
save_plots = False

single_model = False #Only uses 1 model from the models grid
model_dust=0
model_z=2
single_target = True #Only runs for 1 target from the targets grid
target_dust=0
target_z=2

if run_MC_loops:
    n_mc = 1
else:
    n_mc = 0
print('Initiating code')
print()
print('Code parameters:')
print('Obs. errors: ', obs_errors)
print('Negative SFR: ', allow_negative_sfr)
print('ti and tf as fix parameters: ', free_time)
print()

t_grid_steps = t0*(1-np.logspace(-3, 0, 4))

if free_time == False:
    t_grid_steps = t0*(1-np.array([0,1]))
t_grid = [(b, c) for b in t_grid_steps for c in t_grid_steps
              if b.to_value()<c.to_value()]

t_start_values = [0]*u.Gyr
t_end_values = [13.7]*u.Gyr
#t_end_values = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr

if free_time == False:
    t_end_values = [1]*t0
    t_start_values = [0]*t0
t_grid = [(b, c) for b in t_start_values for c in t_end_values
              if b.to_value()<c.to_value()]
# Primordial polynomia
N_min = 1
#N_max = len(obs_filters)
N_max = 3
N_range = np.arange(N_min, N_max+1)  # polynomial degree

##############################################################
#DUST EXTINCTION MODELS: 

#MODELS GRID
model_A_V_grid = [0, 0.5, 1.0, 1.2]
#model_z_grid = np.array([0.004, 0.008, 0.02, 0.05])
#model_z_grid = np.array([0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
model_z_grid = np.array([0.02])
if single_model:
    #ONLY 1 MODEL        
    model_A_V_grid = [model_A_V_grid[model_dust]]   
    model_z_grid = [model_z_grid[model_z]]
    
##############################################################
#TARGETS GRID
target_A_V_grid = np.array([0, 0.5, 0.7, 1.0, 1.2])
target_z_grid = np.array([0.004, 0.008, 0.02, 0.05])
    
if single_target:
    #Only 1 target
    target_A_V_grid= [target_A_V_grid[target_dust]]
    target_z_grid = [target_z_grid[target_z]]
    
n_models = len(model_A_V_grid)*len(model_z_grid)*len(t_grid)*len(N_range)
## MFH TARGET
#We have a couple of synthetic MFH to choose: gaussian, tau, square


#mfh_test_type = 'Illustris'
target_mfh = 'salim'
n_models = 1 #Models computed per run
max_per_folder = 100 #Max amount per folder
n_runs = 4 #Number of runs --> Helps to divide the work into different simultaneus chunks
current_run = 0 #Current run (0 --> n_runs-1)

obs_filters = ['GFUV', 'GNUV', 'u', 'g', 'r', 'i', 'z']
#obs_filters = ['u', 'g', 'r', 'i', 'z']
obs_filters_wl = []
for obs in obs_filters:
    filter_resp, wl_filter = pst.observables.Filter.get_filt(obs)
    wl = np.sum(wl_filter*filter_resp)/np.sum(filter_resp)
    obs_filters_wl.append(wl.to_value())
obs_filters_wl = np.array(obs_filters_wl)

#Swap between Illustris or salim
real_models = pst_tools_v2.get_real_models(ssp, obs_filters, obs_filters_wl, n_models, n_runs, current_run, target_mfh)

#%% 
##################################################################################
##################################################################################
#Main code
output_path = os.path.join(os.getcwd(), 'results_{}'.format(target_mfh)) #Change folder name as desired
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
#Saving the t in Gyrs
with open(os.path.join(output_path, 't_Gyr.txt'), 'w+') as f_MFH:
    if os.stat(os.path.join(output_path, 't_Gyr.txt')).st_size == 0:
        f_MFH.write(" ".join(str(x) for x in t.to_value()) + "\n")
     
for number_model, subhalo in enumerate(real_models):
    print('Computing Model #{} of {}'.format(number_model, len(real_models)))
    
    #Dividing the output into files    
    if number_model % max_per_folder == 0:
        if number_model==0:
            min_ID= 0
        else:
            min_ID+=1
    
    run_folder = 'models_{}_{}'.format(current_run*len(real_models)+min_ID*max_per_folder, 
                         current_run*len(real_models)+min((min_ID+1)*max_per_folder-1, len(real_models)-1))
                        
    run_path = os.path.join(output_path, run_folder)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    
    target = real_models[subhalo]
    polynomial_model_mc = []
    likelihood_mc = []
    metallicities_mc = []
    dust_mc = []
    
    if target_mfh == 'Illustris':
        model_test = pst.models.Tabular_MFH(target['time'], 
                                            target['mass_formed'], 
                                            Z = target['metallicity']) #Generating test-model
        
        target_A_lambda = extinction.ccm89(obs_filters_wl, target['A_V'], R_V)
        dust_extinction_target = np.power(10, -target_A_lambda/2.5)
        
        fraction = np.logspace(-3, 0, len(t))
        real_fraction = 1 - model_test.integral_SFR((t0-lookback_time))/model_test.integral_SFR(t0)
        real_age_of_fraction = np.interp(fraction, real_fraction, lookback_time) 
        
    for montecarlo_loop in range(0, n_mc+1): #n_mc=0 to ignore the montecarlo test
#        print('Computing Montecarlo loop #{} of {}'.format( 
#                      montecarlo_loop, n_mc))

        time_start_loop = tic.time()
        
        polynomial_model_i = []
        likelihood_i = []
        metallicities_i = []
        dust_i = []
               
        #RANDOM OBS ERROR + EXTINCTION
        random_noise = np.random.normal(0, 1, len(obs_filters)) #Normal noise for each wavelength

        L_obs = target['L_obs']
        error_L_obs = target['error_L_obs']
        
        if montecarlo_loop > 0: # MC=0 is without errrors              
            L_obs += random_noise*error_L_obs #Applying the random noise to each luminosity error
        
        L_obs = L_obs/error_L_obs  #Normalizing by the luminosity errors

        #PST                     
        time_before_pst = tic.time()
        first_loop_timer=True
        previous_time = time_before_pst
        for Z_i_index, Z_i in enumerate(model_z_grid):   
            
            for av_index, A_V_model in enumerate(model_A_V_grid):
                
                model_A_lambda = extinction.ccm89(obs_filters_wl, A_V_model, R_V)
                dust_extinction = np.power(10, -model_A_lambda/2.5)
#                    print('Computing model #{} of {}'.format( 
#                          (av_index+1)+(Z_i_index)*len(model_A_V_grid), n_models))
                for N in N_range:
                    
                    for t_initial, t_final in t_grid:
                        previous_time = tic.time()
                        basis = pst.models.Polynomial_MFH_fit(N, ssp, obs_filters, t0, Z_i,                                                  
                                                   dust_extinction, error_L_obs,
                                                   t_hat_start=1-t_initial/t0, 
                                                   t_hat_end = 1-t_final/t0)    #Creating the basis               
                        #True polynomia
                        coeffs = basis.fit(L_obs) #Generating the coefficients c
                        poly_fit = pst.models.Polynomial_MFH(Z=Z_i,
                                                   coeffs=coeffs,
                                                   S=basis.lstsq_solution,
                                                   t_hat_start=1-t_initial/t0, 
                                                   t_hat_end = 1-t_final/t0) #Generating the polynomial model

                        L_model = pst_tools_v2.generate_luminosities(poly_fit, ssp, obs_filters)*dust_extinction / error_L_obs #Luminosities of the model
                        #Saving the uncorrected models (before positive-SFR fit)
                        uncorrected = poly_fit
                        uncorrected_L_model = L_model
                        
                        t_cut = t.clip(t_initial, t_final) #Time limited by the free initial-final time parameters
                        sfr = uncorrected.SFR(t_cut)
                        
                        zeros = np.where(sfr[:-1]*sfr[1:] < 0)[0] #Locating zeros of the uncorrected SFR
                        
                        t_zeros = np.r_[t_initial,np.sqrt(t_cut[zeros]*t_cut[zeros+1]),t_final]
                        
                        if t_zeros[0]==t_zeros[1]: #This patch avoids when it detects a zero at t_initial or t_final and duplicates that point
                            t_zeros = t_zeros[1:]  
                        if t_zeros[-1]==t_zeros[-2]:
                            t_zeros = t_zeros[:-1]
                        
                        if allow_negative_sfr: 
                            t_zeros = np.r_[t_initial, t_final]
                        for t_start, t_end in zip(t_zeros[:-1], t_zeros[1:]): #For each combination of start-end of the zeros
                            if len(t_zeros)==2: #If no zeros: we are done with the uncorrected
                                poly_fit = uncorrected
                                L_model = uncorrected_L_model
                                
                            else:     
                                poly_fit = pst.models.Polynomial_MFH(Z=Z_i, 
                                            t_hat_start=(1-t_start/t0), t_hat_end=(1-t_end/t0), 
                                            coeffs=coeffs, S=basis.lstsq_solution) #New polynomial fit with the new time-limits
                                                                    
                                L_model = pst_tools_v2.generate_luminosities(poly_fit, ssp, obs_filters)*dust_extinction / error_L_obs
                            
                                if poly_fit.t_hat_end == poly_fit.t_hat_start:
                                        print('ERROR!!')
                                        print('t_initial=', t_initial, 't_final=', t_final, 't_start=', t_start, 't_end=', t_end)
                                        print('t_zeros', t_zeros)
                                        print('t_cut')
                            norm = np.sum(L_obs*L_model) / np.sum(L_model**2) #Normalization factor for the coefficients
                            
                            polynomial_model = pst.models.Polynomial_MFH(Z=Z_i, 
                                            t_hat_start=(1-t_start/t0), 
                                            t_hat_end=(1-t_end/t0), coeffs=coeffs*norm, 
                                            compute_sigma=True, S=basis.lstsq_solution)

                            likelihood = np.exp(-np.sum((L_model*norm-L_obs)**2/2))
                            
                            polynomial_model_i.append(polynomial_model)
                            likelihood_i.append(likelihood)  
                            metallicities_i.append(Z_i)
                            dust_i.append(dust_extinction)
                            
#                            if first_loop_timer:
#                                print('LOOP END: ', tic.time()-previous_time)
#                                previous_time = tic.time()
#                                first_loop_timer=False
        if montecarlo_loop == 0:
            polynomial_model_no_errors = polynomial_model_i
            likelihood_no_errors = likelihood_i
            metallicities_no_errors = metallicities_i
            dust_no_errors = dust_i
        else:
            polynomial_model_mc.append(polynomial_model_i)
            likelihood_mc.append(likelihood_i)
            metallicities_mc.append(metallicities_i)
            dust_mc.append(dust_i)
                        
        pst_time = tic.time() - time_before_pst
        
    previous_time = tic.time()
    #%%
    def weighted_mean(x, w):
        return (np.array(w)*np.array(x) / np.sum(np.array(w))).sum(axis=0)
       
    def old_std(x, w):
        mu= weighted_mean(x, w)
        return np.sqrt((np.array(w)*np.array(x)**2 / np.sum(np.array(w))).sum(axis=0)-mu**2)
    
    def yago_std(x, sigma, w):
        mu= weighted_mean(x, w)
        return np.sqrt((np.array(w)*(np.array(x)-mu)**2+ np.array(sigma)**2).sum(axis=0)/ np.sum(np.array(w)))
    
    #t(M) y M(t)        
    previous_time = tic.time()
    
    #MODELO SIN ERRORES             
    model_w_no_errors = []
    age_of_fraction_i_no_errors = []
    fraction_i_no_errors = []
    new_sigma2 = []
    sigma_i = []
    for i, model in enumerate(polynomial_model_no_errors): #For each model of this MC loop:
        mass, sigma = model.mass_formed_since(t0-lookback_time, get_sigma=True) #mass formed and its sigma
        c, M, S = model.mass_formed_since(t0-lookback_time, get_components=True)
        w_i = np.array([likelihood_no_errors[i]])
        new_sigma2.append((w_i*(np.matmul(S.T,  M))**2).sum(axis=0)/mass[-1]**2) 
        
        fraction_model = mass/mass[-1]
        fraction_i_no_errors.append(fraction_model)
        sigma_i.append(sigma/mass[-1])
        model_w_no_errors.append( w_i )       
        fraction = np.logspace(-3, 0, len(t))
        age_of_fraction_i_no_errors.append( np.interp(fraction, fraction_model, lookback_time) )
    
    paper_sigma = np.sqrt(np.array(new_sigma2).sum(axis=0) / np.sum(np.array(model_w_no_errors)).sum(axis=0))
    
    previous_time = tic.time()                 
    fraction_no_errors = weighted_mean(fraction_i_no_errors, model_w_no_errors)
    
    old_fraction_no_errors_std = old_std(fraction_i_no_errors, model_w_no_errors) 
    paper_fraction_no_errors_std = paper_sigma
    yago_fraction_no_errors_std = yago_std(fraction_i_no_errors, sigma_i, model_w_no_errors) 
    
    age_of_fraction_no_errors = weighted_mean(age_of_fraction_i_no_errors, model_w_no_errors) 
    age_of_fraction_no_errors_std = old_std(age_of_fraction_i_no_errors, model_w_no_errors) 
                      
    #MONTECARLO
    if run_MC_loops:
        age_of_fraction_mc = []
        age_of_fraction_mc_std = []
        fraction_mc = []
        fraction_mc_std = []
        
        w_mc = []
        for mc in range(0, n_mc):
           
            age_of_fraction_i = []
            fraction_mc_i = []
            w_i = []
            for i, model in enumerate(polynomial_model_mc[mc]): #For each model of this MC loop:
                        
                mass, sigma = polynomial_model_mc[mc][i].mass_formed_since(t0-lookback_time, get_sigma=True) #mass formed and its sigma
                fraction_model = mass/mass[-1]
                fraction_mc_i.append(fraction_model)
                w_i.append( np.array([likelihood_mc[mc][i]]) )       
                age_of_fraction_i.append( np.interp(fraction, fraction_model, lookback_time) )
                        
                
            age_of_fraction_mc.append(weighted_mean(age_of_fraction_i, w_i))
            age_of_fraction_mc_std.append(yago_std(age_of_fraction_i, w_i))
            fraction_mc.append(weighted_mean(fraction_mc_i, w_i))
            fraction_mc_std.append(yago_std(fraction_mc_i, w_i))
            w_mc.append(w_i)
    print('t after after pst stuff: ', tic.time()-previous_time)
    previous_time = tic.time()
    #%%
    #PLOTS
    def chi2(x,y, mean, std):
        return ((x-mean[y])/std[y])**2
           
    def fake_log(x, pos):
        'The two args are the value and tick position'
        return r'$10^{%d}$' % (x)
    
    def plot_map(z, title, xlabel, ylabel, cmap, savefig):

        fig, (ax) = plt.subplots()
        im = ax.imshow(z, vmin=0, vmax=3, extent=[np.log10(0.0137), np.log10(13.7), 3, 0], cmap=cmap, aspect='auto')
        fig.colorbar(im)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(fake_log)
        ax.yaxis.set_major_formatter(fake_log)
        
        if savefig==True:
            if save_plots:   
                fig.savefig(os.path.join(os.getcwd(), 'cmap_{}.png'.format(subhalo)))

    #Computing recovered metallicity and dust extinction 
    weighted_z_no_errors = 0
    weighted_d_no_errors = 0
    for i, model in enumerate(polynomial_model_no_errors):
        w = np.array([likelihood_no_errors[i]])/np.sum(likelihood_no_errors)
        weighted_z_no_errors+=metallicities_no_errors[i]*w
        weighted_d_no_errors+=dust_no_errors[i]*w
    
    std_weighted_z_no_errors = 0
    std_weighted_d_no_errors = 0
    for i, model in enumerate(polynomial_model_no_errors):
        w = np.array([likelihood_no_errors[i]])/np.sum(likelihood_no_errors)
        std_weighted_z_no_errors+=np.array(w)*metallicities_no_errors[i]**2 
        std_weighted_d_no_errors+=np.array(w)*dust_no_errors[i]**2 
    std_weighted_z_no_errors = np.sqrt(std_weighted_z_no_errors-weighted_z_no_errors**2)
    std_weighted_d_no_errors = np.sqrt(std_weighted_d_no_errors-weighted_d_no_errors**2)
    
#    print('Target with Z = ', target_z_grid[number_model], 'and D = ', dust_extinction_target)
#    print('Weighted model (NO ERRORS) with Z = ', weighted_z_no_errors, 'and D = ', weighted_d_no_errors)
#    print(f'Best Chi = {np.sqrt(-2*np.log(max(likelihood_no_errors))/5):.4g}')     
    #Saving Z and D
    
    if target_mfh == 'Illustris':
        file_ID = subhalo[8:]
        with open(os.path.join(run_path, 'Z_D_{}.txt'.format(file_ID)), 'w+') as f_ZD:
            f_ZD.write('real_Z model_Z std_Z real_ugriz model_ugriz std_ugriz'+'\n')
            f_ZD.write('{} {} {} {} {} {} {}'.format(file_ID, 
                       np.round(real_models[subhalo]['metallicity'], 5), np.round(weighted_z_no_errors[0], 5), np.round(std_weighted_z_no_errors[0], 5), 
                      str(np.round(dust_extinction_target, 3))[1:-1], str(np.round(weighted_d_no_errors, 3))[1:-1], str(np.round(std_weighted_d_no_errors, 3))[1:-1]+'\n'))

    elif target_mfh == 'input_luminosities':
        file_ID = target['objID']
        with open(os.path.join(run_path, 'real_mfh_{}.txt'.format(file_ID)), 'w+') as f_ZD:
            f_ZD.write('lgm_tot_p16 lgm_tot_p50 lgm_tot_p84 specsfr_tot_p16 specsfr_tot_p50 specsfr_tot_p84'+'\n')
            f_ZD.write('{} {} {} {} {} {} {}'.format(file_ID, 
                       str(real_models[subhalo]['lgm_tot_p16']), str(real_models[subhalo]['lgm_tot_p50']), str(real_models[subhalo]['lgm_tot_p84']),
                       str(real_models[subhalo]['specsfr_tot_p16']), str(real_models[subhalo]['specsfr_tot_p50']), str(real_models[subhalo]['specsfr_tot_p84'])+'\n'))
                
        with open(os.path.join(run_path, 'Z_D_{}.txt'.format(file_ID)), 'w+') as f_ZD:
            f_ZD.write('model_ID real_Z model_Z std_Z'+'\n')
            f_ZD.write('{} {} {} {}'.format(file_ID, 
                       str(np.round(target_z_grid[number_model], 5)), str(np.round(weighted_z_no_errors[0], 5)), str(np.round(std_weighted_z_no_errors[0], 5))+'\n'))

    elif target_mfh == 'salim':
        file_ID = target['IAUNAME']
        with open(os.path.join(run_path, 'real_mass_fraction_{}.txt'.format(file_ID)), 'w+') as f_ZD:
            f_ZD.write('IAUNAME nasa_MASS logM logM_sigma logSFR logSFR_sigma'+'\n')
            f_ZD.write('{} {} {} {} {} {}'.format(file_ID, 
                       str(real_models[subhalo]['stellar_mass']),str(real_models[subhalo]['logM']),  str(real_models[subhalo]['logM_sigma']), str(real_models[subhalo]['logSFR']), 
                       str(real_models[subhalo]['logSFR_sigma'])+'\n'))
                
        with open(os.path.join(run_path, 'Z_D_{}.txt'.format(file_ID)), 'w+') as f_ZD:
            f_ZD.write('nasaID real_ext_nasa real_met_nasa real_ext_salim real_ext_salim_error'+'\n')
            f_ZD.write('{} {} {} {} {}'.format(file_ID, 
                       str(real_models[subhalo]['extinction']), str(real_models[subhalo]['metallicity']), str(real_models[subhalo]['A_V']), str(real_models[subhalo]['A_V_sigma'])+'\n'))

    #Saving mass fraction
    with open(os.path.join(run_path, 'mass_fraction_{}.txt'.format(file_ID)), 'w+') as f_MFH:
        f_MFH.write('M(t) and delta_M(t)'+'\n')
        f_MFH.write(" ".join(str(x) for x in fraction_no_errors) + "\n")
        f_MFH.write(" ".join(str(x) for x in old_fraction_no_errors_std) + "\n")
        f_MFH.write(" ".join(str(x) for x in paper_fraction_no_errors_std) + "\n")
        f_MFH.write(" ".join(str(x) for x in yago_fraction_no_errors_std) + "\n")
    #Saving age of fraction
    with open(os.path.join(run_path, 'age_of_fraction_{}.txt'.format(file_ID)), 'w+') as f_MFH:
        f_MFH.write('t(M) and delta_t(M)'+'\n')
        f_MFH.write(" ".join(str(x) for x in age_of_fraction_no_errors) + "\n")
        f_MFH.write(" ".join(str(x) for x in age_of_fraction_no_errors_std) + "\n")        
 #%%
    print("--- total time = %s seconds ---" % (round(tic.time() - start_time, 3)))
    print("--- time pst = %s seconds ---" % (round(pst_time, 3)))

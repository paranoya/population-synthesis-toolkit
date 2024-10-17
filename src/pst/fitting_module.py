"""

"""
import numpy as np
from astropy import units as u
import pst
import time as tic
import extinction
from pst.models import ChemicalEvolutionModel
import csv

def compute_polynomial_models(input_file, output_file, obs_filters, 
                              **kwargs):
    
    # Observables
    t0 = 13.7 * u.Gyr  # time of observation 
    t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
    t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
    lookback_time = (t0-t)[::-1] #t0 --> 0
    R_V = 3.1
    
    t_start_values = kwargs.get('ti_grid', [0]*u.Gyr)
    t_end_values = kwargs.get('tf_grid', [13.7]*u.Gyr)
    z_grid = kwargs.get('z_grid', np.array([0.02]))
    av_grid = kwargs.get('av_grid', [0])
    N_range = kwargs.get('N_grid', np.arange(1, 4))
    
    # SSPs
    ssp_pop = pst.SSP.PopStar(IMF="sal")
    ssp = ssp_pop
    
    #%%    
    #MODEL PARAMETERS
    allow_negative_sfr = False #Allows negative SFRs
    
    print('Initiating code')
    print()
   
    t_grid = [(b, c) for b in t_start_values for c in t_end_values
                  if b.to_value()<c.to_value()]
    
    ##############################################################
    ssp.cut_wavelength(1000, 1600000)
    obs_filters_wl = []
    for name in obs_filters:
        photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = name)
        obs_filters_wl.append(photo.effective_wavelength().to_value())
    obs_filters_wl = np.array(obs_filters_wl)     
    # print('effective_wavelengths: ', obs_filters_wl)       
    #Computing real models
    
    #NEED read the lines [target_ID_name [Fnu] [Fnu_error]]
    input_photo = {}
    
    content = csv.reader(open(input_file, "r"))
    for i, row in enumerate(content):
        target_ID = int(row[0])
        input_photo[target_ID] = {}
        input_photo[target_ID]['Fnu_obs'] = np.array([float(k) for k in row[1].split()])
        input_photo[target_ID]['error_Fnu_obs'] = np.array([float(k) for k in row[2].split()])
    
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
    ##################################################################################
    ##################################################################################
    #Main code
    f_output = open(output_file, 'w+')
    for number_model, target_ID in enumerate(input_photo):
        print('Computing Model #{} of {}'.format(number_model+1, len(input_photo)))
                        
        target = input_photo[target_ID]

        model_poly = []
        chi2_poly = []
        metallicities_poly = []
        dust_extinction_poly = []
               
        #RANDOM OBS ERROR + EXTINCTION
    
        Fnu_obs = np.array(target['Fnu_obs'])
        error_Fnu_obs = np.array(target['error_Fnu_obs'])        
        Fnu_obs = Fnu_obs/error_Fnu_obs  #Normalizing by the luminosity errors
    
        #PST     
        time_before_pst = tic.time()                
        for Z_i_index, Z_i in enumerate(z_grid):   
            
            for av_index, A_V_model in enumerate(av_grid):
                
                model_A_lambda = extinction.ccm89(obs_filters_wl, A_V_model, R_V)
                dust_extinction = np.power(10, -model_A_lambda/2.5)
    
                for N in N_range:
                    for t_initial, t_final in t_grid:
                        basis = pst.models.Polynomial_MFH_fit(N, ssp, obs_filters, obs_filters_wl, t, t0, Z_i,                                                  
                                                   dust_extinction, error_Fnu_obs,
                                                   t_hat_start=1-t_initial/t0, 
                                                   t_hat_end = 1-t_final/t0)    #Creating the basis               
                        #True polynomia
                        coeffs = basis.fit(Fnu_obs) #Generating the coefficients c
                        poly_fit = pst.models.Polynomial_MFH(Z=Z_i,
                                                   coeffs=coeffs,
                                                   S=basis.lstsq_solution,
                                                   t_hat_start=1-t_initial/t0, 
                                                   t_hat_end = 1-t_final/t0) #Generating the polynomial model
    
                        Fnu_model = get_flux_densities(poly_fit, ssp, obs_filters, Z_i, t)*dust_extinction / error_Fnu_obs #Luminosities of the model
                        
                        #Saving the uncorrected models (before positive-SFR fit)
                        uncorrected = poly_fit
                        uncorrected_Fnu_model = Fnu_model
                        
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
                                Fnu_model = uncorrected_Fnu_model
                                
                            else:     
                                poly_fit = pst.models.Polynomial_MFH(Z=Z_i, 
                                            t_hat_start=(1-t_start/t0), t_hat_end=(1-t_end/t0), 
                                            coeffs=coeffs, S=basis.lstsq_solution) #New polynomial fit with the new time-limits
                                                                    
                                Fnu_model = get_flux_densities(poly_fit, ssp, obs_filters, Z_i, t)*dust_extinction / error_Fnu_obs
                                if poly_fit.t_hat_end == poly_fit.t_hat_start:
                                        print('ERROR!!')
                                        print('t_initial=', t_initial, 't_final=', t_final, 't_start=', t_start, 't_end=', t_end)
                                        print('t_zeros', t_zeros)
                                        print('t_cut')
                            
                            #Anti Inf in negative SFR(t) patch
                            if np.any(np.array(Fnu_model)) == False:
                                norm = 1
                                chi2 = 1e10
                            else:
                                norm = np.sum(Fnu_obs*Fnu_model) / np.sum(Fnu_model**2) #Normalization factor for the coefficients
                                chi2 = np.sum((Fnu_model*norm-Fnu_obs)**2)
                            
                            polynomial_model = pst.models.Polynomial_MFH(Z=Z_i, 
                                            t_hat_start=(1-t_start/t0), 
                                            t_hat_end=(1-t_end/t0), coeffs=coeffs*norm, 
                                            compute_sigma=True, S=basis.lstsq_solution)
                                
                            model_poly.append(polynomial_model)
                            chi2_poly.append(chi2)  
    
                            metallicities_poly.append(Z_i)
                            dust_extinction_poly.append(dust_extinction)
                        
        pst_time = tic.time() - time_before_pst
        
        #%%
        def weighted_mean(x, w):
            return np.dot(w, x)
           
        def std(x, w):
            mu= weighted_mean(x, w)
            return np.sqrt(np.dot(w, np.power(x, 2))-np.power(mu, 2))
                
        #t(M) y M(t)           
        norm_chi2_poly = chi2_poly - min(chi2_poly)
        likelihood_poly = np.array(np.exp(-norm_chi2_poly/2))
        likelihood_poly[np.isnan(likelihood_poly)] = 0
        norm_weights = likelihood_poly/np.sum(likelihood_poly)
        
        #MODELO SIN ERRORES             
        age_of_fraction_i = []
        fraction_i = []
        sigma2 = []
        sigma_i = []
        total_mass_i = []
        for i, model in enumerate(model_poly): #For each model of this MC loop:
            mass, sigma = model.mass_formed_since(t0-lookback_time, get_sigma=True) #mass formed and its sigma
            c, M, S = model.mass_formed_since(t0-lookback_time, get_components=True)        
            total_mass = mass[-1].to_value()
            total_mass_i.append(total_mass)        
            fraction_model = np.array(mass/total_mass)
            sigma_model = np.array(sigma/total_mass)
            sigma2.append((norm_weights[i]*(np.matmul(S.T,  M))**2).sum(axis=0)/total_mass**2) 
            fraction = np.logspace(-3, 0, len(t))
            age_of_fraction_model = np.interp(fraction, fraction_model, lookback_time)
            
            fraction_i.append(fraction_model)
            sigma_i.append(sigma_model)
            age_of_fraction_i.append( age_of_fraction_model )
        
                     
        mass_fraction = weighted_mean(fraction_i, norm_weights)
        mass_fraction_std = np.sqrt(np.array(sigma2).sum(axis=0))  
        weighted_total_mass = weighted_mean(total_mass_i, norm_weights)

        age_of_fraction = weighted_mean(age_of_fraction_i, norm_weights) 
        age_of_fraction_std = std(age_of_fraction_i, norm_weights) 
                                     
        #Computing recovered metallicity and dust extinction 
        weighted_z = 0
        weighted_d = 0
        std_weighted_z = 0
        std_weighted_d = 0
        for i, model in enumerate(model_poly):
            w = np.array([likelihood_poly[i]])/np.sum(likelihood_poly)
            weighted_z+=metallicities_poly[i]*w
            weighted_d+=dust_extinction_poly[i]*w
            std_weighted_z+=np.array(w)*metallicities_poly[i]**2 
            std_weighted_d+=np.array(w)*dust_extinction_poly[i]**2 
        std_weighted_z = np.sqrt(std_weighted_z-weighted_z**2)
        std_weighted_d = np.sqrt(std_weighted_d-weighted_d**2)
        
        #%%  
        #Saving results def
        mass_fraction = " ".join(str(x) for x in mass_fraction)
        mass_fraction_error = " ".join(str(x) for x in mass_fraction_std)
        total_mass = "{}".format(np.round(weighted_total_mass, 8))
        age_of_fraction = " ".join(str(x) for x in age_of_fraction)
        age_of_fraction_error = " ".join(str(x) for x in age_of_fraction_std)
        av_model = str(np.round(weighted_d, 3))[1:-1]
        av_model_error = str(np.round(std_weighted_d, 3))[1:-1]
        metallicity_model = np.round(weighted_z[0], 5)
        metallicity_model_error = np.round(std_weighted_z[0], 5)
        
        f_output.write('{}'.format(target_ID) +',')
        f_output.write(total_mass +',')
        
        f_output.write(mass_fraction +',')       
        f_output.write(mass_fraction_error +',')
        
        f_output.write(age_of_fraction +',')
        f_output.write(age_of_fraction_error +',')
        
        f_output.write('{}'.format(av_model +','))
        f_output.write('{}'.format(av_model_error +','))

        f_output.write(str(metallicity_model)+',')
        f_output.write(str(metallicity_model_error)+'\n')     
                        
        print("--- time computing model = %s seconds ---" % (round(pst_time, 3)))
        
    f_output.close()

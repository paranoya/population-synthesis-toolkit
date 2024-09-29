"""

"""
import numpy as np
from astropy import units as u
import pst
import time as tic
import os
import extinction


#-------------------------------------------------------------------------------
class Polynomial_MFH_fit: #Generates the basis for the Polynomial MFH
    def __init__(self, N, ssp, obs_filters, obs_filters_wl, t, t_obs, Z_i, dust_extinction, 
                 error_Fnu_obs, **kwargs):
        self.t_obs = t_obs.to_value()
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        
        primordial_coeffs = []
        primordial_Fnu = []
        for n in range(N):
            
            c = np.zeros(N)
            c[n] = 1
            
            primordial_coeffs.append(c)
            
            fnu = []
            p = pst.models.Polynomial_MFH(Z=Z_i, t_hat_start = self.t_hat_start,
                                          t_hat_end = self.t_hat_end,
                                          coeffs=c)

            cum_mass = np.cumsum(p.stellar_mass_formed(t))
            z_array = Z_i*np.ones(len(t))
            sed, weights = ssp.compute_SED(t, cum_mass, z_array)

            for i, filter_name in enumerate(obs_filters):
                photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = filter_name)
                fnu_Jy, fnu_Jy_err = photo.get_fnu(sed, spectra_err = None)
                fnu.append( fnu_Jy )

            primordial_Fnu.append(u.Quantity(fnu))
        primordial_Fnu = np.array(primordial_Fnu)*dust_extinction / error_Fnu_obs
        
        self.p = p
        self.sed = sed
        self.lstsq_solution = np.matmul(
            np.linalg.pinv(np.matmul(primordial_Fnu, np.transpose(primordial_Fnu))),
            primordial_Fnu)
        self.primordial_coeffs = np.array(primordial_coeffs)
        self.primordial_Fnu = np.array(primordial_Fnu)
        self.primordial_Fnu = primordial_Fnu

    def fit(self, Fnu_obs, **kwargs):

        c = np.matmul(self.lstsq_solution,
                      Fnu_obs)              
        return c


#-------------------------------------------------------------------------------
class Polynomial_MFH(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.t0 = kwargs.get('t0', 13.7*u.Gyr)
        self.M0 = kwargs.get('M_end', 1*u.Msun)
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        self.coeffs = kwargs['coeffs']
        self.S = kwargs.get('S', False)
        
        ChemicalEvolutionModel.__init__(self, **kwargs)
    
    #If you want the raw components: model.xxxx(t, get_components=True)
    #If you want the observable + error: model.xxxx(t, get_sigma=True)
    #If you want only the observable: model.xxxx(t)
    def mass_formed_since(self, cosmic_time, **kwargs):
        t_hat_present_time = (1 - self.t0/self.t0).clip(self.t_hat_end, self.t_hat_start)
        t_hat_since = (1 - cosmic_time/self.t0).clip(self.t_hat_end, self.t_hat_start)
        self.get_sigma = kwargs.get('get_sigma', False)
        self.get_components = kwargs.get('get_components', False)
        self.fit_components = kwargs.get('fit_components', None)
        
        if self.fit_components is None:  
            M=[]
            N = len(self.coeffs)
            for n in range(1, N+1):
                M.append(t_hat_since**n - t_hat_present_time**n)

            self.M = u.Quantity(M)
        else:
            c, M, S = self.fit_components
            return self.M0 * np.matmul(c, M), self.M0 * np.sqrt(((np.matmul(S.T, M))**2).sum(axis=0))
        
        if self.get_components: #if you want the raw components c, M, S
            return self.coeffs, self.M0 *self.M, self.S
        elif self.get_sigma: #If you want the observable + sigma
            return self.M0 * np.matmul(self.coeffs, self.M), self.M0 * np.sqrt(((np.matmul(self.S.T, self.M))**2).sum(axis=0))
        else: #If you just want the observable
            return self.M0 * np.matmul(self.coeffs, self.M)
        
    def stellar_mass_formed(self, time, **kwargs):
        self.get_sigma = kwargs.get('get_sigma', False)
        self.get_components = kwargs.get('get_components', False)
        self.fit_components = kwargs.get('fit_components', None)
        t_hat = (1 - time/self.t0).clip(self.t_hat_end, self.t_hat_start)
        
        if self.fit_components is None:              
            M=[]
            N = len(self.coeffs)
            for n in range(1, N+1):
                M.append(self.t_hat_start**n - t_hat**n)
            self.M = u.Quantity(M)
        
        else:
            c, M, S = self.fit_components
            return self.M0 * np.matmul(c, M), self.M0 * np.sqrt(((np.matmul(S.T, M))**2).sum(axis=0))
        
        if self.get_components: #if you want the raw components c, M, S
            return self.coeffs, self.M0 *self.M, self.S
        elif self.get_sigma: #If you want the observable + sigma
            return self.M0 * np.matmul(self.coeffs, self.M), self.M0 * np.sqrt(((np.matmul(self.S.T, self.M))**2).sum(axis=0))
        else: #If you just want the observable
            return self.M0 * np.matmul(self.coeffs, self.M)

def compute_polynomial_models(input_type):
    
    # Observables
    t0 = 13.7 * u.Gyr  # time of observation 
    t_hat = np.logspace(-3, 0, 1001)[::-1] #t_hat from 1 --> 0
    t = t0*(1-t_hat) #time from 0Gyr --> 13.7Gyr
    lookback_time = (t0-t)[::-1] #t0 --> 0
    R_V = 3.1
    
    # SSPs
    ssp_pop = pst.SSP.PopStar(IMF="sal")
    ssp = ssp_pop
    
    #%%    
    #MODEL PARAMETERS
    allow_negative_sfr = False #Allows negative SFRs
    
    print('Initiating code')
    print()
    print('Code parameters:')
    print('Negative SFR: ', allow_negative_sfr)
    print()
    
    t_start_values = [0]*u.Gyr
    #t_end_values = [1, 2, 3, 4, 7, 10, 11, 12, 12.7, 13, 13.2, 13.5, 13.6, 13.62, 13.65, 13.67, 13.7]*u.Gyr
    t_end_values =  [13.7]*u.Gyr

    t_grid = [(b, c) for b in t_start_values for c in t_end_values
                  if b.to_value()<c.to_value()]
    
    # Primordial polynomia
    N_min = 1
    N_max = 3
    N_range = np.arange(N_min, N_max+1)  # polynomial degree
    
    ##############################################################
    
    #MODELS GRID
    model_A_V_grid = [0, 0.5, 1.0, 1.2]
    model_z_grid = np.array([0.004, 0.008, 0.02, 0.05])
     
    ## MFH TARGET
    filters_names = ['GALEX_FUV', 'GALEX_NUV', 'u', 'g', 'r', 'i', 'z']
    filters_wl = np.array([1538.6, 2315.7, 3543., 4770., 6231., 7625., 9134.])
    
    
    if input_type == 'Illustris':
        obs_filters = filters_names[2:]
        obs_filters_wl = filters_wl[2:]
        ssp.cut_wavelength(1000, 1600000)
        
    elif input_type == 'salim':
        obs_filters = filters_names
        obs_filters_wl = filters_wl
        ssp.cut_wavelength(1000, 1600000)
        
    #Computing real models
    max_per_folder = 100 #Max amount per folder
    current_run = 0 #Current run (0 --> n_runs-1)   
    
    #NEED read the lines [target_ID_name [Fnu] [Fnu_error]]
    input_photo = {}
    input_file = os.path.join(os.getcwd(),'input', '{}_input.txt'.format(input_type))
    
    content = open(input_file, "r").readlines()
    
    for line in content:
        data = line.split(' ')
        target_ID = data[0]
        input_photo[target_ID] = {}
        input_photo[target_ID]['Fnu_obs'] = [float(string) for string in data[1:len(obs_filters)+1]]
        input_photo[target_ID]['error_Fnu_obs'] = [float(string) for string in data[-len(obs_filters):]]
    
    
    def get_flux_densities(model, ssp, obs_filters, Z_i, t, **kwargs):
        fnu = []
    #    fnu_error = []
        cum_mass = np.cumsum(model.integral_SFR(t))
        z_array = Z_i*np.ones(len(t))
        sed, weights = ssp.compute_SED(t, cum_mass, z_array)
        
    
        for i, filter_name in enumerate(obs_filters):
            photo = pst.observables.Filter( wavelength = ssp.wavelength, filter_name = filter_name)
            fnu_Jy, fnu_Jy_err = photo.get_fnu(sed, spectra_err=None)
            fnu.append( fnu_Jy )
    #        fnu_error.append( fnu_Jy_error )
            fnu_Jy
        return u.Quantity(fnu)
    
    #%% 
    ##################################################################################
    ##################################################################################
    #Main code
    output_path = os.path.join(os.getcwd(), 'output', 'results_{}'.format(input_type)) #Change folder name as desired
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    #Saving t in Gyrs
    with open(os.path.join(output_path, 't_Gyr.txt'), 'w+') as f_MFH:
        if os.stat(os.path.join(output_path, 't_Gyr.txt')).st_size == 0:
            f_MFH.write(" ".join(str(x) for x in t.to_value()) + "\n")
         
    for number_model, target_ID in enumerate(input_photo):
        print('Computing Model #{} of {}'.format(number_model+1, len(input_photo)))
        
        #Dividing the output into files    
        if number_model % max_per_folder == 0:
            if number_model==0:
                min_ID= 0
            else:
                min_ID+=1
        
        run_folder = 'models_{}_{}'.format(current_run*len(input_photo)+min_ID*max_per_folder, 
                             current_run*len(input_photo)+min((min_ID+1)*max_per_folder-1, len(input_photo)-1))
                            
        run_path = os.path.join(output_path, run_folder)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        
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
        for Z_i_index, Z_i in enumerate(model_z_grid):   
            
            for av_index, A_V_model in enumerate(model_A_V_grid):
                
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
           
        def old_std(x, w):
            mu= weighted_mean(x, w)
            return np.sqrt(np.dot(w, np.power(x, 2))-np.power(mu, 2))
        
        def yago_std(x, sigma, w):
            mu= weighted_mean(x, w)
            return np.sqrt((np.dot(w, np.power((x-mu), 2))+ np.power(sigma, 2)).sum(axis=0))
    #        return np.sqrt((np.array(w)*(np.array(x)-mu)**2+ np.array(sigma)**2).sum(axis=0))
        
        #t(M) y M(t)           
        norm_chi2_poly = chi2_poly - min(chi2_poly)
        likelihood_poly = np.array(np.exp(-norm_chi2_poly/2))
        likelihood_poly[np.isnan(likelihood_poly)] = 0
        norm_weights = likelihood_poly/np.sum(likelihood_poly)
        
        #MODELO SIN ERRORES             
        age_of_fraction_i = []
        fraction_i = []
        new_sigma2 = []
        sigma_i = []
        total_mass_i = []
        for i, model in enumerate(model_poly): #For each model of this MC loop:
            mass, sigma = model.mass_formed_since(t0-lookback_time, get_sigma=True) #mass formed and its sigma
            c, M, S = model.mass_formed_since(t0-lookback_time, get_components=True)        
            total_mass = mass[-1].to_value()
            total_mass_i.append(total_mass)        
            fraction_model = np.array(mass/total_mass)
            sigma_model = np.array(sigma/total_mass)
            new_sigma2.append((norm_weights[i]*(np.matmul(S.T,  M))**2).sum(axis=0)/total_mass**2) 
            fraction = np.logspace(-3, 0, len(t))
            age_of_fraction_model = np.interp(fraction, fraction_model, lookback_time)
            
            fraction_i.append(fraction_model)
            sigma_i.append(sigma_model)
            age_of_fraction_i.append( age_of_fraction_model )
        
                     
        mass_fraction = weighted_mean(fraction_i, norm_weights)
        paper_fraction_std = np.sqrt(np.array(new_sigma2).sum(axis=0))  
        weighted_total_mass = weighted_mean(total_mass_i, norm_weights)
    #    old_fraction_std = old_std(fraction_i, norm_weights) 
    #    yago_fraction_std = yago_std(fraction_i, sigma_i, norm_weights)
        age_of_fraction = weighted_mean(age_of_fraction_i, norm_weights) 
        age_of_fraction_std = old_std(age_of_fraction_i, norm_weights) 
                                     
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
        file_ID = target_ID[8:]
        with open(os.path.join(run_path, 'met_dust_model_{}.txt'.format(file_ID)), 'w+') as f:
            f.write('model_Z std_Z model_dust_extinction model_dust_extinction_std'+'\n')
            f.write('{} {} {} {} {}'.format(file_ID, 
                           np.round(weighted_z[0], 5), np.round(std_weighted_z[0], 5), 
                           str(np.round(weighted_d, 3))[1:], str(np.round(std_weighted_d, 3))[1:]+'\n'))
        with open(os.path.join(run_path, 'mass_fraction_model_{}.txt'.format(file_ID)), 'w+') as f:
            f.write('M(t) delta_M(t) M0'+'\n')
            f.write(" ".join(str(x) for x in mass_fraction) + "\n")
            f.write(" ".join(str(x) for x in paper_fraction_std) + "\n")
            f.write("{}".format(np.round(weighted_total_mass, 5))+'\n')
            
        with open(os.path.join(run_path, 'age_of_fraction_model_{}.txt'.format(file_ID)), 'w+') as f:
            f.write('t(M) delta_t(M)'+'\n')
            f.write(" ".join(str(x) for x in age_of_fraction) + "\n")
            f.write(" ".join(str(x) for x in age_of_fraction_std) + "\n")
        
        keys_array = []
        for key, value in input_photo[target_ID].items():
            keys_array.append(key)
            
        with open(os.path.join(run_path, 'real_parameters_model_{}.txt'.format(file_ID)), 'w+') as f:
            f.write(" ".join([key for key in input_photo[target_ID].keys()]) + "\n")
            for key, value in input_photo[target_ID].items():
                f.write("{}".format(value) + "\n")
        
     #%%
        print("--- time computing model = %s seconds ---" % (round(pst_time, 3)))

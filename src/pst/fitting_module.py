"""

"""
import numpy as np
from astropy import units as u
import pst
import time as tic
import extinction
from pst.models import ChemicalEvolutionModel

#-------------------------------------------------------------------------------

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

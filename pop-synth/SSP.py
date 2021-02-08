import os
import numpy as np
import units

#-------------------------------------------------------------------------------
class PopStar:
#-------------------------------------------------------------------------------
  path = os.path.join( os.path.dirname(__file__), 'data/PopStar' )
  metallicities = np.array([ 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 ])
  # Z_sun=0.0134
  log_ages_yr = np.array([ 5.00, 5.48, 5.70, 5.85, 6.00, 6.10, 6.18, 6.24, 6.30, 6.35, 6.40, 6.44, 6.48, 6.51, 6.54, 6.57, 6.60, 6.63, 6.65, 6.68, 6.70, 6.72, 6.74, 6.76, 6.78, 6.81, 6.85, 6.86, 6.88, 6.89, 6.90, 6.92, 6.93, 6.94, 6.95, 6.97, 6.98, 6.99, 7.00, 7.04, 7.08, 7.11, 7.15, 7.18, 7.20, 7.23, 7.26, 7.28, 7.30, 7.34, 7.38, 7.41, 7.45, 7.48, 7.51, 7.53, 7.56, 7.58, 7.60, 7.62, 7.64, 7.66, 7.68, 7.70, 7.74, 7.78, 7.81, 7.85, 7.87, 7.90, 7.93, 7.95, 7.98, 8.00, 8.30, 8.48, 8.60, 8.70, 8.78, 8.85, 8.90, 8.95, 9.00, 9.18, 9.30, 9.40, 9.48, 9.54, 9.60, 9.65, 9.70, 9.74, 9.78, 9.81, 9.85, 9.90, 9.95, 10.00, 10.04, 10.08, 10.11, 10.12, 10.13, 10.14, 10.15, 10.18])
  # isochrone age in delta [log(tau)]=0.01
  wavelength = np.loadtxt( os.path.join(path,'SED','spneb_kro_0.15_100_z0500_t9.95'),
			  dtype=np.float, skiprows=0, usecols=(0,), unpack=True
			  ) *units.Angstrom
                          
  SED_data = {}       
  
  def __init__(self, IMF):
    if IMF not in self.SED_data:
      print("> Initialising Popstar models (IMF='"+IMF+"')")
      self.SED_data[IMF] = np.empty( shape=(self.metallicities.size,self.log_ages_yr.size), dtype=np.ndarray )
    
    for i, Z in enumerate(self.metallicities):
	       for j, age in enumerate(self.log_ages_yr):
                 file = os.path.join( PopStar.path,'SED',
	               'spneb_{0}_z{1:04.0f}_t{2:.2f}'.format(IMF, Z*1e4, age) )
                 self.SED_data[IMF][i][j] = np.loadtxt(file, dtype=np.float, skiprows=0, usecols=(1,2,3), unpack=True) \
		            * (3.82e33*units.erg/units.second/units.Angstrom) / units.Msun
				  
                 self.SED = self.SED_data[IMF]

  def get_SED(self, Z, age):
    Z = np.interp( Z, self.metallicities, self.metallicities ) # to prevent extrapolation
    log_age = np.interp( np.log10(age/units.yr), self.log_ages_yr, self.log_ages_yr ) # to prevent extrapolation
    index_Z_hi = self.metallicities.searchsorted( Z ).clip( 1, len(self.metallicities)-1 )
    index_t_hi = self.log_ages_yr.searchsorted( np.log10(age/units.yr) ).clip( 1, len(self.log_ages_yr)-1 )
    weight_Z_hi = np.log( Z/self.metallicities[index_Z_hi-1] ) / np.log( self.metallicities[index_Z_hi]/self.metallicities[index_Z_hi-1] ) # log interpolation in Z
    weight_t_hi = ( log_age-self.log_ages_yr[index_t_hi-1] ) / ( self.log_ages_yr[index_t_hi]-self.log_ages_yr[index_t_hi-1] ) # log interpolation in t
    #print 'Z: ',Z, index_Z_hi, weight_Z_hi, 'log(age/yr):', log_age, index_t_hi, weight_t_hi
    return    weight_Z_hi *   weight_t_hi * self.SED[index_Z_hi  ][index_t_hi  ] \
         +    weight_Z_hi *(1-weight_t_hi)* self.SED[index_Z_hi  ][index_t_hi-1] \
         + (1-weight_Z_hi)*   weight_t_hi * self.SED[index_Z_hi-1][index_t_hi  ] \
         + (1-weight_Z_hi)*(1-weight_t_hi)* self.SED[index_Z_hi-1][index_t_hi-1]
  
#-------------------------------------------------------------------------------

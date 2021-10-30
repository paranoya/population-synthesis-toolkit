import os
import numpy as np
#import pylab as plt
import units

#-------------------------------------------------------------------------------
class Chemical_evolution_model:        
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_gas = kwargs['M_gas']*units.Msun
    self.Z = kwargs['Z']
    
  def get_Z(self, time):
    return self.Z
  
  def set_current_state(self, galaxy):
    galaxy.M_gas = self.M_gas
    galaxy.Z = self.Z
    galaxy.M_stars = self.integral_SFR( galaxy.today ) #TODO: account for stellar death
    
#-------------------------------------------------------------------------------
class Single_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------
 
  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']*units.Msun          
    self.t = kwargs['t']*units.Gyr                      # Born time 
    Chemical_evolution_model.__init__(self, **kwargs)
  
  def integral_SFR(self, time):
    M_t = np.array(time)
    for i, t in np.ndenumerate(time):
      if t<=self.t:
 	      M_t[i] = 0
      else:
	      M_t[i] = self.M_stars
    return M_t

  def integral_Z_SFR(self, time):
    Z_t = np.array(time)
    for i, t in np.ndenumerate(time):
      if t<=self.t:
	      Z_t[i] = 0
      else:
	      Z_t[i] = self.Z * self.M_stars
    return Z_t
  
#-------------------------------------------------------------------------------
class Exponential_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)
      
  def integral_SFR(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau) )

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Exponential_SFR_delayed(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr       
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)
      
  def integral_SFR(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau)*(self.tau+time)/self.tau)

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Exponential_quenched_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr
    self.Z = kwargs['Z']
    self.t_q = kwargs['t_quench']*units.Gyr
    Chemical_evolution_model.__init__(self, **kwargs)
    

  def integral_SFR(self, time):
     if type(time) is float:
        if time<self.t_q:
              M_stars=self.M_inf * ( 1 - np.exp(-time/self.tau) )
                #M _inf is the mass for large t ;  M(t)=M_inf[1-exp(-t/tau)
        else:
            M_stars=self.M_inf * ( 1 - np.exp(-self.t_q/self.tau) )
         
     else:    
         M_stars=[]
         for t in time:
             if t<self.t_q:
                 M=self.M_inf * ( 1 - np.exp(-t/self.tau) )
                #M _inf is the mass for large t ;  M(t)=M_inf[1-exp(-t/tau)
             else:
                 M=self.M_inf * ( 1 - np.exp(-self.t_q/self.tau) )
             M_stars.append(M)
         M_stars=np.array(M_stars)     
        
     return M_stars

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
#class MD05(Chemical_evolution_model):
##-------------------------------------------------------------------------------
#  path = os.path.join( os.path.dirname(__file__), 'data/MD05' )
#  
#  def __init__(self, **kwargs):
#    filename = kwargs['V_rot']+kwargs['eff']
#    self.R = kwargs['R']
#    while self.R > 0 and not os.path.exists( os.path.join(MD05.path,'{}radius_{:02.0f}'.format(filename,self.R)) ):
#      self.R -=1
#    filename = os.path.join(MD05.path,'{}radius_{:02.0f}'.format(filename,self.R))
#    print("> Reading MD05 file: '"+ filename +"'" , t, dm, Z = np.loadtxt( filename, dtype=np.float, unpack=True)
#    self.t_table =np.append([0],(t+0.5)*units.Gyr )
#    self.Z_table = np.append( [0], Z )
#    dm = np.append( [0], dm*1e9*units.Msun )
#    self.integral_SFR_table = np.cumsum(dm)
#    self.integral_Z_SFR_table = np.cumsum( self.Z_table*dm )
#    
#  def set_current_state(self, galaxy):
#    if self.R > 0.5:
#      area = 2*np.pi*self.R*units.kpc**2
#    else:
#      area = np.pi*((self.R+0.5)*units.kpc)**2
#    fraction_of_area_included = min(galaxy.Area_included/area, 1.)
#    
#    galaxy.M_stars = fraction_of_area_included * self.integral_SFR( galaxy.today ) #TODO: account for stellar death
#    Z = np.interp( galaxy.today, self.t_table, self.Z_table )
#    galaxy.M_gas = galaxy.M_stars*max(.03/Z-1,1e-6) # TODO: read from files
#    galaxy.Z = Z
#    
#  def integral_SFR(self, time):
#    return np.interp( time, self.t_table, self.integral_SFR_table )
#
#  def integral_Z_SFR(self, time):
#    return np.interp( time, self.t_table, self.integral_Z_SFR_table )

#-------------------------------------------------------------------------------
class ASCII_file(Chemical_evolution_model):
#-------------------------------------------------------------------------------
  
  def __init__(self, file,
	       time_column = 0,
	       Z_column    = 1,
	       SFR_column  = 2,
	       time_units  = units.Gyr,
	       SFR_units   = units.Msun/units.yr ):
    print("> Reading SFR file: '"+ file +"'")
    t, Z, SFR = np.loadtxt( file, dtype=np.float, usecols=(time_column,Z_column,SFR_column), unpack=True)
    self.t_table   = np.append( [0], t*time_units )
    self.Z_table   = np.append( [0], Z )

    dt = np.ediff1d( self.t_table, to_begin=0 )
    dm = np.append( [0], SFR*SFR_units )*dt
    self.integral_SFR_table = np.cumsum( dm )
    self.integral_Z_SFR_table = np.cumsum( self.Z_table*dm )
    
  def set_current_state(self, galaxy):
    galaxy.M_stars = self.integral_SFR( galaxy.today ) #TODO: account for stellar death
    Z = np.interp( galaxy.today, self.t_table, self.Z_table )
    galaxy.M_gas = galaxy.M_stars*max(.03/(Z+1e-6)-1,1e-6) # TODO: read from files
    galaxy.Z = Z
    
  def integral_SFR(self, time):
    return np.interp( time, self.t_table, self.integral_SFR_table )

  def integral_Z_SFR(self, time):
    return np.interp( time, self.t_table, self.integral_Z_SFR_table )

  #def plot(self):
    #plt.semilogy( self.t_table/units.Gyr, self.SFR_table/(units.Msun/units.yr) )
    #plt.semilogy( self.t_table/units.Gyr, self.integral_SFR_table/units.Msun )
    #plt.semilogy( self.t_table/units.Gyr, self.Z_table )
    #plt.show()

#-------------------------------------------------------------------------------

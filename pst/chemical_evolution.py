import numpy as np
#import pylab as plt
from astropy import units as u
from math import erf
from scipy import special


#-------------------------------------------------------------------------------
class Chemical_evolution_model:
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.M_gas = kwargs.get('M_gas', 0*u.Msun)
        self.Z = kwargs.get('Z', 0.02)

    def get_Z(self, time):
        return self.Z

    def integral_Z_SFR(self, time):
      return self.Z * self.integral_SFR(time)

    # def set_current_state(self, galaxy):
    #     galaxy.M_gas = self.M_gas*units.Msun
    #     galaxy.Z = self.Z
    #     galaxy.M_stars = self.integral_SFR( galaxy.today/units.Gyr ) #TODO: account for stellar death

    def compute_SED(self, SSP, t_obs, allow_negative=True ):

        age_bins = np.hstack(
            [0*u.yr, np.sqrt(SSP.ages[1:]*SSP.ages[:-1]), 1e12*u.yr])
        t_bins = t_obs - age_bins
        t_bins = t_bins[t_bins > 0]

        M_t = self.integral_SFR(t_bins)
        M_bin = np.hstack([M_t[:-1]-M_t[1:], M_t[-1]])

        MZ_t = self.integral_Z_SFR(t_bins)
        MZ_bin = np.hstack([MZ_t[:-1]-MZ_t[1:], MZ_t[-1]])
        iZ_max = len(SSP.metallicities)-1

        SED = np.zeros(SSP.wavelength.size)*u.Lsun/u.Angstrom

        # Sum over the SSP ages
        for i, m in enumerate(M_bin):
            if m > 0 or (allow_negative and m<0):
                Z = np.clip(MZ_bin[i] / m,
                            SSP.metallicities[0], SSP.metallicities[-1])
                index_Z_hi = SSP.metallicities.searchsorted(Z).clip(1, iZ_max)
                # log interpolation in Z
                weight_Z_hi = np.log(
                    Z/SSP.metallicities[index_Z_hi-1]
                    ) / np.log(SSP.metallicities[index_Z_hi]
                               / SSP.metallicities[index_Z_hi-1])
                SED += m * (SSP.SED[index_Z_hi][i] * weight_Z_hi
                            + SSP.SED[index_Z_hi-1][i] * (1-weight_Z_hi))
        return SED

#-------------------------------------------------------------------------------
class Single_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']
    self.t = kwargs['t_burst']
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    M_t = []
    if type(time)==float:
        time=[time]
    for  t in time:
      if t<=self.tb:
           M_t.append(0)
      else:
          M_t.append( self.M_stars)
    return M_t

  def integral_Z_SFR(self, time):
    Z_t = []
    time=time*units.Gyr
    for t in time:
      if t<=self.tb:
          Z_t.append(0)
      else:
          Z_t.append( self.Z * self.M_stars)
    return Z_t

#-------------------------------------------------------------------------------
class Gaussian_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_stars']*units.Msun
    self.tb = kwargs['t']*units.Gyr             # Born time
    self.c=kwargs['c']*units.Myr # En Myr
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    time=time*units.Gyr
    return self.M_inf/2*( -special.erf((-self.tb)/(np.sqrt(2)*self.c)) +  special.erf((time-self.tb)/(np.sqrt(2)*self.c)) )

  def give_SFR(self, time):
    time=time*units.Gyr
    a = self.M_inf/(2*self.c*np.sqrt(np.pi/2))
    return a * np.exp(-(time-self.tb)**2/(2*self.c**2))

  def give_der_SFR(self,time):
    time=time*units.Gyr
    a = self.M_inf/(self.c*np.sqrt(np.pi/2))
    return -a/self.c**2 * (time-self.tb) * np.exp(-(time-self.tb)**2/(2*self.c**2))


  def give_2der_SFR(self,time):
    time=time*units.Gyr
    a = self.M_inf/(self.c*np.sqrt(np.pi/2))
    return a/self.c**4 * (time -self.c -self.tb)*(time +self.c -self.tb) * np.exp(-(time-self.tb)**2/(2*self.c**2))


  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Exponential_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.M_inf = kwargs['M_inf']*units.Msun
        self.tau = kwargs['tau']*units.Gyr
        self.Z = kwargs['Z']
        Chemical_evolution_model.__init__(self, **kwargs)

    def integral_SFR(self, time):
      time=time*units.Gyr
      return self.M_inf * ( 1 - np.exp(-time/self.tau) )

    def give_SFR(self, time):
      time=time*units.Gyr
      return self.M_inf*(np.exp(-time/self.tau))/self.tau

    def give_der_SFR(self,time):
      time=time*units.Gyr
      return -self.M_inf*(np.exp(-time/self.tau))/(self.tau**2)


    def give_2der_SFR(self,time):
      time=time*units.Gyr
      return self.M_inf*(np.exp(-time/self.tau))/(self.tau**3)


    def integral_Z_SFR(self, time):
      return self.Z * self.integral_SFR(time)


#-------------------------------------------------------------------------------
class Polynomial_MFH(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.t_start = kwargs.get('t_start', 0*u.Gyr)
        self.t_end = kwargs['t_end']
        self.M_end = kwargs.get('M_end', 1*u.Msun)
        self.coeffs = kwargs['coeffs']
        Chemical_evolution_model.__init__(self, **kwargs)

    def integral_SFR(self, time):
        t_hat = ((self.t_end-time) / (self.t_end-self.t_start)).clip(0, 1)
        M_hat = 0
        for n, c in enumerate(self.coeffs):
            M_hat += c*t_hat**n
        return M_hat*self.M_end

#-------------------------------------------------------------------------------
class Polynomial(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.M_0 = kwargs['M_0']*units.Msun
        self.t_0 = kwargs['t_0']*units.Gyr
        self.Z = kwargs['Z']
        self.index=kwargs['index']
        self.n_base=kwargs['n_base']
        self.lbt_min = kwargs['lbt_min']*units.Gyr
        self.lbt_max = kwargs['lbt_max']*units.Gyr
        Chemical_evolution_model.__init__(self, **kwargs)

    def integral_SFR(self,time):
        time=time*units.Gyr
        tp=(self.t_0-self.lbt_min-time)/(self.lbt_max-self.lbt_min)
        M= self.M_0*(tp**self.index-tp**self.n_base)
        return M

    def integral_Z_SFR(self, time):
        return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Parametric_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.t_0 = kwargs['t_0']*units.Gyr
    self.Z = kwargs['Z']
    self.a=kwargs['a']
    self.b=kwargs['b']
    self.e=kwargs['e']
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    self.time=time*units.Gyr
    #SFR = (M0/t0)*( b*(1+a*tp*(1+e*tp/2)) - 4*K*tp**3 )
    tp=1-self.time/self.t_0
    K = self.b*(1+self.a/2*(1+self.e/3)) - 1
    return self.M_inf*( 1-self.b*tp*(1+self.a*(tp/2)*(1+self.e*(tp/3))) + K*tp**4 )

  def give_SFR(self, t):
     tp=1-t/self.t_0
     K = self.b*(1+self.a/2*(1+self.e/3)) - 1
     return (self.M_inf/self.t0)*( self.b*(1+self.a*tp*(1+self.e*tp/2)) - 4*K*tp**3 )

  def integral_Z_SFR(self, t):
      tp=1-t/self.t_0
      return self.Z * self.integral_SFR(tp)


#-------------------------------------------------------------------------------
class Tabular_MFH(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, times, masses, **kwargs):
        self.table_t = times
        self.table_M = masses
        Chemical_evolution_model.__init__(self, **kwargs)

    def integral_SFR(self, times):
        return np.interp(times, self.table_t, self.table_M)

    def integral_Z_SFR(self, times):
        return self.Z * self.integral_SFR(times)


#-------------------------------------------------------------------------------
class Exponential_SFR_delayed(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*units.Msun
    self.tau = kwargs['tau']*units.Gyr
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    self.time=time*units.Gyr
    return self.M_inf * ( 1 - np.exp(-self.time/self.tau)*(self.tau+self.time)/self.tau)

  def give_SFR(self,time):
      self.time=time*units.Gyr
      return self.M_inf*(self.time/self.tau**2)*np.exp(-self.time/self.tau)

  def give_der_SFR(self,time):
      self.time=time*units.Gyr
      return -self.M_inf*((self.time-self.tau)/self.tau**3)*np.exp(-self.time/self.tau)

  def give_2der_SFR(self,time):
      self.time=time*units.Gyr
      return self.M_inf*((self.time-2*self.tau)/self.tau**4)*np.exp(-self.time/self.tau)

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
class ASCII_file(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, file,
           time_column = 0,
           Z_column    = 1,
           SFR_column  = 2,
           time_units  = u.Gyr,
           SFR_units   = u.Msun/u.yr ):
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


# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------

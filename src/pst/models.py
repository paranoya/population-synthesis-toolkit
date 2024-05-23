import numpy as np
#import pylab as plt
from astropy import units as u
from astropy.io import fits
from scipy import special
import pst

from scipy import interpolate

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

    def compute_SED(self, SSP : pst.SSP.SSPBase, t_obs : u.Quantity, allow_negative=True ):
        """Compute the SED of a given model observed at a given time.
        
        Description
        -----------
        This method computes the spectra energy distribution resulting from the
        chemical evolution model observed at a given time.

        Parameters
        ----------
        - SSP: pst.SSP.SSP
            The SSP model to used for synthezising the SED.
        - t_obs: astropy.Quantity

        """
        age_bins = np.hstack(
            [0 * u.yr, np.sqrt(SSP.ages[1:] * SSP.ages[:-1]), 1e12 * u.yr])
        t_bins = t_obs - age_bins
        t_bins = t_bins[t_bins > 0]
        M_t = self.integral_SFR(t_bins)
        M_bin = np.hstack([M_t[:-1]-M_t[1:], M_t[-1]])
        MZ_t = self.integral_Z_SFR(t_bins)
        MZ_bin = np.hstack([MZ_t[:-1]-MZ_t[1:], MZ_t[-1]])
        iZ_max = len(SSP.metallicities)-1

        SED = np.zeros(SSP.wavelength.size) * u.Lsun / u.Angstrom

        # Sum over the SSP ages
        for i, m in enumerate(M_bin):
            if m > 0 or (allow_negative and m<0):
                Z = np.clip(MZ_bin[i] / m,
                            SSP.metallicities[0], SSP.metallicities[-1])
                index_Z_hi = SSP.metallicities.searchsorted(Z).clip(1, iZ_max)
                # log interpolation in Z
                weight_Z_hi = u.dimensionless_unscaled * np.log(
                    Z / SSP.metallicities[index_Z_hi-1]
                    ) / np.log(SSP.metallicities[index_Z_hi]
                               / SSP.metallicities[index_Z_hi-1]
                               ) 
                SED += m * (SSP.L_lambda[index_Z_hi][i] * weight_Z_hi
                            + SSP.L_lambda[index_Z_hi-1][i] * (1-weight_Z_hi))
        return SED


#-------------------------------------------------------------------------------
class Single_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']
    self.t = kwargs['t_burst']
    Chemical_evolution_model.__init__(self, **kwargs)

# TODO: do this using np.select(); actually, does tb exist at all?
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



#-------------------------------------------------------------------------------
class Exponential_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.M_inf = kwargs['M_inf']
        if not isinstance(self.M_inf, u.Quantity):
           print("Assuming that input M_inf is in Msun")
           self.M_inf *= u.Msun
        self.tau = kwargs['tau']
        if not isinstance(self.tau, u.Quantity):
            print("Assuming that input tau is in Gyr")
            self.tau *= u.Gyr

        self.Z = kwargs['Z']
        Chemical_evolution_model.__init__(self, **kwargs)

    def integral_SFR(self, time):
      return self.M_inf * ( 1 - np.exp(-time/self.tau) )

    def SFR(self, time):
      return self.M_inf*(np.exp(-time/self.tau))/self.tau

    def dot_SFR(self,time):
      return -self.M_inf*(np.exp(-time/self.tau))/(self.tau**2)


    def ddot_SFR(self,time):
      return self.M_inf*(np.exp(-time/self.tau))/(self.tau**3)


    def integral_Z_SFR(self, time):
      return self.Z * self.integral_SFR(time)


#-------------------------------------------------------------------------------
class Exponential_SFR_delayed(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau)*(self.tau+time)/self.tau)

  def SFR(self,time):
      return self.M_inf*(time/self.tau**2)*np.exp(-time/self.tau)

  def dot_SFR(self,time):
      return -self.M_inf*((time-self.tau)/self.tau**3)*np.exp(-time/self.tau)

  def ddot_SFR(self,time):
      return self.M_inf*((time-2*self.tau)/self.tau**4)*np.exp(-time/self.tau)

  def integral_Z_SFR(self, time):
    return self.Z * self.integral_SFR(time)

#-------------------------------------------------------------------------------
class Polynomial_MFH_fit: #Generates the basis for the Polynomial MFH
#-------------------------------------------------------------------------------
    def __init__(self, N, SSP, obs_filters, t_obs, Z_i, dust_extinction, 
                 error_L_obs, **kwargs):
        self.t_obs = t_obs.to_value()
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        
        primordial_coeffs = []
        primordial_L = []
        for n in range(N):
            
            c = np.zeros(N)
            c[n] = 1
            
            primordial_coeffs.append(c)
            
            L = []
            p = pst.models.Polynomial_MFH(Z=Z_i, t_hat_start = self.t_hat_start,
                                          t_hat_end = self.t_hat_end,
                                          coeffs=c)
            sed = p.compute_SED(SSP, t_obs)
            #print(primordial_coeffs)
            for filter_name in obs_filters:
                photo = pst.observables.luminosity(
                    flux=sed, wavelength=SSP.wavelength, filter_name=filter_name)
                L.append(photo.integral_flux.to_value(u.Lsun))  
                # linalg complains about units
            primordial_L.append(np.array(L))
        primordial_L = np.array(primordial_L)*dust_extinction / error_L_obs
        
        self.p = p
        self.sed = sed
        self.lstsq_solution = np.matmul(
            np.linalg.pinv(np.matmul(primordial_L, np.transpose(primordial_L))),
            primordial_L)
        self.primordial_coeffs = np.array(primordial_coeffs)
        self.primordial_L_Lsun = np.array(primordial_L)
        self.primordial_L = primordial_L

    def fit(self, L_obs_Lsun, **kwargs):

        c = np.matmul(self.lstsq_solution,
                      L_obs_Lsun)              
        return c
    
#-------------------------------------------------------------------------------
class Polynomial_MFH(Chemical_evolution_model):
#-------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self.t0 = kwargs.get('t0', 13.7*u.Gyr)
        self.M0 = kwargs.get('M_end', 1*u.Msun)
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        self.coeffs = kwargs['coeffs']
        self.S = kwargs.get('S', False)
        
        Chemical_evolution_model.__init__(self, **kwargs)
    
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
                M.append(t_hat_present_time**n - t_hat_since**n)
            self.M = M
        else:
            c, M, S = self.fit_components
            return self.M0 * np.matmul(c, M), self.M0 * np.sqrt(((np.matmul(S.T, M))**2).sum(axis=0))
        
        if self.get_components: #if you want the raw components c, M, S
            return self.coeffs, self.M0 *self.M, self.S
        elif self.get_sigma: #If you want the observable + sigma
            return self.M0 * np.matmul(self.coeffs, self.M), self.M0 * np.sqrt(((np.matmul(self.S.T, self.M))**2).sum(axis=0))
        else: #If you just want the observable
            return self.M0 * np.matmul(self.coeffs, self.M)
        
    def integral_SFR(self, time, **kwargs):
        self.get_sigma = kwargs.get('get_sigma', False)
        self.get_components = kwargs.get('get_components', False)
        self.fit_components = kwargs.get('fit_components', None)
        t_hat = (1 - time/self.t0).clip(self.t_hat_end, self.t_hat_start)
        
        if self.fit_components is None:              
            M=[]
            N = len(self.coeffs)
            for n in range(1, N+1):
                M.append(t_hat**n-self.t_hat_start**n)
            self.M = M
        
        else:
            c, M, S = self.fit_components
            return self.M0 * np.matmul(c, M), self.M0 * np.sqrt(((np.matmul(S.T, M))**2).sum(axis=0))
        
        if self.get_components: #if you want the raw components c, M, S
            return self.coeffs, self.M0 *self.M, self.S
        elif self.get_sigma: #If you want the observable + sigma
            return self.M0 * np.matmul(self.coeffs, self.M), self.M0 * np.sqrt(((np.matmul(self.S.T, self.M))**2).sum(axis=0))
        else: #If you just want the observable
            return self.M0 * np.matmul(self.coeffs, self.M)
        
       
    def SFR(self, time, **kwargs):       
        t_hat = (1 - time/self.t0)
        self.get_sigma = kwargs.get('get_sigma', False)
        self.get_components = kwargs.get('get_components', False)
        self.fit_components = kwargs.get('fit_components', None)
        
        if self.fit_components is None:   
            M=[]
            N = len(self.coeffs)
            for n in range(1, N+1):      
                m=-(n*t_hat**(n-1))/self.t0
                m[t_hat > self.t_hat_start] = 0.
                m[t_hat < self.t_hat_end] = 0.
                M.append(m)
            self.M = M
        else:
            c, M, S = self.fit_components
            return self.M0 * np.matmul(c, M), self.M0 * np.sqrt(((np.matmul(S.T, M))**2).sum(axis=0))
        
        if self.get_components: #if you want the raw components c, M, S
            return self.coeffs, self.M0 *self.M, self.S
        elif self.get_sigma: #If you want the observable + sigma
            return self.M0 * np.matmul(self.coeffs, self.M), self.M0 * np.sqrt(((np.matmul(self.S.T, self.M))**2).sum(axis=0))
        else: #If you just want the observable
            return self.M0 * np.matmul(self.coeffs, self.M)

    def dot_SFR(self, time):
        t_hat = (1 - time/self.t0)
        
        M=[]
        N = len(self.coeffs)
        
        for n in range(0, N):
            m =(n*(n-1)*t_hat**(n-2))/self.t0**2
            m[t_hat > self.t_hat_start] = 0.
            m[t_hat < self.t_hat_end] = 0.
            M.append(m)
      
        if self.compute_sigma:
            
            return self.M0*self.coeffs, M, self.S
        else:
            return self.M0 * np.matmul(self.coeffs, M)
    
    def ddot_SFR(self, time):
        t_hat = (1 - time/self.t0)
        
        M=[]
        N = len(self.coeffs)
        for n in range(0, N):
            m=-(n*(n-1)*(n-2)*t_hat**(n-3))/self.t0**3
            m[t_hat > self.t_hat_start] = 0.
            m[t_hat < self.t_hat_end] = 0.
            M.append(m)

        if self.compute_sigma:
            return self.M0*self.coeffs, M, self.S
        else:
            return self.M0 * np.matmul(self.coeffs, M)


#-------------------------------------------------------------------------------
class Gaussian_burst(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_stars']*u.Msun
    self.tb = kwargs['t']*u.Gyr             # Born time
    self.c = kwargs['c']*u.Gyr # En Myr
    Chemical_evolution_model.__init__(self, **kwargs)

  def integral_SFR(self, time):
    return self.M_inf/2*( -special.erf((-self.tb)/(np.sqrt(2)*self.c)) +  special.erf((time-self.tb)/(np.sqrt(2)*self.c)) )

  def SFR(self, time):
    a = self.M_inf/(2*self.c*np.sqrt(np.pi/2))
    return a * np.exp(-(time-self.tb)**2/(2*self.c**2))

  def dot_SFR(self,time):
    a = self.M_inf/(self.c*np.sqrt(np.pi/2))
    return -a/self.c**2 * (time-self.tb) * np.exp(-(time-self.tb)**2/(2*self.c**2))

  def ddot_SFR(self,time):
    a = self.M_inf/(self.c*np.sqrt(np.pi/2))
    return a/self.c**4 * (time -self.c -self.tb)*(time +self.c -self.tb) * np.exp(-(time-self.tb)**2/(2*self.c**2))

#-------------------------------------------------------------------------------
class Tabular_MFH(Chemical_evolution_model):
    """Chemical evolution model based on an arbitrary grid of times and metallicities."""
    def __init__(self, times, masses, **kwargs):
        super().__init__(**kwargs)
        self.table_t = times
        # Make sure that time is cresscent
        sort_times = np.argsort(self.table_t)
        self.table_t = self.table_t[sort_times]
        self.table_M = masses[sort_times]
        # 
        self.Z = self.Z[sort_times]
        # Unused variables
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        self.table_SFR = np.gradient(masses, times)
        self.table_dot_SFR = np.gradient(self.table_SFR, times)
        self.table_ddot_SFR = np.gradient(self.table_dot_SFR, times)
        

    def integral_SFR(self, times):
        # interpolator = interpolate.interp1d(self.table_t.value, self.table_M.value,
        #                                     kind='cubic',
        #                                     bounds_error=False,
        #                                     fill_value=(0, self.table_M.value[-1])
        #                                     )
        interpolator = interpolate.Akima1DInterpolator(
           self.table_t.value, self.table_M.value)
        integral = interpolator(times.to(self.table_t.unit).value) * self.table_M.unit
        integral[times > self.table_t[-1]] = self.table_M[-1]
        integral[times < self.table_t[0]] = 0
        return integral
    
    def integral_Z_SFR(self, times):
        interpolator = interpolate.Akima1DInterpolator(
           self.table_t.value, self.table_M.value * self.Z.value)
        integral = interpolator(times.to(self.table_t.unit).value
                                ) * self.table_M.unit * self.Z.unit
        integral[times > self.table_t[-1]] = self.table_M[-1] * self.Z[-1].value
        integral[times < self.table_t[0]] = 0
        idx = np.where((times > self.table_t[0]) & (times < self.table_t[1]))
        integral[idx] *= np.sqrt((
           times[idx] - self.table_t[0]) / (self.table_t[1] - self.table_t[0]))
        return integral

    def SFR(self, times):
        return np.interp(times, self.table_t, self.table_SFR, left=0, right=0)

    def dot_SFR(self, times):
        return np.interp(times, self.table_t, self.table_dot_SFR, left=0, right=0)
    
    def ddot_SFR(self, times):
        return np.interp(times, self.table_t, self.table_ddot_SFR, left=0, right=0)

#-------------------------------------------------------------------------------
class Tabular_Illustris(Tabular_MFH):
#-------------------------------------------------------------------------------

    def __init__(self, filename, t0, **kwargs):
        # TODO: get rid of t0 !
        with fits.open(filename) as hdul:
            lb_time = hdul[1].data['lookback_time'] * u.Gyr
            mass_formed = np.sum(hdul[3].data, axis=1) *u.Msun # sum over metallicities
            t_sorted = (t0-lb_time)[::-1]
            mfh_sorted = np.cumsum(mass_formed[::-1])
            #print('aqui', t_sorted, mfh_sorted)
            Tabular_MFH.__init__(self, t_sorted, mfh_sorted, **kwargs) # t [Gyr], M[Msun]

#-------------------------------------------------------------------------------
class Tabular_CIGALE(Tabular_MFH):
#-------------------------------------------------------------------------------

    def __init__(self, filename, t0, **kwargs):
        f=open(filename,"r")
        lines=f.readlines()
        time=[]
        mass=[]
        for x in lines:
            time.append(float(x.strip().split(' ')[0])*1e-3) #Myr -> Gyr
            mass.append(float(x.strip().split(' ')[1])*1e6) #Msun/year -> Msun/Myr
        f.close()
        time = 13.7*u.Gyr-time*u.Gyr
        cum_mass = np.cumsum(mass[::-1])*u.Msun #Msun/Myr -> Msun
        Tabular_MFH.__init__(self, time[::-1], cum_mass, **kwargs)
        

#-------------------------------------------------------------------------------
class Tabular_CIGALE_models(Tabular_MFH):
#-------------------------------------------------------------------------------

    def __init__(self, hdul, factor, best_sfh_age, t0, free_age, **kwargs):
        
        if free_age:
            age_start = 13699- best_sfh_age
        else:
            age_start = 0
            
        #print('a',age_start)
        time = age_start*1e-3*u.Gyr +hdul[1].data['time']*1e-3*u.Gyr
        cum_mass = np.cumsum(factor*hdul[1].data['SFR'])*1e6*u.Msun #Msun/Myr -> Msun
        Tabular_MFH.__init__(self, time, cum_mass, **kwargs)

            
#-------------------------------------------------------------------------------
class Tabular_Prospector(Tabular_MFH):
#-------------------------------------------------------------------------------

    def __init__(self, prospector_model, t0, **kwargs):
        # TODO: get rid of t0 !
        x_bins_prospector = np.power(10, prospector_model['agebins'][()]).flatten()*1e-9 #Gyrs
            
        y_prospector=prospector_model['best_log_mass'][()]
        y_prospector = 10**y_prospector #Pasamos a mass = Msun
        y_prospector = np.cumsum(y_prospector[::-1]) #Masa acumulada
        y_prospector=np.vstack((y_prospector,y_prospector)).T.flatten() #Msun // Msun/year // Se añade lbt = 0 (ojo con el log)
        #print('prospector', x_bins_prospector*u.Gyr, y_prospector*u.Msun)
        Tabular_MFH.__init__(self, x_bins_prospector*u.Gyr, y_prospector*u.Msun, **kwargs)
            
#-------------------------------------------------------------------------------
class Exponential_quenched_SFR(Chemical_evolution_model):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    self.t_q = kwargs['t_quench']*u.Gyr
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

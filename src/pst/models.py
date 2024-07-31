import numpy as np
#import pylab as plt
from astropy import units as u
from astropy.io import fits
from scipy import special
import pst

from scipy import interpolate
from abc import ABC, abstractmethod


class ChemicalEvolutionModel(ABC):
    """TODO

    Description
    -----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, **kwargs):
        self.M_gas = kwargs.get('M_gas', 0*u.Msun)
        self.Z = kwargs.get('Z', 0.02)

    def interpolate_ssp_masses(self, SSP: pst.SSP.SSPBase, t_obs: u.Quantity):
        """Interpolate the star formation history to compute the SSP stellar massess.

        Description
        -----------
        This method computes the spectra energy distribution resulting from the
        chemical evolution model observed at a given time.

        Parameters
        ----------
        - SSP: pst.SSP.SSP
            The SSP model to used for synthezising the SED.
        - t_obs: astropy.Quantity
            Cosmic time at which the galaxy is observed. This will prevent the
            use the SSP with ages older than `t_obs`.

        Returns
        -------
        - masses: astropy.Quantity
            Corresponding stellar mass of each SSP.
        """
        age_bins = np.hstack(
            [0 << u.yr, np.sqrt(SSP.ages[1:] * SSP.ages[:-1]), 1e12 << u.yr])
        t_bins = t_obs - age_bins
        t_bins = t_bins[t_bins >= 0]
        M_t = self.integral_SFR(t_bins)
        M_bin = np.hstack([M_t[:-1]-M_t[1:], M_t[-1]])
        MZ_t = self.integral_Z_SFR(t_bins)
        MZ_bin = np.hstack([MZ_t[:-1]-MZ_t[1:], MZ_t[-1]])
        z_bin = np.clip(MZ_bin / (M_bin + 1 * u.kg),
                        SSP.metallicities[0],
                        SSP.metallicities[-1]) << u.dimensionless_unscaled

        weights = np.zeros((SSP.metallicities.size, SSP.ages.size))
        z_indices = np.searchsorted(
            SSP.metallicities, z_bin).clip(
            min=1, max=SSP.metallicities.size - 1)
        t_indices = np.arange(0, M_bin.size, dtype=int)
        weight_Z = np.log(
                    z_bin / SSP.metallicities[z_indices - 1]) / np.log(
                    SSP.metallicities[z_indices] / SSP.metallicities[z_indices-1]
                    )
        weights[z_indices, t_indices] = weight_Z
        weights[z_indices - 1, t_indices] = 1 - weight_Z
        weights[:, t_indices] = weights[:, t_indices] * M_bin[np.newaxis, :]
        weights = weights << u.Msun
        return weights
        
    def compute_SED(self, SSP : pst.SSP.SSPBase, t_obs : u.Quantity,
                    allow_negative=True):
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
            Cosmic time at which the galaxy is observed. This will prevent the
            use the SSP with ages older than `t_obs`.
        - allow_negative: bool, default=True
            Allow for some SSPs to have negative masses during the computation
            of the resulting SED.
        
        Returns
        -------
        - sed: astropy.Quantity
            Spectral energy distribution in the same units as `SSP.L_lambda`.
        """
        weights = self.interpolate_ssp_masses(SSP, t_obs)
        if not allow_negative:
            mask = (weights > 0) & np.isfinite(weights)
        else:
            mask = np.isfinite(weights)
        sed = np.sum(weights[mask, np.newaxis] * SSP.L_lambda[mask, :],
                    axis=(0))
        return sed

    def compute_photometry(self, ssp, t_obs, photometry=None):
        weights = self.interpolate_ssp_masses(ssp, t_obs)
        extra_dim = photometry.ndim - weights.ndim
        if extra_dim > 0:
            new_dims = tuple(np.arange(extra_dim))
            np.expand_dims(weights, new_dims)
        if photometry is None:
            photometry = ssp.photometry
        model_photometry = np.sum(photometry * weights, axis=(-1, -2))
        return model_photometry

    @abstractmethod
    def integral_SFR(self):
       pass

    @abstractmethod
    def integral_Z_SFR(self):
       pass

#-------------------------------------------------------------------------------
class Single_burst(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']
    self.t = kwargs['t_burst']
    ChemicalEvolutionModel.__init__(self, **kwargs)

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
class Exponential_SFR(ChemicalEvolutionModel):
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
        ChemicalEvolutionModel.__init__(self, **kwargs)

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
class Exponential_SFR_delayed(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    ChemicalEvolutionModel.__init__(self, **kwargs)

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

            cum_mass = np.cumsum(p.integral_SFR(t))
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
        
    def integral_SFR(self, time, **kwargs):
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
        
       
    def SFR(self, time, **kwargs):       
        t_hat = (1 - time/self.t0)
        self.get_sigma = kwargs.get('get_sigma', False)
        self.get_components = kwargs.get('get_components', False)
        self.fit_components = kwargs.get('fit_components', None)
        
        if self.fit_components is None:   
            M=[]
            N = len(self.coeffs)
            for n in range(1, N+1):      
                m=(n*t_hat**(n-1))/self.t0
                
                m[t_hat > self.t_hat_start] = 0.
                m[t_hat < self.t_hat_end] = 0.
                M.append(m)
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
        
    #The derivatives are not up to date
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
class Gaussian_burst(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_stars']*u.Msun
    self.tb = kwargs['t']*u.Gyr             # Born time
    self.c = kwargs['c']*u.Gyr # En Myr
    ChemicalEvolutionModel.__init__(self, **kwargs)

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

class LogNormal_MFH(ChemicalEvolutionModel):
    def __init__(self, alpha : float, z_today : u.Quantity,
                 lnt0: float, scale:float, m_today=1.0 << u.Msun,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.z_today = z_today
        self.lnt0 = lnt0
        self.scale = scale
        self.m_today = m_today

    @property
    def Z(self):
        return 

    @Z.setter
    def Z(self, z):
        pass

    def integral_SFR(self, times: u.Quantity):
        z = - (np.log(times.to_value("Gyr")) - self.lnt0) / self.scale
        m = 0.5 * (1 - special.erf(z / np.sqrt(2)))
        return m / m.max() * self.m_today

    def integral_Z_SFR(self, times: u.Quantity):
        m = self.integral_SFR(times)
        z_star = self.z_today * np.power(m / m.max(), self.alpha)
        return m * z_star

class LogNormalQuenched_MFH(LogNormal_MFH):
    __slots__ = 'alpha', 'z_today', 'lnt0', 'scale', 't_quench', 'tau_quench'
    def __init__(self, alpha : float, z_today : u.Quantity,
                 lnt0: float, scale:float,
                 t_quench: u.Quantity, tau_quench: u.Quantity, **kwargs):
        super().__init__(alpha, z_today, lnt0, scale, **kwargs)
        self.alpha = alpha
        self.z_today = z_today
        self.lnt0 = lnt0
        self.scale = scale
        self.t_quench = t_quench
        self.tau_quench = tau_quench

    def integral_SFR(self, times: u.Quantity):
        lognorm = super().integral_SFR(times)
        q = times >= self.t_quench
        if q.any():
            lognorm[q] = lognorm[q][0] * (1 - np.exp(-times[q] / self.tau_quench))
        return lognorm / lognorm.max() * self.m_today

#-------------------------------------------------------------------------------
class Tabular_MFH(ChemicalEvolutionModel):
    """Chemical evolution model based on a grid of times and metallicities.
    
    Description
    -----------
    This model represents the chemical evolution of a galaxy by means of a
    discrete grid of ages and metallicities

    Attributes
    ----------
    - table_t: astropy.Quantity
        Tabulated cosmic time.
    - table_M: astropy.Quantity
        Total stellar mass at each cosmic time step.
    - Z: astropy.Quantity
        Average stellar metallicity at each cosmic time step.

    Methods
    -------
    See `pst.models.ChemicalEvolutionModel` documentation. #FIXME

    """
    def __init__(self, times, masses, **kwargs):
        super().__init__(**kwargs)
        self.table_t = times
        # Make sure that time is cresscent
        sort_times = np.argsort(self.table_t)
        self.table_t = self.table_t[sort_times]
        self.table_M = masses[sort_times]
        # 
        self.Z = self.Z[sort_times]
        # FIXME: this variables should not be declared here
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        # FIXME: are we using this quantities?
        self.table_SFR = np.gradient(masses, times)
        self.table_dot_SFR = np.gradient(self.table_SFR, times)
        self.table_ddot_SFR = np.gradient(self.table_dot_SFR, times)


    def integral_SFR(self, times: u.Quantity):
        """Evaluate the integral of the SFR over a given set of times.
        
        Description
        -----------
        This method evaluates the integral:
            math::
            \int_{0}^{t} SFR(t') dt'
        at each time input time :math:`t`.
        ``
        Parameters
        ----------
        - times: astropy.units.Quantity
            Cosmic times at which the integral will be evaluated.

        Returns
        -------
        - integral: astropy.units.Quantity
            Integral evaluated at each input time.
        """
        interpolator = interpolate.Akima1DInterpolator(
           self.table_t, self.table_M)
        integral = interpolator(times) << self.table_M.unit
        integral[times > self.table_t[-1]] = self.table_M[-1]
        integral[times < self.table_t[0]] = 0

        # integral = np.interp(times.to(self.table_t.unit).value,
        #                      self.table_t.value, self.table_M.value,
        #                      left=0, right=self.table_M[-1].value) * self.table_M.unit
        return integral
    
    def integral_Z_SFR(self, times):
        """Evaluate the integral of the average metallicity over a given set of times.
        
        Description
        -----------
        This method evaluates the integral:
            math::
            \int_{0}^{t} Z(t') SFR(t') dt'
        at each time input time :math:`t`.
        ``
        Parameters
        ----------
        - times: astropy.units.Quantity
            Cosmic times at which the integral will be evaluated.

        Returns
        -------
        - integral: astropy.units.Quantity
            Integral evaluated at each input time.
        """
        interpolator = interpolate.Akima1DInterpolator(
           self.table_t, self.table_M * self.Z)
        integral = interpolator(times) << self.table_M.unit * self.Z.unit
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


class Tabular_ZPowerLaw(Tabular_MFH):
    """Chemical evolution model based on a grid of times and metallicities.
    
    Description
    -----------
    This model represents the chemical evolution of a galaxy by means of a
    discrete grid of ages and metallicities

    Attributes
    ----------
    - table_t: astropy.Quantity
        Tabulated cosmic time.
    - table_M: astropy.Quantity
        Total stellar mass at each cosmic time step.
    - Z: astropy.Quantity
        Average stellar metallicity at each cosmic time step.

    Methods
    -------
    See `pst.models.ChemicalEvolutionModel` documentation. #FIXME

    """
    def __init__(self, times, masses, alpha, z_today, **kwargs):
        self.z_today = z_today
        self.alpha = alpha
        super().__init__(times, masses, **kwargs)
        

    @property
    def Z(self):
        return self.z_today * np.power(self.table_M / self.table_M[-1], self.alpha)

    @Z.setter
    def Z(self, z):
        pass

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
class Exponential_quenched_SFR(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    self.t_q = kwargs['t_quench']*u.Gyr
    ChemicalEvolutionModel.__init__(self, **kwargs)


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
class ASCII_file(ChemicalEvolutionModel):
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

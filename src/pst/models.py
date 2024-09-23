import numpy as np
from astropy import units as u
from scipy import special
import pst

from scipy import interpolate
from abc import ABC, abstractmethod

from pst.SSP import SSPBase
from astropy.visualization import quantity_support
quantity_support()

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
        pass
        #self.M_gas = kwargs.get('M_gas', 0*u.Msun)
        #self.Z = kwargs.get('Z', 0.02)

    @abstractmethod
    def stellar_mass_formed(self, time):
        return

    @abstractmethod
    def ism_metallicity(self, time):
        return

    @u.quantity_input
    def interpolate_ssp_masses(self, SSP: pst.SSP.SSPBase, t_obs: u.Gyr, oversample_factor=10):
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
        
        # define age bins from 0 to t_obs
        age_bins = np.hstack(
            [0 << u.yr, np.sqrt(SSP.ages[1:] * SSP.ages[:-1]), 1e12 << u.yr])
        age_bins = age_bins[:age_bins.searchsorted(t_obs) + 1]
        age_bins[-1] = t_obs
        # oversample
        w1 = np.arange(oversample_factor) / oversample_factor
        age_bins = np.hstack(
            [(1-w1)*age_bins[i] + w1*age_bins[i+1] for i in range(age_bins.size - 1)]
            + [t_obs])

        # find bin properties
        mass = self.stellar_mass_formed(t_obs - age_bins).to_value(u.Msun)
        bin_mass = mass[:-1] - mass[1:]
        bin_age = (age_bins[1:] + age_bins[:-1]) / 2
        bin_metallicity = self.ism_metallicity(t_obs - bin_age)
        
        '''
        # test plot
        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = axes
        ax.plot(bin_age, bin_metallicity, 'k-+')
        ax.plot(t_obs-self.table_t, self.table_metallicity, 'm-+')
        ax.set_xscale('log')
        '''
        
        # 2D interpolation
        
        age_index = np.clip(SSP.ages.searchsorted(bin_age),
                            1, SSP.ages.size - 1)
        weight_age = np.log(bin_age / SSP.ages[age_index - 1])
        weight_age /= np.log(SSP.ages[age_index] / SSP.ages[age_index-1])
        
        z_index = np.clip(SSP.metallicities.searchsorted(bin_metallicity),
                          1, SSP.ages.size - 1)
        weight_z = np.log(bin_metallicity / SSP.metallicities[z_index - 1])
        weight_z /= np.log(
            SSP.metallicities[z_index] / SSP.metallicities[z_index-1])

        weights = np.zeros((SSP.metallicities.size, SSP.ages.size))
        np.add.at(weights, (z_index, age_index),
                  bin_mass * weight_age * weight_z)
        np.add.at(weights, (z_index - 1, age_index),
                  bin_mass * weight_age * (1 - weight_z))
        np.add.at(weights, (z_index - 1, age_index - 1),
                  bin_mass * (1 - weight_age) * (1 - weight_z))
        np.add.at(weights, (z_index, age_index - 1),
                  bin_mass * (1 - weight_age) * weight_z)
        return weights << u.Msun
    
    #TODO: This method should be renamed by compute_spectra or compute_L_lambda
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

    def compute_photometry(self, ssp, t_obs, photometry=None,
                           allow_negative=True):
        """
        Compute the SED of a given model, observed at a given time.

        Description
        -----------
        This method computes the spectra energy distribution resulting from the
        chemical evolution model observed at a given time.

        Parameters
        ----------
        - ssp: pst.SSP.SSP
            The SSP model to used for synthezising the SED.
        - t_obs: astropy.Quantity
            Cosmic time at which the galaxy is observed. This will prevent the
            use the SSP with ages older than `t_obs`.
        - photometry: np.ndarray
            Array containing a grid of luminosities in multiple bands. If none,
            the default SSP photometry will be used. The last two dimensions of
            the array must be equal to the metallicity and age dimensionality of
            the SSP model.
        - allow_negative: bool, default=True
            Allow for some SSPs to have negative masses during the computation
            of the resulting SED.
        
        Returns
        -------
        - sed: astropy.Quantity
            Spectral energy distribution in the same units as `photometry`.
        """
        weights = self.interpolate_ssp_masses(ssp, t_obs)
        if photometry is None:
            photometry = ssp.photometry
        if not isinstance(photometry, u.Quantity):
            print("Assuming input photometry array in Jy/Msun")
            photometry *= u.Jy / u.Msun
        extra_dim = photometry.ndim - weights.ndim
        if extra_dim > 0:
            new_dims = tuple(np.arange(extra_dim))
            np.expand_dims(weights, new_dims)
        
        model_photometry = np.sum(photometry * weights, axis=(-1, -2))
        return model_photometry

    '''
    @abstractmethod
    def integral_Z_SFR(self):
       pass
    '''

#-------------------------------------------------------------------------------
class Single_burst(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_stars = kwargs['M_stars']
    self.t = kwargs['t_burst']
    ChemicalEvolutionModel.__init__(self, **kwargs)

# TODO: do this using np.select(); actually, does tb exist at all?
  def stellar_mass_formed(self, time):
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

    def stellar_mass_formed(self, time):
      return self.M_inf * ( 1 - np.exp(-time/self.tau) )

    def SFR(self, time):
      return self.M_inf*(np.exp(-time/self.tau))/self.tau

    def dot_SFR(self,time):
      return -self.M_inf*(np.exp(-time/self.tau))/(self.tau**2)


    def ddot_SFR(self,time):
      return self.M_inf*(np.exp(-time/self.tau))/(self.tau**3)


    def integral_Z_SFR(self, time):
      return self.Z * self.stellar_mass_formed(time)


#-------------------------------------------------------------------------------
class Exponential_SFR_delayed(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    ChemicalEvolutionModel.__init__(self, **kwargs)

  def stellar_mass_formed(self, time):
    return self.M_inf * ( 1 - np.exp(-time/self.tau)*(self.tau+time)/self.tau)

  def SFR(self,time):
      return self.M_inf*(time/self.tau**2)*np.exp(-time/self.tau)

  def dot_SFR(self,time):
      return -self.M_inf*((time-self.tau)/self.tau**3)*np.exp(-time/self.tau)

  def ddot_SFR(self,time):
      return self.M_inf*((time-2*self.tau)/self.tau**4)*np.exp(-time/self.tau)

  def integral_Z_SFR(self, time):
    return self.Z * self.stellar_mass_formed(time)


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

  def stellar_mass_formed(self, time):
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

    def stellar_mass_formed(self, times: u.Quantity):
        z = - (np.log(times.to_value("Gyr")) - self.lnt0) / self.scale
        m = 0.5 * (1 - special.erf(z / np.sqrt(2)))
        return m / m.max() * self.m_today

    def integral_Z_SFR(self, times: u.Quantity):
        m = self.stellar_mass_formed(times)
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

    def stellar_mass_formed(self, times: u.Quantity):
        lognorm = super().stellar_mass_formed(times)
        q = times >= self.t_quench
        if q.any():
            lognorm[q] = lognorm[q][0] * (1 - np.exp(-times[q] / self.tau_quench))
        return lognorm / lognorm.max() * self.m_today

#-------------------------------------------------------------------------------
class Tabular_CEM(ChemicalEvolutionModel):
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
        Gas metallicity at each cosmic time step.

    See also
    --------
    :class:`pst.models.ChemicalEvolutionModel` documentation.

    """
    def __init__(self, times, masses, metallicities, **kwargs):
        super().__init__(**kwargs)
        self.table_t = times
        # Make sure that time is crescent
        sort_times = np.argsort(self.table_t)
        self.table_t = self.table_t[sort_times]
        self.table_mass = masses[sort_times]
        self.table_metallicity = metallicities[sort_times]

        '''
        # FIXME: this variables should not be declared here
        self.t_hat_start = kwargs.get('t_hat_start', 1.)
        self.t_hat_end = kwargs.get('t_hat_end', 0.)
        # FIXME: are we using this quantities?
        self.table_SFR = np.gradient(masses, times)
        self.table_dot_SFR = np.gradient(self.table_SFR, times)
        self.table_ddot_SFR = np.gradient(self.table_dot_SFR, times)
        '''


    @u.quantity_input
    def stellar_mass_formed(self, times: u.Gyr) -> u.Msun:
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
           self.table_t, self.table_mass)
        integral = interpolator(times) << self.table_mass.unit
        integral[times > self.table_t[-1]] = self.table_mass[-1]
        integral[times < self.table_t[0]] = 0
        return integral
    
    @u.quantity_input
    def ism_metallicity(self, times: u.Gyr) -> u.dimensionless_unscaled:
        """Evaluate the integral of the SFR over a given set of times.
        
        Description
        -----------
        Return the metallicity Z(t) of the interstellar medium (gas and dust)
        at a certain set of cosmic times (i.e. since the Big Bang).
        
        Parameters
        ----------
        times: astropy.units.Quantity
            Cosmic times at which the metallicity will be evaluated.

        Returns
        -------
        z_t: astropy.units.Quantity
            Vector with the ISM metallicity at each input time.
        """
        interpolator = interpolate.Akima1DInterpolator(
           self.table_t, self.table_metallicity)
        integral = interpolator(times)
        integral[times > self.table_t[-1]] = self.table_metallicity[-1]
        integral[times < self.table_t[0]] = self.table_metallicity[0]

        # integral = np.interp(times.to(self.table_t.unit).value,
        #                      self.table_t.value, self.table_M.value,
        #                      left=0, right=self.table_M[-1].value) * self.table_M.unit
        return integral
    
    '''
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
           self.table_t, self.table_MZ)
        integral = interpolator(times) << self.table_M.unit * self.Z.unit
        integral[times > self.table_t[-1]] = self.table_MZ[-1]
        integral[times < self.table_t[0]] = 0
        return integral

    def SFR(self, times):
        return np.interp(times, self.table_t, self.table_SFR, left=0, right=0)

    def dot_SFR(self, times):
        return np.interp(times, self.table_t, self.table_dot_SFR, left=0, right=0)
    
    def ddot_SFR(self, times):
        return np.interp(times, self.table_t, self.table_ddot_SFR, left=0, right=0)

    def interpolate_ssp_masses(self, ssp: pst.SSP.SSPBase, t_obs: u.Quantity):
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
            [0 << u.yr, np.sqrt(ssp.ages[1:] * ssp.ages[:-1]), 1e12 << u.yr])
        t_bins = t_obs - age_bins
        # Discard those SSPs older than t_obs
        valid_bins = t_bins >= 0
        t_bins = t_bins[valid_bins]


        met_matrix = np.zeros((ssp.metallicities.size, self.Z.size))
        idx = np.arange(0, met_matrix.shape[1])
        met_bins = np.searchsorted(ssp.metallicities, self.Z, side='right')
        met_bins = met_bins.clip(min=1, max=ssp.metallicities.size - 1)
        
        int_z_history = self.Z.clip(min=ssp.metallicities[0].value,
                                    max=ssp.metallicities[-1].value)
        
        weight_Z = np.log(
                        int_z_history / ssp.metallicities.value[met_bins - 1]) / np.log(
                        ssp.metallicities.value[met_bins] / ssp.metallicities.value[met_bins-1]
                        )
        # Clip negative values close to zero due to numerical errors
        weight_Z = weight_Z.clip(min=0, max=1)
        
        met_matrix[met_bins, idx] = weight_Z
        met_matrix[met_bins - 1, idx] = 1 - weight_Z
        
        met_matrix[0, met_bins == 0] = 1
        met_matrix[-1, met_bins == ssp.metallicities.size] = 1
        
        cum_met_matrix = np.cumsum(met_matrix, axis=1)

        interpolator = interpolate.interp1d(self.table_t, cum_met_matrix,
                                            fill_value=(
                                                np.zeros(cum_met_matrix.shape[0]),
                                                cum_met_matrix[:, -1]),
                                            kind="linear",
                                            bounds_error=False,
                                            axis=1)
        
        cum_z_mat = interpolator(t_bins.value)
        z_weights = np.hstack((cum_z_mat[:, :-1] - cum_z_mat[:, 1:],
                               cum_z_mat[:, -1][:, np.newaxis]))
        # Renormalize across metallicity axis
        z_weights = np.clip(z_weights, a_min=0, a_max=None)
        z_weights /= np.sum(z_weights, axis=0)
        z_weights = np.nan_to_num(z_weights, nan=0.0)
        
        M_t = self.stellar_mass_formed(t_bins)
        M_bin = np.hstack([M_t[:-1]-M_t[1:], M_t[-1]])

        weights = np.zeros((ssp.metallicities.size, ssp.ages.size))
        t_indices = np.arange(0, M_bin.size, dtype=int)
        weights[:, t_indices] = z_weights * M_bin[np.newaxis, :]
        return weights << u.Msun
    '''


class Tabular_CEM_ZPowerLaw(Tabular_CEM):
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
    def __init__(self, times, masses, alpha, ism_metallicity_today, **kwargs):
        self.ism_metallicity_today = ism_metallicity_today
        self.alpha = alpha
        super().__init__(times, masses, **kwargs)
        

    @property
    def table_metallicity(self):
        return self.ism_metallicity_today * np.power(self.table_mass / self.table_mass[-1], self.alpha)
    
"""
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
'''
#-------------------------------------------------------------------------------
class Exponential_quenched_SFR(ChemicalEvolutionModel):
#-------------------------------------------------------------------------------

  def __init__(self, **kwargs):
    self.M_inf = kwargs['M_inf']*u.Msun
    self.tau = kwargs['tau']*u.Gyr
    self.Z = kwargs['Z']
    self.t_q = kwargs['t_quench']*u.Gyr
    ChemicalEvolutionModel.__init__(self, **kwargs)


  def stellar_mass_formed(self, time):
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
    return self.Z * self.stellar_mass_formed(time)


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
    self.stellar_mass_formed_table = np.cumsum( dm )
    self.integral_Z_SFR_table = np.cumsum( self.Z_table*dm )

  def set_current_state(self, galaxy):
    galaxy.M_stars = self.stellar_mass_formed( galaxy.today ) #TODO: account for stellar death
    Z = np.interp( galaxy.today, self.t_table, self.Z_table )
    galaxy.M_gas = galaxy.M_stars*max(.03/(Z+1e-6)-1,1e-6) # TODO: read from files
    galaxy.Z = Z

  def stellar_mass_formed(self, time):
    return np.interp( time, self.t_table, self.stellar_mass_formed_table )

  def integral_Z_SFR(self, time):
    return np.interp( time, self.t_table, self.integral_Z_SFR_table )

  #def plot(self):
    #plt.semilogy( self.t_table/units.Gyr, self.SFR_table/(units.Msun/units.yr) )
    #plt.semilogy( self.t_table/units.Gyr, self.stellar_mass_formed_table/units.Msun )
    #plt.semilogy( self.t_table/units.Gyr, self.Z_table )
    #plt.show()
"""

class ParticleGridCEM(ChemicalEvolutionModel):
    """A Chemical Evolution Model using a individual SSP data.
    
    Description
    -----------
    This CEM uses indiviudal SSP to reconstruct the CSP.

    Attributes
    ----------
    time_form : :class:`numpy.array` or :class:`astropy.units.Quantity`
        Vector containing the cosmic formation time of each SSP particle.
        If the ``type`` is ``numpy.array``, units are assumed to be in Gyr.
    metallicities : :class:`numpy.array` or :class:`astropy.units.Quantity`
        Vector containing the metallicity of each SSP particle. If the ``type`` is
        ``numpy.array``, ``astropy.units.dimensionless_unscaled`` is used as unit.
    masses : :class:`numpy.array` or :class:`astropy.units.Quantity`
        Vector containing the mass of each SSP particle. If the ``type`` is
        ``numpy.array``, units are assumed to be in solar masses.
    """
    def __init__(self, time_form, metallicities, masses):
        self.time_form, self.metallicities, self.masses = (
            time_form, metallicities, masses)

    @property
    def time_form(self):
        return self._time_form

    @time_form.setter
    def time_form(self, values):
        if not isinstance(values, u.Quantity):
            self._time_form = values << u.Gyr
        else:
            self._time_form = values

    @property
    def metallicities(self):
        return self._metallicities

    @metallicities.setter
    def metallicities(self, values):
        if not isinstance(values, u.Quantity):
            self._metallicities = values << u.dimensionless_unscaled
        else:
            self._metallicities = values

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, values):
        if not isinstance(values, u.Quantity):
            self._masses = values << u.Msun
        else:
            self._masses = values

    def interpolate_ssp_masses(self, ssp: SSPBase, t_obs: u.Quantity):
        """Interpolate the particles to a base of SSPs.
        
        Parameters
        ----------
        ssp : :class:`pst.SSP.SSPBase`
            A SSP model.
        t_obs : :class:`astropy.units.Quantity`
            The age of the Universe at the time of the observation.
        
        Returns
        -------
        ssp_weights : :class:`astropy.units.Quantity`
            A 2D array containing the stellar mass associated to each SSP.
        """
        valid_particles = self.time_form <= t_obs
        particle_ages = (t_obs - self.time_form[valid_particles]).to_value("Gyr").clip(
            ssp.ages.to_value('Gyr').min(), ssp.ages.to_value('Gyr').max())
        particle_metallicity = self.metallicities[valid_particles].clip(
            ssp.metallicities.value[0], ssp.metallicities.value[-1]).value

        age_idx = np.searchsorted(ssp.ages.to_value("Gyr"), particle_ages,
                                  side="right").clip(min=1, max=ssp.ages.size - 1)
        metalliticy_idx = np.searchsorted(ssp.metallicities.value,
                                          particle_metallicity,
                                          side="right").clip(min=1, max=ssp.metallicities.size - 1)
        # Interpolation weights of the right element
        weight_Z = np.log(
                    particle_metallicity / ssp.metallicities.value[metalliticy_idx - 1]) / np.log(
                    ssp.metallicities.value[metalliticy_idx] / ssp.metallicities.value[metalliticy_idx - 1]
                    )
        weight_age = (particle_ages - ssp.ages.to_value('Gyr')[age_idx - 1]) / (
                    ssp.ages.to_value('Gyr')[age_idx] - ssp.ages.to_value('Gyr')[age_idx - 1]
                    )

        weights = np.zeros(ssp.L_lambda.shape[:-1]) << self.masses.unit
        weights[metalliticy_idx, age_idx] = weight_Z * weight_age * self.masses[valid_particles]
        weights[metalliticy_idx - 1, age_idx] = (1 - weight_Z) * weight_age * self.masses[valid_particles]
        weights[metalliticy_idx - 1, age_idx - 1] = (1 - weight_Z) * (1 - weight_age) * self.masses[valid_particles]
        weights[metalliticy_idx, age_idx - 1] = weight_Z * (1 - weight_age) * self.masses[valid_particles]

        return weights

    def integral_SFR(self, time):
        sort_idx = np.argsort(self.time_form)
        mass_history = np.cumsum(self.masses[sort_idx])
        return np.interp(time, self.time_form[sort_idx], mass_history)

    def integral_Z_SFR(self, time):
        sort_idx = np.argsort(self.time_form)
        z_mass_history = np.cumsum(self.masses[sort_idx]
                                 * self.metallicities[sort_idx])
        return np.interp(time, self.time_form[sort_idx], z_mass_history)

# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/
# -----------------------------------------------------------------------------

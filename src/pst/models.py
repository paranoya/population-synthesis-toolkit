from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from scipy import special
from scipy import interpolate

from pst.SSP import SSPBase
from pst.utils import check_unit, SQRT_2


# CEM utils and mixins

class MassPropMetallicityMixin:
    r"""Model mixin where the metallicity is proportional to the stellar mass
    
    .. math::
        Z(t) = Z_{\rm today} \cdot \left(\frac{M_{\star}(t)}{M_{\star}(\rm today)}\right)^\alpha
        :label: powlaw-metallicity

    where :math:`Z_{\rm today}` is the ISM metallicity at present,
    :math:`M_{\star}(t)` is the stellar mass formed at time :math:`t`, and
    :math:`M_{\star}(\rm today)` is the stellar mass at present.
    The exponent :math:`\alpha` regulates the chemical enrichment rate with stellar mass.

    """
    @property
    def ism_metallicity_today(self):
        """ISM metals mass fraction at present."""
        return self._ism_metallicity_today
    
    @ism_metallicity_today.setter
    def ism_metallicity_today(self, value):
        self._ism_metallicity_today = value
    
    @property
    def alpha_powerlaw(self):
        """Stellar mass power-law exponent."""
        return self._alpha_powerlaw

    @alpha_powerlaw.setter
    def alpha_powerlaw(self, value):
        self._alpha_powerlaw = value

    def ism_metallicity(self, times):
        """Evaluate the ISM metallicity at a given set of times, see eq. :eq:`powlaw-metallicity`.

        Parameters
        ----------
        times : :class:`astropy.units.Quantity`
            Cosmic times at which the metallicity will be evaluated.

        Returns
        -------
        z_t : :class:`astropy.units.Quantity`
            Vector with the ISM metallicity at each input time.
        """
        m = self.stellar_mass_formed(times)
        return self.ism_metallicity_today * np.power(m / self.mass_today, self.alpha_powerlaw)

def sfh_quenching_decorator(stellar_mass_formed):
    """A decorator for including a quenching event in a given SFH."""
    def wrapper_stellar_mass_formed(*args):
        quenching_time = getattr(args[0], "quenching_time", 20.0 << u.Gyr)
        stellar_mass = stellar_mass_formed(*args)
        final_mass = stellar_mass_formed(args[0], quenching_time)
        return np.where(args[1] < quenching_time, stellar_mass, final_mass)
    return wrapper_stellar_mass_formed


class ChemicalEvolutionModel(ABC):
    """
    Abstract base class for chemical evolution models.

    This class provides an interface for modeling the chemical and stellar
    evolution of a galaxy over time. It includes methods for computing the 
    Spectral Energy Distribution (SED), stellar mass, and photometry from 
    a given Simple Stellar Population (SSP) model. The specific methods for 
    computing the star formation rate (SFR) and the metallicity evolution 
    (Z-SFR) need to be implemented in a subclass.
    """
    
    @property
    def today(self):
        return self._today

    @today.setter
    def today(self, value):
        if value is not None:
            self._today = check_unit(value, u.Gyr)
        else:
            self._today = None

    def __init__(self, **kwargs):
        self.today = kwargs.get("today")

    @abstractmethod
    def stellar_mass_formed(self, time) -> u.Quantity:
        """Total stellar mass formed at a given time."""
        return

    @abstractmethod
    def ism_metallicity(self, time) -> u.Quantity:
        """ISM metals mass fraction at a given time."""
        return

    @u.quantity_input
    def interpolate_ssp_masses(self, ssp: SSPBase, t_obs: u.Gyr,
                               oversample_factor=10) -> u.Quantity:
        """
        Interpolate the star formation history to compute the SSP stellar masses.

        This method computes the star formation history of a galaxy over time
        and uses it to interpolate the stellar masses for a given Simple Stellar
        Population (SSP) model at the time of observation.

        Parameters
        ----------
        SSP : :class:`pst.SSP.SSPBase`
            The Simple Stellar Population (SSP) model used for synthesizing the SED.
        t_obs : astropy.Quantity
            The cosmic time at which the galaxy is observed. Only SSPs with ages
            younger than `t_obs` are used.
        oversample_factor : int
            Ages oversampling factor. 

        Returns
        -------
        weights : astropy.Quantity
            Stellar masses corresponding to each SSP age and metallicity, in units
            of solar masses.
        """

        # define age bins from 0 to t_obs
        age_bins = np.hstack(
            [0 << u.yr, np.sqrt(ssp.ages[1:] * ssp.ages[:-1]), 1e12 << u.yr])
        age_bins = age_bins[:age_bins.searchsorted(t_obs) + 1]
        age_bins[-1] = t_obs
        # oversample
        w1 = np.arange(oversample_factor) / oversample_factor
        age_bins = np.hstack(
            [(1-w1) * age_bins[i] + w1 * age_bins[i + 1] for i in range(age_bins.size - 1)]
            + [t_obs])

        # find bin properties
        mass = self.stellar_mass_formed(t_obs - age_bins)
        bin_mass = mass[:-1] - mass[1:]
        bin_age = (age_bins[1:] + age_bins[:-1]) / 2
        bin_metallicity = self.ism_metallicity(t_obs - bin_age)
 
        return ssp.get_weights(ages=bin_age,
                               metallicities=bin_metallicity,
                               masses=bin_mass)

    def time_at_stellar_mass_frac(self, frac, today=None, time_res=30 << u.Myr):
        """
        Get the cosmic time at which the model formed a fraction of the total stellar mass.

        Parameters
        ----------
        frac : float
            Input fraction.
        today : float or astropy.units.Quantity
            Age of the Universe at the time of the observation. If not provided,
            the defult is to use ``self.today'' (returning an error if not set).
        time_res : float or astropy.units.Quantity
            Time-step resolution for sampling the SFH.
        Returns
        -------
        frac_times : astropy.units.Quantity
            Fraction formation times.
        """
        frac = np.atleast_1d(frac)
        if today is None:
            if self.today is None:
                raise ValueError(
                    "current CEM does not have attribute 'today' set,"
                    + " it must be provided by the user")
            else:
                today = self.today
        else:
            today = check_unit(today, u.Gyr)

        time_res = check_unit(time_res, u.Gyr)
        dummy_time = np.arange(0, today.to_value("Gyr"),
                               time_res.to_value("Gyr")) << u.Gyr
        mass_history = self.stellar_mass_formed(dummy_time)
        frac_history = mass_history / mass_history[-1]
        idx = np.searchsorted(frac_history, frac).clip(
            min=1, max=frac_history.size - 1)
        t_frac = (
            dummy_time[idx - 1] * (frac - frac_history[idx]) / (frac_history[idx] - frac_history[idx - 1])
         + dummy_time[idx] * (1 - (frac - frac_history[idx]) / (frac_history[idx] - frac_history[idx - 1])
        ))
        return t_frac

    def compute_SED(self, ssp : SSPBase, t_obs : u.Quantity,
                    allow_negative=False) -> u.Quantity:
        """
        Compute the Spectral Energy Distribution (SED) resulting from the SFH.

        This method synthesizes the SED resulting from the chemical evolution model,
        observed at a given time, using the provided SSP model.

        Parameters
        ----------
        SSP : pst.SSP.SSPBase
            The Simple Stellar Population (SSP) model used for synthesizing the SED.
        t_obs : astropy.Quantity
            The cosmic time at which the galaxy is observed.
        allow_negative : bool, optional
            Whether to allow SSPs with negative masses in the SED computation.
            Default is True.

        Returns
        -------
        sed : astropy.Quantity
            The spectral energy distribution, in the same units as `SSP.L_lambda`.
        
        See also
        --------
        :func:`interpolate_ssp_masses`
        """
        weights = self.interpolate_ssp_masses(ssp, t_obs)
        if not allow_negative:
            mask = (weights > 0) & np.isfinite(weights)
        else:
            mask = np.isfinite(weights)
        sed = np.sum(weights[mask, np.newaxis] * ssp.L_lambda[mask, :],
                    axis=(0))
        return sed

    def compute_photometry(self, ssp, t_obs, photometry=None) -> u.Quantity:
        """
        Compute the synthetic photometry using a SSP at a given time.

        This method computes the photometric fluxes associated to
        the SFH by synthesizing the fluxes of the input SSP model.

        Parameters
        ----------
        ssp : pst.SSP.SSPBase
            The Simple Stellar Population (SSP) model used for generating the
            synthetic photometry.
        t_obs : astropy.Quantity
            The cosmic time at which the galaxy is observed.
        photometry : np.ndarray, optional
            A grid of luminosities in multiple photometric bands. If None, the 
            default SSP photometry will be used. The last two dimensions must 
            match the metallicity and age grid of the SSP model.

        Returns
        -------
        model_photometry : astropy.Quantity
            The photometry of the galaxy in the same units as the input photometry.
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


#-------------------------------------------------------------------------------
class SingleBurstCEM(ChemicalEvolutionModel):
    """
    Single-burst star formation model.

    This class models a galaxy's star formation history as a single burst
    occurring at a specific time, after which no further star formation occurs.

    Attributes
    ----------
    mass_burst : astropy.Quantity
        Total stellar mass formed in the burst.
    time_burst : astropy.Quantity
        Time of the starburst in cosmic time.
    burst_metallicity : astropy.Quantity
        Metallicity of the burst.
    """
    def __init__(self, **kwargs):
        self.mass_burst = check_unit(kwargs['mass_burst'], u.Msun)
        self.time_burst = check_unit(kwargs['time_burst'], u.Gyr)
        self.burst_metallicity = kwargs.get("burst_metallicity",
                                            0.02 * u.dimensionless_unscaled)
        ChemicalEvolutionModel.__init__(self, **kwargs)

    @u.quantity_input
    def stellar_mass_formed(self, time : u.Gyr):
        """Total stellar mass formed at a given time."""
        mass = np.zeros(time.size, dtype=float) * self.mass_burst.unit
        mass[time >= self.time_burst] = self.mass_burst
        return mass

    @u.quantity_input
    def ism_metallicity(self, time : u.Gyr):
        """ISM metals mass fraction at a given time."""
        return np.full(time.size, fill_value=self.burst_metallicity)


#-------------------------------------------------------------------------------
class ExponentialCEM(ChemicalEvolutionModel):
    r"""
    Exponentially declining star formation history model.

    This class models a galaxy's star formation rate as an exponentially
    declining function of time:

    .. math::
        M_\star(t) = M_{\rm inf} \cdot (1 - e^{-t/\tau})
 
    Attributes
    ----------
    stellar_mass_inf : astropy.Quantity
        Asymptotic stellar mass at infinite time.
    tau : astropy.Quantity
        Timescale of the exponential decline in star formation.
    metallicity : float
        Metallicity of the gas (constant).

    Examples
    --------
    Create an instance of the `ExponentialCEM` model and compute the stellar mass formed:

    >>> from astropy import units as u
    >>> from pst.models import ExponentialCEM
    >>> model = ExponentialCEM(stellar_mass_inf=1e10 * u.Msun, tau=2 * u.Gyr, metallicity=0.02)
    >>> time = [0.5, 1.0, 2.0] * u.Gyr
    >>> model.stellar_mass_formed(time)
    <Quantity [2.21199217e+09, 3.93469340e+09, 6.32120559e+09] solMass>    
    """
    def __init__(self, **kwargs):
        self.stellar_mass_inf = check_unit(kwargs['stellar_mass_inf'],
                                           default_unit=u.Msun)
        self.tau = check_unit(kwargs['tau'], default_unit=u.Gyr)
        self.metallicity = kwargs['metallicity']
        super().__init__(**kwargs)

    @u.quantity_input
    def stellar_mass_formed(self, time : u.Gyr):
      return self.stellar_mass_inf * ( 1 - np.exp(-time/self.tau) )

    @u.quantity_input
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.metallicity)


class ExponentialQuenchedCEM(ExponentialCEM):
    """
    Exponentially declining CEM model including a quenching event.

    Examples
    --------
    Create an instance of the `ExponentialQuenchedCEM` model and compute the stellar mass formed:

    >>> from astropy import units as u
    >>> from pst.models import ExponentialQuenchedCEM
    >>> model = ExponentialQuenchedCEM(stellar_mass_inf=1e10 * u.Msun, tau=2 * u.Gyr,
    ...                                 metallicity=0.02, quenching_time=1.5 * u.Gyr)
    >>> time = [0.5, 1.0, 2.0] * u.Gyr
    >>> model.stellar_mass_formed(time)
    <Quantity [2.21199217e+09, 3.93469340e+09, 5.27633447e+09] solMass>

    See also
    --------
    :class:`ExponentialCEM`
    """
    def __init__(self, **kwargs):
        self.quenching_time = check_unit(kwargs['quenching_time'], u.Gyr)
        super().__init__(**kwargs)

    @sfh_quenching_decorator
    def stellar_mass_formed(self, time : u.Gyr):
        return super().stellar_mass_formed(time)


#-------------------------------------------------------------------------------
class ExponentialDelayedCEM(ChemicalEvolutionModel):
    r"""
    Exponentially delayed star formation history model.

    This CEM models a galaxy's star formation rate as a delayed exponential
    function of time, where the SFR rises initially and then decays.

    .. math::
        M_\star(t) = M_{\rm inf} \cdot (1 - \frac{t + \tau}{\tau} \cdot e^{-t/\tau})
 
    The main difference from the :class:`ExponentialCEM` is that the stellar mass 
    grows linearly at early times before transitioning to an exponential decay.

    Attributes
    ----------
    tau : astropy.Quantity
        Timescale of the delayed exponential star formation rate.
    today : float or astropy.Quantity
        Cosmic time at the the time of the observation.
    mass_today : float or astropy.Quantity
        Total stellar mass formed at present.
    ism_metallicity_today : float
        Metallicity of the gas (constant).
    
    Examples
    --------
    Create an instance of the `ExponentialDelayedCEM` model and compute the stellar mass formed:

    >>> from astropy import units as u
    >>> from pst.models import ExponentialDelayedCEM
    >>> model = ExponentialDelayedCEM(tau=2 * u.Gyr, today=10 * u.Gyr,
    ...                                mass_today=1e10 * u.Msun, ism_metallicity_today=0.02)
    >>> time = [0.5, 1.0, 2.0] * u.Gyr
    >>> model.stellar_mass_formed(time)
    <Quantity [2.76154498e+08, 9.40043900e+08, 2.75373844e+09] solMass>
    """

    def __init__(self, **kwargs):
        self.tau = check_unit(kwargs['tau'], default_unit=u.Gyr)
        self.today = check_unit(kwargs['today'], u.Gyr)
        self.mass_today = check_unit(kwargs['mass_today'], u.Msun)
        self._mass_norm = 1
        mtoday = self.stellar_mass_formed(self.today)
        self._mass_norm = self.mass_today / mtoday

        self.ism_metallicity_today =  kwargs['ism_metallicity_today']
        ChemicalEvolutionModel.__init__(self, **kwargs)

    @u.quantity_input
    def stellar_mass_formed(self, time):
        return self.mass_today * (1 - np.exp(-time / self.tau)
            * (self.tau + time) / self.tau) * self._mass_norm

    @u.quantity_input
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.ism_metallicity_today)


class ExponentialDelayedZPowerLawCEM(MassPropMetallicityMixin, ExponentialDelayedCEM):
    """A :class:`ExponentialDelayedCEM` with a Mass-dependent Metallicity chemical model.
    
    See also
    --------
    :class:`MassPropMetallicityMixin`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ism_metallicity_today = kwargs["ism_metallicity_today"]
        self.alpha_powerlaw = kwargs["alpha_powerlaw"]


class ExponentialDelayedQuenchedCEM(ExponentialDelayedZPowerLawCEM):
    """A :class:`ExponentialDelayedZPowerLawCEM` with a quenching event."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quenching_time = check_unit(kwargs["quenching_time"], u.Gyr)
        mtoday = self.stellar_mass_formed(self.today)
        self._mass_norm *= self.mass_today / mtoday

    @sfh_quenching_decorator
    def stellar_mass_formed(self, times: u.Quantity):
        return super().stellar_mass_formed(times)

#-------------------------------------------------------------------------------
class GaussianBurstCEM(ChemicalEvolutionModel):
    """
    Gaussian burst star formation model.

    This model represents a galaxy's star formation history as a single Gaussian 
    burst of star formation. The burst occurs at a specific time and is characterized 
    by a Gaussian distribution with a given standard deviation.

    Attributes
    ----------
    mass_burst : float or astropy.Quantity
        Total stellar mass formed in the burst.
    time_burst : float or astropy.Quantity
        Time of the starburst in cosmic time.
    sigma_burst : float or astropy.Quantity
        Span time of the burst in terms of the standard deviation.
    burst_metallicity : float or astropy.Quantity
        Metallicity of the burst.
    
    Examples
    --------
    Create an instance of the `GaussianBurstCEM` model and compute the stellar mass formed:

    >>> from astropy import units as u
    >>> from pst.models import GaussianBurstCEM
    >>> model = GaussianBurstCEM(mass_burst=1e10 * u.Msun, time_burst=2 * u.Gyr,
    ...                          sigma_burst=0.5 * u.Gyr, burst_metallicity=0.02)
    >>> time = [1.0, 2.0, 3.0] * u.Gyr
    >>> model.stellar_mass_formed(time)
    <Quantity [2.27501319e+08, 5.00000000e+09, 9.77249868e+09] solMass>
    """
 
    def __init__(self, **kwargs):
        self.mass_burst = check_unit(kwargs["mass_burst"], u.Msun)
        self.time_burst = check_unit(kwargs["time_burst"], u.Gyr)
        self.sigma_burst = check_unit(kwargs["sigma_burst"], u.Gyr)
        self.burst_metallicity = kwargs["burst_metallicity"]
        ChemicalEvolutionModel.__init__(self, **kwargs)

    @u.quantity_input
    def stellar_mass_formed(self, time):
        return self.mass_burst / 2 * (1 + special.erf(
            (time-self.time_burst) / (SQRT_2 * self.sigma_burst))
            )
  
    @u.quantity_input
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.burst_metallicity)


class LogNormalCEM(ChemicalEvolutionModel):
    r"""
    Log-normal star formation history model.

    This CEM models a galaxy's star formation rate as a log-normal
    function of time (e.g. Gladders et al 2013), where the SFR rises initially
    and then decays:

    .. math::
        M_\star(t) = \frac{M_{\rm today}}{2} \cdot \left(1 - {\rm erf}\left(\frac{{\rm ln}(t) - {\rm ln}(t_0)}{\sigma \sqrt{2}}\right) \right)

    Attributes
    ----------
    lnt0 : float
        Mean of the lognormal SFH.
    tau : astropy.Quantity
        Standard deviation of the lognormal SFH.
    mass_today : float or astropy.Quantity
        Total stellar mass formed at present.
    metallicity : float
        Metallicity of the gas (constant).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t0 = check_unit(kwargs['t0'], u.Gyr)
        self.scale = kwargs['scale']
        self.today = check_unit(kwargs['today'], u.Gyr)
        self.mass_today = check_unit(kwargs['mass_today'], u.Msun)
        self.mass_norm = 1
        mtoday = self.stellar_mass_formed(self.today)
        self.mass_norm = self.mass_today / mtoday
        self.ism_metallicity_today =  kwargs['ism_metallicity_today']

    @u.quantity_input
    def stellar_mass_formed(self, times: u.Quantity):
        z = - np.log(times[times > 0] / self.t0) / self.scale
        m = np.zeros(times.shape)
        m[times > 0] = 0.5 * (1 - special.erf(z / SQRT_2))
        return m * self.mass_norm

    @u.quantity_input
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.ism_metallicity_today)


class LogNormalZPowerLawCEM(MassPropMetallicityMixin, LogNormalCEM):
    """A :class:`LogNormalCEM` with a Mass-dependent Metallicity chemical model.
    
    See also
    --------
    :class:`MassPropMetallicityMixin`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ism_metallicity_today = kwargs["ism_metallicity_today"]
        self.alpha_powerlaw = kwargs["alpha_powerlaw"]


class LogNormalQuenchedCEM(LogNormalZPowerLawCEM):
    """A :class:`LogNormalCEM` with a quenching event."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quenching_time = check_unit(kwargs["quenching_time"], u.Gyr)
        mtoday = self.stellar_mass_formed(self.today)
        self.mass_norm *= self.mass_today / mtoday

    @sfh_quenching_decorator
    def stellar_mass_formed(self, times: u.Quantity):
        return super().stellar_mass_formed(times)


#-------------------------------------------------------------------------------
class TabularCEM(ChemicalEvolutionModel):
    """Chemical evolution model based on a grid of times and metallicities.
    
    Description
    -----------
    This model represents the chemical evolution of a galaxy by means of a
    discrete grid of ages and metallicities

    Attributes
    ----------
    table_t: astropy.Quantity
        Tabulated cosmic time.
    table_M: astropy.Quantity
        Total stellar mass at each cosmic time step.
    table_metallicity: astropy.Quantity
        ISM metallicity at each cosmic time step.

    See also
    --------
    :class:`pst.models.ChemicalEvolutionModel` documentation.
    """
    def __init__(self, times, masses, metallicities, **kwargs):
        super().__init__(**kwargs)
        self.table_t = check_unit(times, u.Gyr)
        # Make sure that time is crescent
        sort_times = np.argsort(self.table_t)
        self.table_t = self.table_t[sort_times]
        self.table_mass = check_unit(masses[sort_times], u.Msun)
        self.table_metallicity = metallicities[sort_times]

    @u.quantity_input
    def stellar_mass_formed(self, times: u.Gyr):
        r"""Evaluate the integral of the SFR over a given set of times.
        
        Description
        -----------
        This method evaluates the integral:

        .. math::
            \int_{0}^{t} {\rm SFR}(t') dt'

        at each time input time :math:`t`.

        Parameters
        ----------
        times : :class:`astropy.units.Quantity`
            Array of cosmic times at which the integral will be evaluated.

        Returns
        -------
        integral : :class:`astropy.units.Quantity`
            The cumulative stellar mass formed at each input time.
        """
        interpolator = interpolate.PchipInterpolator(
           self.table_t, self.table_mass)
        integral = interpolator(times) << self.table_mass.unit
        integral[times > self.table_t[-1]] = self.table_mass[-1]
        integral[times < self.table_t[0]] = 0
        return integral
    
    @u.quantity_input
    def ism_metallicity(self, times: u.Gyr):
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
        interpolator = interpolate.PchipInterpolator(
           self.table_t, self.table_metallicity)
        integral = interpolator(times)
        integral[times > self.table_t[-1]] = self.table_metallicity[-1]
        integral[times < self.table_t[0]] = self.table_metallicity[0]
        return integral


class TabularCEM_ZPowerLaw(MassPropMetallicityMixin, TabularCEM):
    """Chemical evolution model based on a grid of times that assumes an analytic chemical enrichment history.

    See Also
    --------
    :class:`TabularCEM`
    :class:`MassPropMetallicityMixin`

    """
    def __init__(self, times, masses, alpha_powerlaw, ism_metallicity_today,
                 mass_today, **kwargs):
        self.ism_metallicity_today = ism_metallicity_today
        self.alpha_powerlaw = alpha_powerlaw
        self.mass_today = mass_today
        # Create a dummy metallicity that is passed to the TabularCEM constructor
        # but never used
        metallicity = np.zeros(times.size)
        super().__init__(times, masses, metallicity, **kwargs)


class ParticleListCEM(ChemicalEvolutionModel):
    """
    Chemical Evolution Model using individual Simple Stellar Population (SSP) data.
    
    This model represents the chemical evolution of a galaxy by reconstructing 
    a composite stellar population (CSP) using individual stellar population (SSP) 
    particles, each of which is defined by its formation time, metallicity, and mass.

    Parameters
    ----------
    time_form : numpy.array or astropy.units.Quantity
        Array representing the formation times of each SSP particle. If the input 
        is a `numpy.array`, it is assumed to be in Gyr. Otherwise, an `astropy.units.Quantity` 
        with appropriate units is required.
    metallicities : numpy.array or astropy.units.Quantity
        Array representing the metallicities of each SSP particle. If the input 
        is a `numpy.array`, it is assumed to be dimensionless (i.e., no units).
    masses : numpy.array or astropy.units.Quantity
        Array representing the masses of each SSP particle. If the input is a 
        `numpy.array`, it is assumed to be in solar masses.
    """
    def __init__(self, time_form, metallicities, masses):
        self.time_form, self.metallicities, self.masses = (
            time_form, metallicities, masses)

    @property
    def time_form(self) -> u.Quantity:
        """Array of SSP formation times in Gyr."""
        return self._time_form

    @time_form.setter
    def time_form(self, values):
        if not isinstance(values, u.Quantity):
            self._time_form = values << u.Gyr
        else:
            self._time_form = values

    @property
    def metallicities(self) -> u.Quantity:
        """Array of SSP metallicities, assumed to be dimensionless."""
        return self._metallicities

    @metallicities.setter
    def metallicities(self, values):
        if not isinstance(values, u.Quantity):
            self._metallicities = values << u.dimensionless_unscaled
        else:
            self._metallicities = values

    @property
    def masses(self) -> u.Quantity:
        """Array of SSP masses in solar masses."""
        return self._masses

    @masses.setter
    def masses(self, values):
        if not isinstance(values, u.Quantity):
            self._masses = values << u.Msun
        else:
            self._masses = values

    def interpolate_ssp_masses(self, ssp: SSPBase, t_obs: u.Quantity):
        """
        Interpolate the SSP particles onto an SSP base model at the observed time.
        
        This method interpolates the stellar masses of the SSP particles based on 
        their ages and metallicities at the observed cosmic time `t_obs` using an 
        SSP model grid.

        Parameters
        ----------
        ssp : pst.SSP.SSPBase
            SSP model providing the ages and metallicities for interpolation.
        t_obs : astropy.units.Quantity
            The age of the Universe at the time of the observation.

        Returns
        -------
        ssp_weights : astropy.units.Quantity
            A 2D array representing the stellar mass associated with each SSP 
            particle in the base SSP grid. Units are in solar masses.
        """
        valid_particles = self.time_form <= t_obs
        return ssp.get_weights(ages=t_obs - self.time_form[valid_particles],
                               metallicities=self.metallicities[valid_particles],
                               masses=self.masses[valid_particles])

    def stellar_mass_formed(self, time):
        sort_idx = np.argsort(self.time_form)
        mass_history = np.cumsum(self.masses[sort_idx])
        return np.interp(time, self.time_form[sort_idx], mass_history)

    def ism_metallicity(self, time):
        return np.full(time.size, fill_value=np.nan)

# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/
# -----------------------------------------------------------------------------

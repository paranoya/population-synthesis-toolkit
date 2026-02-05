from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
from astropy import units as u
from scipy import special
from scipy import interpolate

from pst.SSP import SSPBase
from pst.model import ModelBase, Parameter
from pst.utils import SQRT_2, check_parameter, check_unit

# ---------------------------------------------------------------------
# CEM utils and mixins
# ---------------------------------------------------------------------
def _check_time_dec(func):
    """
    Decorator that normalizes a ``time`` argument to an ``astropy.units.Quantity``.

    The wrapped method may be called with:

    - a bare number / numpy array (assumed to be in Gyr),
    - an :class:`astropy.units.Quantity` convertible to Gyr,
    - a :class:`pst.model.Parameter` whose ``.q`` is convertible to Gyr.

    The wrapped function is always called with a Quantity in Gyr.

    Notes
    -----
    This decorator only converts the *first* positional argument after ``self``
    (conventionally named ``time`` or ``times``). Any additional positional
    and keyword arguments are passed through unchanged.
    """
    @wraps(func)
    def wrapper(self, time, *args, **kwargs):
        if isinstance(time, Parameter):
            tq = time.q
        else:
            tq = check_unit(time, u.Gyr)
        return func(self, tq, *args, **kwargs)
    return wrapper

class MassPropMetallicityMixin:
    r"""
    Mixin implementing a mass-dependent metallicity evolution.

    This mixin provides a default implementation of
    :meth:`~pst.cem.ChemicalEvolutionModel.ism_metallicity` where the ISM
    metallicity scales with the cumulative stellar mass formed:

    .. math::
        Z(t) = Z_{\rm today}\,\left(\frac{M_\star(t)}{M_\star({\rm today})}\right)^\alpha

    The intention is to allow any star-formation-history (SFH) model to be
    combined with a simple analytic chemical enrichment prescription.

    Required host attributes
    ------------------------
    The host class must define:

    - ``mass_today`` : :class:`~astropy.units.Quantity` or :class:`pst.model.Parameter`
        Total stellar mass formed at the observing time (same units as
        :meth:`stellar_mass_formed`).
    - ``ism_metallicity_today`` : Quantity or Parameter
        Present-day ISM metallicity (dimensionless mass fraction).
    - ``alpha_powerlaw`` : Quantity/Parameter or float
        Power-law exponent controlling the enrichment rate.
    - :meth:`stellar_mass_formed(times)`
        Cumulative stellar mass formed as a function of cosmic time.

    Notes
    -----
    - The returned metallicity is dimensionless (mass fraction).
    - For physical results, host models should ensure
      :meth:`stellar_mass_formed` is non-decreasing and non-negative.
    """
    @property
    def ism_metallicity_today(self) -> Parameter:
        """ISM metals mass fraction at present."""
        return self._ism_metallicity_today
    
    @ism_metallicity_today.setter
    def ism_metallicity_today(self, value: Parameter):
        self._ism_metallicity_today = value
    
    @property
    def alpha_powerlaw(self) -> Parameter:
        """Stellar mass power-law exponent."""
        return self._alpha_powerlaw

    @alpha_powerlaw.setter
    def alpha_powerlaw(self, value: Parameter):
        self._alpha_powerlaw = value

    def ism_metallicity(self, times: Union[np.array, u.Quantity]) -> u.Quantity:
        """
        Evaluate ISM metallicity at one or more cosmic times.

        Parameters
        ----------
        times : array-like or :class:`astropy.units.Quantity`
            Cosmic times since the Big Bang. If a bare array is provided, it is
            interpreted as Gyr.

        Returns
        -------
        z_t : :class:`astropy.units.Quantity`
            Dimensionless ISM metallicity (mass fraction) evaluated at each input
            time. Shape matches ``times``.
        """
        m = self.stellar_mass_formed(times)
        return self.ism_metallicity_today * np.power(m / self.mass_today, self.alpha_powerlaw)

def sfh_quenching_decorator(stellar_mass_formed):
    """
    Decorator that enforces a hard quenching event in a cumulative SFH.

    The decorated :meth:`stellar_mass_formed` behaves normally for
    ``time < quenching_time``. For later times it returns the constant value
    :math:`M_\\star(\\mathrm{quenching\\_time})`, i.e. no further stellar mass
    is formed.

    The host instance is expected to provide an attribute ``quenching_time``,
    which may be a :class:`pst.model.Parameter`, an
    :class:`astropy.units.Quantity`, or a bare number (assumed Gyr).

    Notes
    -----
    - This decorator assumes the wrapped function returns the *cumulative*
      stellar mass formed.
    - The quenching is instantaneous and permanent (a step in the cumulative SFH).
    """
    @wraps(stellar_mass_formed)
    def wrapper(self, time, *args, **kwargs):
        tq = time.q if isinstance(time, Parameter) else check_unit(time, u.Gyr)
        qt = getattr(self, "quenching_time", None)
        if qt is None:
            return stellar_mass_formed(self, tq, *args, **kwargs)
        qtq = qt.q if isinstance(qt, Parameter) else check_unit(qt, u.Gyr)

        m = stellar_mass_formed(self, tq, *args, **kwargs)
        m_q = stellar_mass_formed(self, qtq, *args, **kwargs)
        return np.where(tq < qtq, m, m_q)
    return wrapper

@dataclass
class ChemicalEvolutionModel(ModelBase, ABC):
    """
    Base class for chemical evolution models (CEMs).

    A Chemical Evolution Model defines two core functions of cosmic time:

    - :meth:`stellar_mass_formed(t)`: cumulative stellar mass formed up to time ``t``
    - :meth:`ism_metallicity(t)`: ISM metallicity (mass fraction) at time ``t``

    The class also provides utilities to synthesize spectra and photometry from
    a :class:`pst.SSP.SSPBase` grid by integrating the SFH over SSP age bins.

    Time convention
    ---------------
    All "time" arguments are **cosmic time since the Big Bang**, not lookback time.
    The optional attribute ``today`` represents the cosmic age of the Universe
    at the observing epoch and is used by helper methods that require a
    normalization point (e.g. formation-time percentiles).

    Parameters
    ----------
    today : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`, optional
        Cosmic time of observation. If a bare number is provided it is assumed
        to be in Gyr.

    Notes
    -----
    Implementations should strive to ensure:

    - :meth:`stellar_mass_formed` is non-decreasing with time,
    - :meth:`stellar_mass_formed(t < 0) = 0`,
    - behavior for ``t > today`` is well-defined (commonly clipped to
      :math:`M_\\star(\\mathrm{today})`).
    """

    name: str = field(default="base_cem", init=False)
    today: Optional[Parameter | u.Quantity | float] = None

    def __post_init__(self):
        if self.today is not None:
            self.today = check_parameter(self.today, u.Gyr,
                                         doc="Age of the Universe at observation")

    @abstractmethod
    def stellar_mass_formed(self, time: u.Quantity) -> u.Quantity:
        """
        Cumulative stellar mass formed as a function of cosmic time.

        Parameters
        ----------
        time : :class:`astropy.units.Quantity`
            Cosmic time(s) since the Big Bang.

        Returns
        -------
        mstar : :class:`astropy.units.Quantity`
            Cumulative stellar mass formed up to each input time. Units should be
            mass (typically Msun). Shape matches ``time``.
        """
        return

    @abstractmethod
    def ism_metallicity(self, time: u.Quantity) -> u.Quantity:
        """
        ISM metallicity (mass fraction) as a function of cosmic time.

        Parameters
        ----------
        time : :class:`astropy.units.Quantity`
            Cosmic time(s) since the Big Bang.

        Returns
        -------
        z_t : :class:`astropy.units.Quantity`
            Dimensionless metallicity (mass fraction). Shape matches ``time``.
        """
        return

    def interpolate_ssp_masses(self, ssp: SSPBase, t_obs: u.Gyr,
                               oversample_factor=10) -> u.Quantity:
        """
        Compute SSP mass weights by integrating the SFH over SSP age bins.

        This routine discretizes the interval ``[0, t_obs]`` into SSP age bins and
        computes the stellar mass formed in each bin via differences of the cumulative
        SFH, then maps each bin to the SSP grid using :meth:`pst.SSP.SSPBase.get_weights`.

        Parameters
        ----------
        ssp : :class:`pst.SSP.SSPBase`
            SSP model providing an age grid ``ssp.ages`` (shape ``(A,)``),
            metallicity grid, and a mapping function ``get_weights``.
        t_obs : :class:`astropy.units.Quantity`
            Cosmic time of observation. Only SSP ages younger than ``t_obs`` contribute.
        oversample_factor : int, optional
            Factor by which SSP age bins are oversampled for smoother integration.
            Default is 10.

        Returns
        -------
        weights : :class:`astropy.units.Quantity`
            Mass weights on the SSP grid. The expected shape is ``(Z, A)``
            (metallicity, age). Units are mass (typically Msun).

        Notes
        -----
        The method uses midpoints of SSP age bins and evaluates
        :meth:`stellar_mass_formed` and :meth:`ism_metallicity` at corresponding
        cosmic times (``t_obs - age``).
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
        Time at which a fraction of the final stellar mass has formed.

        This function samples the cumulative SFH on a regular cosmic-time grid and
        linearly interpolates the time(s) at which:

        .. math::
            \\frac{M_\\star(t)}{M_\\star(\\mathrm{today})} = f

        Parameters
        ----------
        frac : float or array-like
            Target mass fraction(s) in the range [0, 1].
        today : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`, optional
            Cosmic observing time. If not provided, uses ``self.today``.
        time_res : float or :class:`astropy.units.Quantity`, optional
            Sampling resolution of the time grid. If a bare number is provided it is
            assumed to be in Gyr. Default is 30 Myr.

        Returns
        -------
        t_frac : :class:`astropy.units.Quantity`
            Cosmic time(s) at which the specified fraction(s) are reached.
            Shape matches ``frac``.

        Raises
        ------
        ValueError
            If neither ``today`` nor ``self.today`` is set.
        """
        frac = np.atleast_1d(frac)
        if today is None:
            if self.today is None:
                raise ValueError("Attribute today is not set; provide today.")
            today = self.today.q
        else:
            today = check_unit(today.q if isinstance(today, Parameter) else today, u.Gyr)

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

    def _age_bin_matrix(self, idx: np.ndarray, nbin: int) -> np.ndarray:
        # idx: (A,)
        a = np.arange(idx.size)[:, None]
        b = np.arange(nbin)[None, :]
        M = (idx[:, None] == b).astype(float)  # (A, Nbin)

        # zero out invalid ages outside [edges[0], edges[-1])
        valid = (idx >= 0) & (idx < nbin)
        M *= valid[:, None]
        return M

    def compute_SED(self, ssp : SSPBase, t_obs : u.Quantity, *,
                    allow_negative: bool=False, age_bin_edges=None) -> u.Quantity:
        """
        Synthesize the spectral energy distribution (SED) at an observing time.

        The SED is obtained by summing SSP spectra weighted by the stellar mass formed
        in each SSP age/metallicity bin.

        Parameters
        ----------
        ssp : :class:`pst.SSP.SSPBase`
            SSP model providing ``ssp.L_lambda`` with shape ``(Z, A, W)`` where
            ``W`` is wavelength.
        t_obs : :class:`astropy.units.Quantity`
            Cosmic time of observation.
        allow_negative : bool, optional
            If False (default), any negative SSP weights are clipped to zero.
        age_bin_edges : array-like or :class:`astropy.units.Quantity`, optional
            If provided, the SED is returned in coarse age bins. Edges are in the
            same units as ``ssp.ages`` and define ``Nbin = len(edges) - 1`` bins.

        Returns
        -------
        sed : :class:`astropy.units.Quantity`
            If ``age_bin_edges`` is None: shape ``(W,)``.
            If ``age_bin_edges`` is provided: shape ``(Nbin, W)``.
            Units are ``(mass unit) * (ssp.L_lambda unit)``.

        Notes
        -----
        ``age_bin_edges`` bins SSP *ages* (not cosmic times). Internally the
        mapping is performed using :func:`numpy.digitize` on ``ssp.ages``.
        """
        weights = self.interpolate_ssp_masses(ssp, t_obs)
        weights = np.where(np.isfinite(weights), weights, 0.0 << weights.unit)
        if not allow_negative:
            weights = np.where(weights > 0, weights, 0.0 << weights.unit)
        
        if age_bin_edges is None:
            sed_val = np.einsum("za,zaw->w", weights.value, ssp.L_lambda.value)
            return sed_val * (weights.unit * ssp.L_lambda.unit)

        idx = np.digitize(ssp.ages, check_unit(age_bin_edges, ssp.ages.unit)
                          ).clip(0, len(age_bin_edges) - 1) - 1
        M = self._age_bin_matrix(idx, len(age_bin_edges) - 1)
        sed_val = np.einsum("za,zaw,an->nw", weights.value, ssp.L_lambda.value, M)
        return sed_val * (weights.unit * ssp.L_lambda.unit)

    def compute_photometry(self, ssp, t_obs, *,
                           photometry: u.Quantity=None, allow_negative: bool=False,
                           age_bin_edges=None) -> u.Quantity:
        """
        Synthesize broadband photometry at an observing time.

        Parameters
        ----------
        ssp : :class:`pst.SSP.SSPBase`
            SSP model defining the SSP grid used to compute weights.
        t_obs : :class:`astropy.units.Quantity`
            Cosmic time of observation.
        photometry : :class:`astropy.units.Quantity` or ndarray, optional
            Photometry grid on the SSP axes with shape ``(B, Z, A)``, where
            ``B`` is the number of bands. If an ndarray is provided, it is assumed
            to be in ``Jy / Msun``.
            If None, uses ``ssp.photometry``.
        allow_negative : bool, optional
            If False (default), negative SSP weights are clipped to zero.
        age_bin_edges : array-like or :class:`astropy.units.Quantity`, optional
            If provided, photometry is returned per coarse age bin defined over
            SSP ages. Output will have ``Nbin = len(edges) - 1`` bins.

        Returns
        -------
        model_photometry : :class:`astropy.units.Quantity`
            If ``age_bin_edges`` is None: shape ``(B,)``.
            If ``age_bin_edges`` is provided: shape ``(Nbin, B)``.
            Units are ``photometry.unit * weights.unit``.

        Notes
        -----
        This method assumes the photometry grid is expressed per unit formed mass
        (e.g. ``Jy / Msun``). The output is therefore an absolute flux/luminosity
        in the same system as the input grid.
        """
        weights = self.interpolate_ssp_masses(ssp, t_obs)
        weights = np.where(np.isfinite(weights), weights, 0.0 << weights.unit)
        if not allow_negative:
            weights = np.where(weights > 0, weights, 0.0 << weights.unit)

        if photometry is None:
            photometry = ssp.photometry
        if not isinstance(photometry, u.Quantity):
            print("Assuming input photometry array in Jy/Msun")
            photometry = photometry << u.Jy / u.Msun

        if age_bin_edges is None:
            # (band, z, age) * (z, age) -> band
            out_val = np.einsum("bza,za->b", photometry.value, weights.value)
            return out_val * (photometry.unit * weights.unit)

        idx = np.digitize(ssp.ages, check_unit(age_bin_edges, ssp.ages.unit)
                          ).clip(0, len(age_bin_edges) - 1) - 1
        M = self._age_bin_matrix(idx, len(age_bin_edges) - 1)
        out_val = np.einsum("bza,za,an->nb", photometry.value, weights.value, M)
        return out_val * (photometry.unit * weights.unit)


#-------------------------------------------------------------------------------
@dataclass
class SingleBurstCEM(ChemicalEvolutionModel):
    """
    Instantaneous single-burst star formation history.

    The model forms a total mass ``mass_burst`` at cosmic time ``time_burst`` and
    forms no additional stars thereafter. The cumulative formed mass is:

    - 0 for ``t < time_burst``
    - ``mass_burst`` for ``t >= time_burst``

    Parameters
    ----------
    mass_burst : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`, optional
        Total stellar mass formed in the burst. Bare numbers are assumed in Msun.
    time_burst : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`, optional
        Cosmic time of the burst. Bare numbers are assumed in Gyr.
    burst_metallicity : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`, optional
        Constant ISM metallicity assigned to all stars (dimensionless mass fraction).

    Notes
    -----
    This is a cumulative SFH model; the instantaneous SFR is a delta function.
    """
    mass_burst: Parameter | u.Quantity | float = 1 << u.Msun
    time_burst: Parameter | u.Quantity | float = 0 << u.Gyr
    burst_metallicity: Parameter | u.Quantity | float = 0.02 << u.dimensionless_unscaled
    name: str = field(default="single_burst_cem", init=False)


    def __post_init__(self):
        super().__post_init__()
        self.mass_burst = check_parameter(self.mass_burst, u.Msun,
                                          doc="Total stellar mass in the burst")
        self.time_burst = check_parameter(self.time_burst, u.Gyr,
                                          doc="Cosmic time of the burst")
        self.burst_metallicity = check_parameter(self.burst_metallicity,
                                                 u.dimensionless_unscaled,
                                                 doc="Burst metallicity")

    @_check_time_dec
    def stellar_mass_formed(self, time: u.Gyr) -> u.Quantity:
        """Total stellar mass formed at a given time."""
        mass = np.zeros(time.size, dtype=float) * self.mass_burst.q.unit
        mass[time >= self.time_burst] = self.mass_burst.q
        return mass

    @_check_time_dec
    def ism_metallicity(self, time: u.Gyr) -> u.Quantity:
        """ISM metals mass fraction at a given time."""
        return np.full(time.size,
                       fill_value=self.burst_metallicity.q.value) << self.burst_metallicity.q.unit


#-------------------------------------------------------------------------------
@dataclass
class ExponentialCEM(ChemicalEvolutionModel):
    r"""
    Exponentially-declining cumulative SFH.

    The cumulative formed mass follows:

    .. math::
        M_\star(t) = M_{\infty}\,\left(1 - e^{-t/\tau}\right)

    where ``M_infty`` is the asymptotic formed mass and ``tau`` is the
    e-folding time.

    Parameters
    ----------
    stellar_mass_inf : float, Quantity, or Parameter, optional
        Asymptotic formed stellar mass ``M_infty``. Bare numbers assumed in Msun.
    tau : float, Quantity, or Parameter, optional
        E-folding time in Gyr.
    metallicity : float, Quantity, or Parameter, optional
        Constant ISM metallicity (dimensionless mass fraction).

    Notes
    -----
    The metallicity returned by :meth:`ism_metallicity` is constant in time.
    """
    stellar_mass_inf: Parameter = 1 << u.Msun
    tau: Parameter = 1 << u.Gyr
    metallicity: Parameter = 0.02 << u.dimensionless_unscaled
    today: Parameter | None = None

    name: str = field(default="exponential_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.stellar_mass_inf = check_parameter(self.stellar_mass_inf, u.Msun,
                                                doc="Total stellar mass at infinity")
        self.tau = check_parameter(self.tau, u.Gyr,
                                   doc="E-folding SFH timescale")
        self.metallicity = check_parameter(self.metallicity,
                                           u.dimensionless_unscaled,
                                           doc="Burst metallicity")

    @_check_time_dec
    def stellar_mass_formed(self, time: u.Gyr) -> u.Quantity:
      return self.stellar_mass_inf * ( 1 - np.exp(-time/self.tau) )

    @_check_time_dec
    def ism_metallicity(self, time: u.Gyr) -> u.Quantity:
        return np.full(time.size, fill_value=self.metallicity.q)


@dataclass
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
    quenching_time: Parameter | u.Quantity | float = None
    name: str = field(default="exponential_quenched_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.quenching_time = check_parameter(self.quenching_time, u.Gyr,
                                                doc="Cosmic quenching time")

    @_check_time_dec
    @sfh_quenching_decorator
    def stellar_mass_formed(self, time : u.Gyr):
        return super().stellar_mass_formed(time)


#-------------------------------------------------------------------------------
@dataclass
class ExponentialDelayedCEM(ChemicalEvolutionModel):
    r"""
    Delayed-exponential cumulative SFH normalized at ``today``.

    The cumulative formed mass is:

    .. math::
        M_\star(t) \propto 1 - e^{-t/\tau}\,\frac{t+\tau}{\tau}

    This class rescales the proportional form so that
    :math:`M_\star(\\mathrm{today}) = \\mathrm{mass\\_today}`.

    Parameters
    ----------
    tau : float, Quantity, or Parameter, optional
        Timescale parameter (Gyr).
    mass_today : float, Quantity, or Parameter, optional
        Total formed stellar mass at ``today`` (Msun).
    ism_metallicity_today : float, Quantity, or Parameter, optional
        Constant metallicity (dimensionless mass fraction).
    today : float, Quantity, or Parameter
        Cosmic observing time required for normalization.

    Raises
    ------
    ValueError
        If ``today`` is not set.
    """

    tau: Parameter | u.Quantity | float = 2.0 * u.Gyr
    mass_today: Parameter | u.Quantity | float = 1.0 * u.Msun
    ism_metallicity_today: Parameter | u.Quantity | float = 0.02

    name: str = field(default="exponential_delayed_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.mass_today = check_parameter(self.mass_today, u.Msun,
                                          doc="Total stellar mass at observing time")
        self.tau = check_parameter(self.tau, u.Gyr,
                                   doc="E-folding SFH timescale")
        self.ism_metallicity_today = check_parameter(self.ism_metallicity_today,
                                           u.dimensionless_unscaled,
                                           doc="Constant stellar metallicity")

        self._mass_norm = 1
        if self.today is None:
            raise ValueError("Parameter ``today`` must be set for ExponentialDelayedCEM")
        mtoday = self.stellar_mass_formed(self.today)
        self._mass_norm = self.mass_today / mtoday

    @_check_time_dec
    def stellar_mass_formed(self, time):
        return self.mass_today * (1 - np.exp(-time / self.tau)
            * (self.tau + time) / self.tau) * self._mass_norm

    @_check_time_dec
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.ism_metallicity_today.q)

@dataclass
class ExponentialDelayedZPowerLawCEM(MassPropMetallicityMixin, ExponentialDelayedCEM):
    """A :class:`ExponentialDelayedCEM` with a Mass-dependent Metallicity chemical model.
    
    See also
    --------
    :class:`MassPropMetallicityMixin`
    """

    ism_metallicity_today: Parameter = 0.02 << u.dimensionless_unscaled
    alpha_powerlaw: Parameter = 1 << u.dimensionless_unscaled

    name: str = field(default="exponential_delayed_zpowlaw_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.ism_metallicity_today = check_parameter(self.ism_metallicity_today,
                                                     u.dimensionless_unscaled,
                                                     doc="Metallicity of stars born at present")
        self.alpha_powerlaw = check_parameter(self.alpha_powerlaw,
                                              u.dimensionless_unscaled,
                                              doc="Metallicity evolution power-law exponent")    


@dataclass
class ExponentialDelayedQuenchedCEM(ExponentialDelayedZPowerLawCEM):
    """A :class:`ExponentialDelayedZPowerLawCEM` with a quenching event."""

    quenching_time: Parameter | u.Quantity | float = None

    name: str = field(default="exponential_quenched_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.quenching_time = check_parameter(self.quenching_time, u.Gyr,
                                                doc="Cosmic quenching time")
        mtoday = self.stellar_mass_formed(self.today)
        self._mass_norm *= self.mass_today / mtoday

    @_check_time_dec
    @sfh_quenching_decorator
    def stellar_mass_formed(self, times: u.Quantity):
        return super().stellar_mass_formed(times)

#-------------------------------------------------------------------------------

@dataclass(kw_only=True)
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
    mass_burst: Parameter | u.Quantity | float = None
    time_burst: Parameter | u.Quantity | float = None
    sigma_burst: Parameter | u.Quantity | float = None
    burst_metallicity: Parameter | u.Quantity | float = None
    name: str = field(default="gaussian_burst_cem", init=False)

    def __post_init__(self):
        super().__post_init__()

        self.mass_burst = check_parameter(self.mass_burst, u.Msun,
                                          doc="Total stellar mass formed during the burst")
        self.time_burst = check_parameter(self.time_burst, u.Gyr,
                                          doc="Burst cosmic time")
        self.sigma_burst = check_parameter(self.sigma_burst, u.Gyr,
                                           doc="Burst duration (Gaussian sigma)")
        self.burst_metallicity = check_parameter(self.burst_metallicity, u.dimensionless_unscaled,
                                                  doc="Burst stellar metallicity")

    @_check_time_dec
    def stellar_mass_formed(self, time):
        return self.mass_burst.q / 2 * (1 + special.erf(
            (time-self.time_burst.q) / (SQRT_2 * self.sigma_burst.q))
            )
  
    @_check_time_dec
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.burst_metallicity.q)


@dataclass
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
    t0: Parameter | u.Quantity | float = 1 << u.Gyr
    scale: Parameter | u.Quantity | float = 1 << u.dimensionless_unscaled
    mass_today: Parameter | u.Quantity | float = 1 << u.Msun
    ism_metallicity_today: Parameter | u.Quantity | float = 0.02 << u.dimensionless_unscaled

    name: str = field(default="lognormal_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        # do the checks and add default docs
        self.t0 = check_parameter(self.t0, u.Gyr)
        self.scale = check_parameter(self.scale, u.dimensionless_unscaled)
        self.mass_today = check_parameter(self.mass_today, u.Msun)
        self.ism_metallicity_today = check_parameter(self.ism_metallicity_today, u.dimensionless_unscaled)

        if self.today is None:
            raise ValueError("Parameter ``today`` must be set")
        # re-scale mass normalization
        self.mass_norm = 1
        mtoday = self.stellar_mass_formed(self.today)
        self.mass_norm = self.mass_today / mtoday

    @_check_time_dec
    def stellar_mass_formed(self, times: u.Quantity):
        z = - np.log(times[times > 0] / self.t0) / self.scale
        m = np.zeros(times.shape)
        m[times > 0] = 0.5 * (1 - special.erf(z / SQRT_2))
        return m * self.mass_norm

    @_check_time_dec
    def ism_metallicity(self, time : u.Gyr):
        return np.full(time.size, fill_value=self.ism_metallicity_today.q)


@dataclass
class LogNormalZPowerLawCEM(MassPropMetallicityMixin, LogNormalCEM):
    """A :class:`LogNormalCEM` with a Mass-dependent Metallicity chemical model.
    
    See also
    --------
    :class:`MassPropMetallicityMixin`
    """

    ism_metallicity_today: Parameter = 0.02 << u.dimensionless_unscaled
    alpha_powerlaw: Parameter = 1 << u.dimensionless_unscaled

    name: str = field(default="lognormal_zpowlaw_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.ism_metallicity_today = check_parameter(self.ism_metallicity_today,
                                                     u.dimensionless_unscaled,
                                                     doc="Metallicity of stars born at present")
        self.alpha_powerlaw = check_parameter(self.alpha_powerlaw,
                                              u.dimensionless_unscaled,
                                              doc="Metallicity evolution power-law exponent")    


@dataclass
class LogNormalQuenchedCEM(LogNormalZPowerLawCEM):
    """A :class:`LogNormalCEM` with a quenching event."""
    quenching_time: Parameter | u.Quantity | float = None
    name: str = field(default="lognormal_quenched_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.quenching_time = check_parameter(self.quenching_time, u.Gyr,
                                                doc="Cosmic quenching time")
        mtoday = self.stellar_mass_formed(self.today)
        self.mass_norm *= self.mass_today / mtoday

    @_check_time_dec
    @sfh_quenching_decorator
    def stellar_mass_formed(self, times: u.Quantity):
        return super().stellar_mass_formed(times)


@dataclass
class BetaCEM(ChemicalEvolutionModel):
    r"""
    Beta-CMF chemical evolution model.

    This model defines the *cumulative formed* stellar mass using the regularised
    incomplete Beta function:

    .. math::
        f_\star(t) = I_x(\alpha, \beta), \qquad
        x = \frac{t - t_{\rm start}}{t_{\rm end} - t_{\rm start}} \in [0,1]

    and

    .. math::
        M_{\rm formed}(t) = M_{\rm today}\, f_\star(t)

    where :math:`I_x` is the regularised incomplete Beta function.

    Parameterisation
    ----------------
    You can specify the Beta shape parameters directly (``alpha``, ``beta``),
    or provide the mean formation time and concentration (``t_mean``, ``kappa``):

    - ``kappa = alpha + beta``
    - ``m = t_mean / t_end`` (fractional mean in (0,1))
    - ``alpha = m * kappa``, ``beta = (1-m) * kappa``

    Parameters
    ----------
    mass_today : float, :class:`astropy.units.Quantity`, or :class:`pst.model.Parameter`
        Total formed stellar mass by ``t_end``. Bare numbers are assumed in Msun.
    alpha, beta : float, Quantity, or Parameter, optional
        Beta-CMF shape parameters (>0). Provide both if using the direct form.
        If provided as Quantity/Parameter they must be dimensionless.
    t_mean : float, Quantity, or Parameter, optional
        Mean formation time. If a Quantity/Parameter, interpreted as cosmic time (Gyr).
        If a bare float, interpreted as fractional mean ``m`` in (0,1).
        Must be provided together with ``kappa``.
    kappa : float, Quantity, or Parameter, optional
        Concentration parameter (>0). Dimensionless. Used with ``t_mean``.
    t_start : float, Quantity, or Parameter, optional
        Cosmic time at which star formation begins. Default is 0 Gyr.
    t_end : float, Quantity, or Parameter, optional
        Cosmic time at which the model stops. If None, uses ``today``.
        Must satisfy ``t_end > t_start``.
    ism_metallicity_today : float, Quantity, or Parameter, optional
        Constant ISM metallicity (dimensionless mass fraction). Default 0.02.

    Notes
    -----
    - All time arguments are **cosmic time since the Big Bang** (not lookback).
    - :meth:`ism_metallicity` returns a constant metallicity (no enrichment model).
    """
    mass_today: Parameter | u.Quantity | float = 1.0 << u.Msun

    # Either set (alpha, beta) or set (t_mean, kappa)
    alpha: Optional[Parameter | u.Quantity | float] = None
    beta: Optional[Parameter | u.Quantity | float] = None
    t_mean: Optional[Parameter | u.Quantity | float] = None
    kappa: Optional[Parameter | u.Quantity | float] = None

    t_start: Parameter | u.Quantity | float = 0.0 << u.Gyr
    t_end: Optional[Parameter | u.Quantity | float] = None

    ism_metallicity_today: Parameter | u.Quantity | float = 0.02 << u.dimensionless_unscaled

    name: str = field(default="beta_cem", init=False)

    def __post_init__(self):
        super().__post_init__()

        self.mass_today = check_parameter(self.mass_today, u.Msun,
                                          doc="Total formed stellar mass by t_end")
        self.t_start = check_parameter(self.t_start, u.Gyr,
                                       doc="Cosmic start time of star formation")
        if self.t_end is not None:
            self.t_end = check_parameter(self.t_end, u.Gyr,
                                         doc="Cosmic end time of star formation horizon")

        self.ism_metallicity_today = check_parameter(
            self.ism_metallicity_today, u.dimensionless_unscaled,
            doc="Constant ISM metallicity (mass fraction)"
        )

        # Resolve t_end (needs today if not provided)
        _ = self._t_end_q  # triggers validation + today requirement

        # Resolve alpha/beta from either (alpha,beta) or (t_mean,kappa)
        self._resolve_shape_params()

    # -----------------------------------------------------------------
    # Internal helpers / validated quantities
    # -----------------------------------------------------------------
    @property
    def _t_end_q(self) -> u.Quantity:
        """Return t_end as a Quantity (falls back to today)."""
        if self.t_end is not None:
            te = self.t_end.q if isinstance(self.t_end, Parameter) else check_unit(self.t_end, u.Gyr)
        else:
            if self.today is None:
                raise ValueError("t_end is not set and `today` is None; set one of them.")
            te = self.today.q if isinstance(self.today, Parameter) else check_unit(self.today, u.Gyr)

        ts = self.t_start.q if isinstance(self.t_start, Parameter) else check_unit(self.t_start, u.Gyr)
        if not np.all(te > ts):
            raise ValueError(f"Require t_end > t_start, got t_end={te}, t_start={ts}.")
        return te

    def _resolve_shape_params(self) -> None:
        """
        Set internal dimensionless Parameters _alpha/_beta from either:
        - direct alpha/beta, or
        - mean+concentration (t_mean,kappa)
        """
        if (self.t_mean is not None) or (self.kappa is not None):
            if (self.t_mean is None) or (self.kappa is None):
                raise ValueError("Provide both `t_mean` and `kappa` (or neither).")
            a, b = self._alpha_beta_from_mean_concentration(self.t_mean, self.kappa)
            self._alpha = a
            self._beta = b
            return

        # direct alpha/beta
        if (self.alpha is None) or (self.beta is None):
            raise ValueError("Provide either (alpha,beta) or (t_mean,kappa).")

        a = check_parameter(self.alpha, u.dimensionless_unscaled, doc="Beta-CMF alpha").q.value
        b = check_parameter(self.beta, u.dimensionless_unscaled, doc="Beta-CMF beta").q.value
        if a <= 0 or b <= 0:
            raise ValueError("Beta-CMF shape parameters alpha and beta must be > 0.")

        self._alpha = float(a)
        self._beta = float(b)

    @property
    def alpha_val(self) -> float:
        """Beta-CMF early-time shape parameter (alpha > 0)."""
        return self._alpha

    @property
    def beta_val(self) -> float:
        """Beta-CMF late-time shape parameter (beta > 0)."""
        return self._beta

    @property
    def kappa_val(self) -> float:
        """Concentration parameter kappa = alpha + beta."""
        return self._alpha + self._beta

    @property
    def t_mean_q(self) -> u.Quantity:
        """Mean formation time E[t] implied by alpha/beta (in Gyr)."""
        te = self._t_end_q
        return te * (self._alpha / (self._alpha + self._beta))

    def _alpha_beta_from_mean_concentration(
        self,
        t_mean: Parameter | u.Quantity | float,
        kappa: Parameter | u.Quantity | float
    ) -> tuple[float, float]:
        """
        Map mean+concentration to alpha/beta.

        Parameters
        ----------
        t_mean : float or Quantity or Parameter
            If float: fractional mean m in (0,1).
            If Quantity/Parameter: cosmic mean time in Gyr, converted to m=t_mean/t_end.
        kappa : float or Quantity or Parameter
            Concentration (>0), dimensionless.

        Returns
        -------
        alpha, beta : float
        """
        te = self._t_end_q

        # kappa
        kq = check_parameter(kappa, u.dimensionless_unscaled).q.value
        if kq <= 0:
            raise ValueError("kappa must be > 0.")

        # fractional mean m
        if isinstance(t_mean, (u.Quantity, Parameter)):
            tmq = t_mean.q if isinstance(t_mean, Parameter) else check_unit(t_mean, u.Gyr)
            m = (tmq / te).decompose().value
        else:
            m = float(t_mean)

        if not (0.0 < m < 1.0):
            raise ValueError("Mean fraction m must be in (0,1).")

        alpha = m * float(kq)
        beta = (1.0 - m) * float(kq)
        if alpha <= 0 or beta <= 0:
            raise ValueError("Derived alpha/beta must be > 0.")
        return alpha, beta

    def _rescaled_x(self, time: u.Quantity) -> np.ndarray:
        """
        Rescale cosmic time to x in [0,1] over [t_start, t_end], clipping outside.
        """
        t = check_unit(time, u.Gyr).to_value("Gyr")
        t0 = (self.t_start.q if isinstance(self.t_start, Parameter) else check_unit(self.t_start, u.Gyr)).to_value("Gyr")
        t1 = self._t_end_q.to_value("Gyr")
        x = (t - t0) / (t1 - t0)
        return np.clip(x, 0.0, 1.0)

    # -----------------------------------------------------------------
    # CEM API
    # -----------------------------------------------------------------
    @_check_time_dec
    def stellar_mass_formed(self, time: u.Gyr) -> u.Quantity:
        """
        Cumulative formed stellar mass M_formed(t).
        """
        x = self._rescaled_x(time)
        f = special.betainc(self._alpha, self._beta, x)  # regularised incomplete beta
        return self.mass_today.q * f

    @_check_time_dec
    def ism_metallicity(self, time: u.Gyr) -> u.Quantity:
        """
        Constant ISM metallicity (mass fraction).
        """
        # handle scalar and array times
        shape = np.shape(time.value)
        z = np.full(shape, self.ism_metallicity_today.q.value)
        return z << self.ism_metallicity_today.q.unit

    @_check_time_dec
    def sfr(self, time: u.Gyr) -> u.Quantity:
        """
        Instantaneous SFR(t) (analytic Beta PDF mapped onto [t_start,t_end]).

        Notes
        -----
        - Outside (t_start, t_end) the SFR is set to 0 by construction.
        - Integrating this over time yields ``mass_today``.
        """
        x = self._rescaled_x(time)  # clipped to [0,1]
        dt = self._t_end_q - (self.t_start.q if isinstance(self.t_start, Parameter) else check_unit(self.t_start, u.Gyr))

        # Beta PDF (0 at endpoints after clipping); ensure zeros outside open interval
        pdf = np.zeros_like(x, dtype=float)
        mask = (x > 0.0) & (x < 1.0)
        if np.any(mask):
            # betapdf = x^(a-1) (1-x)^(b-1) / B(a,b)
            logB = (special.gammaln(self._alpha) + special.gammaln(self._beta) - special.gammaln(self._alpha + self._beta))
            logpdf = ((self._alpha - 1.0) * np.log(x[mask]) +
                      (self._beta - 1.0) * np.log1p(-x[mask]) -
                      logB)
            pdf[mask] = np.exp(logpdf)

        return (self.mass_today.q / dt) * pdf  # Msun/Gyr

    def t_peak(self) -> Optional[u.Quantity]:
        """
        Cosmic time of peak SFR (exists only if alpha>1 and beta>1).
        """
        if (self._alpha > 1.0) and (self._beta > 1.0):
            t0 = self.t_start.q if isinstance(self.t_start, Parameter) else check_unit(self.t_start, u.Gyr)
            dt = self._t_end_q - t0
            x_peak = (self._alpha - 1.0) / (self._alpha + self._beta - 2.0)
            return t0 + x_peak * dt
        return None

#-------------------------------------------------------------------------------
@dataclass
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
    times: Parameter | u.Quantity | np.array = None
    masses: Parameter | u.Quantity | np.array = None
    metallicities: Parameter | u.Quantity | np.array = None

    name: str = field(default="tabular_cem", init=False)

    def __post_init__(self):
        super().__post_init__()
        self.times = check_parameter(self.times, u.Gyr, doc="")
        self.masses = check_parameter(self.masses, u.Msun, doc="")
        self.metallicities = check_parameter(self.metallicities, u.dimensionless_unscaled,
                                             doc="")
        
        sort_times = np.argsort(self.times.q)
        self.times.q = self.times.q[sort_times]
        self.masses.q = self.masses.q[sort_times]
        self.metallicities.q = self.metallicities.q[sort_times]

    @property
    def table_t(self):
        return self.times.q

    @property
    def table_mass(self):
        return self.masses.q

    @property
    def table_metallicity(self):
        return self.metallicities.q

    @_check_time_dec
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
        integral = interpolator(times.to_value(self.table_t.unit)
                                ) << self.table_mass.unit
        integral[times > self.table_t[-1]] = self.table_mass[-1]
        integral[times < self.table_t[0]] = 0
        return integral

    @_check_time_dec
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

@dataclass
class TabularCEM_ZPowerLaw(MassPropMetallicityMixin, TabularCEM):
    """Chemical evolution model based on a grid of times that assumes an analytic chemical enrichment history.

    See Also
    --------
    :class:`TabularCEM`
    :class:`MassPropMetallicityMixin`
    """
    mass_today: Parameter | u.Quantity | float = None
    ism_metallicity_today: Parameter = 0.02 << u.dimensionless_unscaled
    alpha_powerlaw: Parameter = 1 << u.dimensionless_unscaled
    name: str = field(default="tabular_zpowlaw_cem", init=False)

    def __post_init__(self):
        # If metallicities were not provided, create a dummy one BEFORE TabularCEM __post_init__
        if self.metallicities is None and self.times is not None:
            # times may be Parameter/Quantity/array; we just need a length
            t = self.times.q if isinstance(self.times, Parameter) else check_unit(self.times, u.Gyr)
            self.metallicities = np.zeros(np.size(t)) << u.dimensionless_unscaled

        super().__post_init__()

        if self.mass_today is None:
            self.mass_today = Parameter(self.stellar_mass_formed(self.today),
                                        doc="Stellar mass at observing time")
        self.ism_metallicity_today = check_parameter(
            self.ism_metallicity_today, u.dimensionless_unscaled, doc="Metallicity at present"
        )
        self.alpha_powerlaw = check_parameter(
            self.alpha_powerlaw, u.dimensionless_unscaled, doc="Metallicity evolution exponent"
        )


class CC25TabularCEM(TabularCEM_ZPowerLaw):
    r"""Chemical evolution model based on the SFH parameterization from Corcho-Caballero et al. (2025).

    The SFH is defined in terms of the average specific star formation rate (``ssfr``)
    several timescales (``tau_ssfr``), expressed in lookback time with respect
    to the age of the Universe at the time of observation, related by the expression:

    .. math::
        M_\star(t) = M_{\rm today} \cdot \left(1 - \tau_{\rm ssfr} \cdot ssfr \right)

    See Also
    --------
    :class:`TabularCEM_ZPowerLaw`
    :class:`MassPropMetallicityMixin`
    """

    def __init__(self, *, tau_ssfr, ssfr, **kwargs):
        # Read-only, set by parent class
        today = check_unit(kwargs["today"], u.Gyr)
        mass_today = check_unit(kwargs["mass_today"], u.Msun)

        self._tau_ssfr = check_parameter(tau_ssfr, u.Gyr)
        self._ssfr = check_parameter(ssfr, 1 / u.Gyr)
        
        if np.any(self.tau_ssfr.q <= 0 * u.Gyr):
            raise ValueError("All tau_ssfr values must be positive.")
        if np.any(self.tau_ssfr.q > check_unit(today, u.Gyr)):
            raise ValueError("All tau_ssfr values must be less than today.")

        # Sort in descending order
        sort_idx = np.argsort(self.tau_ssfr.q)[::-1]
        self.tau_ssfr.q = self.tau_ssfr.q[sort_idx]
        self.ssfr.q = self.ssfr.q[sort_idx]

        times = today - self.tau_ssfr.q
        times = np.insert(times, (0, times.size), (0 * u.Gyr, today))

        masses = mass_today * (1 - self.tau_ssfr.q * self.ssfr.q)
        masses = np.insert(masses, (0, masses.size), (0.0 * u.Msun, mass_today))

        super().__init__(
            times=times,
            masses=masses,
            **kwargs,
        )
    
    @property
    def ssfr(self):
        return self._ssfr

    @ssfr.setter
    def ssfr(self, value):
        value = check_parameter(value, 1 / u.Gyr)
        self._ssfr = value
        masses = self.mass_today.q * (1 - self.tau_ssfr.q * self._ssfr.q)
        self.masses = Parameter(np.insert(masses, (0, masses.size), (0.0 * u.Msun, self.mass_today)))

    @property
    def tau_ssfr(self):
        return self._tau_ssfr
    
    @tau_ssfr.setter
    def tau_ssfr(self, value):
        value = check_parameter(value, u.Gyr)
        if np.any(value.q <= 0 * u.Gyr):
            raise ValueError("All tau_ssfr values must be positive.")
        if np.any(value.q > self.today):
            raise ValueError("All tau_ssfr values must be less than today.")
        if np.diff(value.q) >= 0:
            raise ValueError("Input ``tau_ssfr`` must be strictly decreasing")
        self._tau_ssfr = value

        times = self.today - self._tau_ssfr.q
        self.times = Parameter(np.insert(times, (0, times.size), (0 * u.Gyr, self.today)))


@dataclass
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
    time_form: Parameter | u.Quantity | np.array = None
    metallicities: Parameter | u.Quantity | np.array = None
    masses: Parameter | u.Quantity | np.array = None

    def __post_init__(self):
        super().__post_init__()
        self.time_form = check_parameter(self.time_form, u.Gyr,
                                         doc="Particle formation time")
        self.metallicities = check_parameter(self.metallicities, u.dimensionless_unscaled,
                                             doc="Particle metallicity (at observing time)")
        self.masses = check_parameter(self.masses, u.Msun,
                                      doc="Particle stellar mass")

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
        valid_particles = self.time_form.q <= t_obs
        return ssp.get_weights(ages=t_obs - self.time_form.q[valid_particles],
                               metallicities=self.metallicities.q[valid_particles],
                               masses=self.masses.q[valid_particles])

    def stellar_mass_formed(self, time):
        sort_idx = np.argsort(self.time_form.q)
        mass_history = np.cumsum(self.masses.q[sort_idx])
        return np.interp(time, self.time_form.q[sort_idx], mass_history)

    def ism_metallicity(self, time):
        return np.full(time.size, fill_value=np.nan)

# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/
# -----------------------------------------------------------------------------

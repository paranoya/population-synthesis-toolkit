"""
Dust Extinction/Emission Models for Stellar Populations

This module implements dust extinction and emission models for stellar population synthesis,
including a base class for general dust models and specific implementations such as a dust screen 
and the Charlot & Fall (2000) extinction model.

Usage
-----
This module is intended for applying dust extinction or emission to stellar spectra, either
to synthetic stellar populations or other types of spectra.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from astropy import units as u
from astropy import constants as const
from astropy.modeling.physical_models import BlackBody
import numpy as np

import extinction as _extinction_lib

from pst.utils import check_unit, broadcast_to_axis

## Utils ###


def modified_blackbody(
    lam: u.Quantity,
    T: u.Quantity,
    beta: float,
    lam_0: Optional[u.Quantity] = None,
    lam_ref: u.Quantity = 100 * u.um,
) -> u.Quantity:
    """
    Modified blackbody (greybody) spectrum.

    This returns a *shape* proportional to:
      S_nu ∝ (1 - exp(-(nu/nu0)^beta)) * B_nu(T)
    If lam_0 is None, uses an optically thin emissivity:
      S_nu ∝ (nu/nu_ref)^beta * B_nu(T)

    Parameters
    ----------
    lam : Quantity
        Wavelength array.
    T : Quantity
        Dust temperature.
    beta : float
        Emissivity index.
    lam_0 : Quantity, optional
        Wavelength where tau(nu)=1 (i.e. turnover/optical-depth scale).
        If provided, uses the optical-depth form.
    lam_ref : Quantity, optional
        Reference wavelength for the optically thin emissivity normalization
        (only used when lam_0 is None). Default 100 micron.

    Returns
    -------
    mbb : Quantity
        Greybody spectrum proportional to S_nu (same units as B_nu(T)).
    """
    lam = lam.to(u.um)
    T = T.to(u.K)
    lam_ref = lam_ref.to(u.um)

    nu = (const.c / lam).to(u.Hz)

    bb = BlackBody(temperature=T)
    Bnu = bb(nu)  # spectral radiance per Hz

    if lam_0 is not None:
        lam_0 = lam_0.to(u.um)
        # tau(nu) = (nu/nu0)^beta = (lam0/lam)^beta
        tau = (lam_0 / lam) ** beta
        factor = -np.expm1(-tau)  # = 1 - exp(-tau), dimensionless
    else:
        # optically thin: emissivity ∝ nu^beta, make it dimensionless with a reference
        nu_ref = (const.c / lam_ref).to(u.Hz)
        factor = (nu / nu_ref) ** beta  # dimensionless

    return Bnu * factor

### Attenuation curve (wavelength dependence) ###

class AttenuationCurve(ABC):
    """
    Dust attenuation curve.

    Must implement A_lambda in magnitudes for given A_V.
    """
    name: str

    @abstractmethod
    def a_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """Return attenuation in magnitudes (same length as wavelength)."""
        raise NotImplementedError

    def tau_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> np.ndarray:
        """Convert magnitudes to optical depth tau: A = 1.086 * tau."""
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        return a_lam / 1.086

    def attenuation_factor(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """Return multiplicative attenuation factor: 10^(-0.4 a_lambda)."""
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        f = 10.0 ** (-0.4 * a_lam)
        return f << u.dimensionless_unscaled

@dataclass
class ExtinctionLibCurve(AttenuationCurve):
    """
    Thin wrapper around the `extinction` python package (ccm89, odonnell94, calzetti00, ...).

    Notes
    -----
    The `extinction` package functions generally expect wavelength in Angstrom (float),
    and return A_lambda (magnitudes) given (A_V, R_V).
    """
    name: str

    def __post_init__(self):
        try:
            self._law = getattr(_extinction_lib, self.name)
        except AttributeError as e:
            raise ValueError(f"Unknown extinction law '{self.name}' in extinction package.") from e

    def a_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        wav = check_unit(wavelength).to_value(u.AA)
        r_v = params.get('r_v', 3.1)  # ensure R_V is in params
        a_lam = self._law(np.array(wav, dtype=float), float(a_v), float(r_v))
        return a_lam << u.mag


### Attenuation model (geometrical effects) ###

class AttenuationModel(ABC):
    """
    Geometry-aware attenuation model.

    Returns a multiplicative factor that can be applied to spectra.
    """

    @abstractmethod
    def attenuation_factor(self, wavelength: u.Quantity, **params) -> u.Quantity:
        raise NotImplementedError

    def apply(self, wavelength, spectra, axis: int = -1, **params):
        wavelength = check_unit(wavelength, u.AA)
        f = self.attenuation_factor(wavelength, **params)  # (N,) dimensionless
        f_np = broadcast_to_axis(f.to_value(u.dimensionless_unscaled), np.ndim(spectra), axis=axis)
        return spectra * f_np


@dataclass
class DustScreenAttenuation(AttenuationModel):
    """
    Simple foreground screen: factor = curve.factor(wave, A_V).
    """
    curve: AttenuationCurve | str

    def __post_init__(self):
        if isinstance(self.curve, str):
            self.curve = ExtinctionLibCurve(name=self.curve)

    def attenuation_factor(self, wavelength: u.Quantity, *, A_V: float = 1.0, **params) -> u.Quantity:
        return self.curve.factor(wavelength, A_V=float(A_V), **params)


@dataclass
class CharlotFall00Attenuation(AttenuationModel):
    """
    Charlot & Fall (2000)-style two-component attenuation:
    - young populations get A_V_young
    - old populations get A_V_old
    threshold set by young_age

    """
    curve: str | List[AttenuationCurve] | List[str]
    young_age: u.Quantity = 10.0 << u.Myr

    def __post_init__(self):
        if isinstance(self.curve, str):
            # Both components use the same curve
            self.curve = 2 * [ExtinctionLibCurve(name=self.curve)]
        elif isinstance(self.curve, list):
            self.curve = [ExtinctionLibCurve(name=c) if isinstance(c, str) else c for c in self.curve]

    def attenuation_factor(
        self,
        wavelength: u.Quantity,
        *,
        age: u.Quantity,
        a_v_young: float = 1.0,
        a_v_old: float = 0.3,
        **params,
    ) -> u.Quantity:
        wavelength = check_unit(wavelength, u.AA)
        age = check_unit(age, u.Gyr)

        age = np.atleast_1d(age)
        young = age < self.young_age

        f_y = self.curve[0].factor(wavelength, a_v=float(a_v_young),
                                   **params).to_value(u.dimensionless_unscaled)
        f_o = self.curve[1].factor(wavelength, a_v=float(a_v_old),
                                   **params).to_value(u.dimensionless_unscaled)

        out = np.empty((age.size, wavelength.size), dtype=float)
        out[young] = f_y
        out[~young] = f_o
        return out * u.dimensionless_unscaled

# -----------------------------------------------------------------------------
# Emission models
# -----------------------------------------------------------------------------

class EmissionModel(ABC):
    """
    Dust emission component.

    Returns an additive spectrum on the wavelength grid.
    """

    @abstractmethod
    def emission_spectrum(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Returns emission spectrum as Quantity.

        The *units* are determined by your chosen normalization parameterization
        (e.g., L_IR scaling or amplitude in L_lambda). Keep this consistent with PST.
        """
        raise NotImplementedError

    def add_to(self, wavelength, spectra, axis: int = -1, **params):
        wavelength = check_unit(wavelength, u.AA)
        em = self.emission_spectrum(wavelength, **params)
        em_val = em.to_value(em.unit)
        em_val = broadcast_to_axis(em_val, np.ndim(spectra), axis=axis)
        return spectra + em_val * em.unit


@dataclass
class Casey2012Emission(EmissionModel):
    r"""
    Casey (2012) IR SED parameterization

    This model combines a modified blackbody (MBB) component for the cold dust
    emission with a mid-infrared (MIR) power-law component to account for warmer
    dust and PAH features. The two components are smoothly joined at a pivot wavelength
    by the following expression

    .. math::
        S(\lambda) = N_{bb} \frac{\left(1 - e^{-(\lambda_0/\lambda)^\beta}\right)\left(\frac{c}{\lambda}\right)^3}{e^{hc/\lambda kT} - 1} + N_{pl} \lambda^{\alpha} e^{-(\lambda/\lambda_c)^2}

    """

    optically_thin: bool = False
    ir_range_default: Tuple[u.Quantity, u.Quantity] = (8*u.um, 1000*u.um)

    def emission_spectrum(
        self,
        wavelength: u.Quantity,
        *,
        t_dust: float = 35.0,
        beta: float = 1.5,
        alpha: float = 2.0,
        lam0: Optional[u.Quantity] = None,
        lum_ir: Optional[u.Quantity] = None,
        lam_pivot: u.Quantity = None,
    ) -> u.Quantity:
        """
        Return L_lambda-like spectrum for dust emission.

        Notes
        -----
        This is a *draft* implementation: the exact Casey2012 join functional form
        is encapsulated in `_shape_nu()` so you can refine it without touching the API.
        """
        wav = check_unit(wavelength, u.AA)
        t_dust = check_unit(t_dust, u.K)

        if lam_pivot is None:
            lam_pivot = self._lambda_pivot(t_dust, alpha)

        # Build components
        mbb, powlaw = self._shape_l_nu(
            lam=wav,
            t_dust=t_dust,
            beta=beta,
            alpha=alpha,
            lam0=lam0,
            lam_pivot=lam_pivot)

        # Combine components
        shape_nu = (mbb.to_value("Jy /sr") + powlaw.to_value("Jy /sr")) << u.Jy

        if lum_ir is not None:
            lum_ir = check_unit(lum_ir)
            # Normalize to given L_IR over 8-1000 micron
            ir_min, ir_max = self.ir_range_default
            ir_mask = (wav >= ir_min) & (wav <= ir_max)
            wav_ir = wav[ir_mask]

            shape_nu_ir = shape_nu[ir_mask]
            # L_IR = integral L_nu dnu = integral L_nu * c / lambda^2 d lambda
            integrand_ir = shape_nu_ir * (const.c / wav_ir**2)
            l_ir_model = np.trapz(integrand_ir, wav_ir)
            print(l_ir_model.to("W / m2"))
            norm_factor = (lum_ir / l_ir_model).decompose()
            print(norm_factor)
            shape_nu *= norm_factor
        # Convert to L_lambda-like quantity: L_lambda = L_nu * c / lambda^2
        # shape_lam = (shape_nu_q * u.speed_of_light / wav**2).to(u.W / u.m)
        return shape_nu

    def _shape_l_nu(
        self,
        lam: u.Quantity,
        *,
        t_dust: u.Quantity,
        beta: float,
        alpha: float,
        lam_pivot: u.Quantity = None,
        lam0: Optional[u.Quantity]= 200 << u.um,
    ) -> np.ndarray:
        """
        Draft spectral shape in frequency space.

        You should replace the join with the exact Casey2012 prescription you prefer.
        The important thing is: keep it smooth and monotonic around the pivot.

        Returns
        -------
        shape_nu : ndarray
            Dimensionless shape proportional to L_nu.
        """
        if lam_pivot is None:
            lam_pivot = self._lambda_pivot(t_dust, alpha)
        # Cold-dust modified blackbody
        mbb = modified_blackbody(lam, T=t_dust, beta=beta, lam_0=lam0 if lam0 is not None else None)
        # MIR power-law term
        powlaw = (lam/lam_pivot)**alpha * np.exp(-(lam/lam_pivot)**2)
        # Power-law contribution
        mbb_pivot = modified_blackbody(lam_pivot, T=t_dust, beta=beta, lam_0=lam0).to(u.Jy/u.sr)
        powlaw *= mbb_pivot  # Jy/sr
        return mbb.to(u.Jy / u.sr), powlaw.to(u.Jy / u.sr)
    
    def _lambda_pivot(self, t_dust: u, alpha: float) -> u.Quantity:
        """
        Estimate pivot wavelength where MBB and power-law components intersect.

        This is a rough estimate; you may want to refine it.
        """
        if self.optically_thin:
            lam_c = 1 / (t_dust.to_value(u.K) + (26.68 + 6.246 * alpha))
        else:
            lam_c = 1 / ((26.68 + 6.246 * alpha)**-2 + (1.905e-4 + 7.243e-5 * alpha) * t_dust.to_value(u.K))
        return lam_c << u.um


class DustModelBase(ABC):
    """
    Abstract base class for dust extinction and emission models.

    Description
    -----------
    This class provides the framework for implementing dust models. Subclasses
    should define methods to compute the extinction and emission due to dust.
    
    Attributes
    ----------
    extinction_law : str
        The name of the extinction law to be used. This is retrieved from the 
        `extinction` library.
    """

    @abstractmethod
    def get_extinction(self, *args, **kwargs):
        """
        Compute the dust extinction for a given set of parameters.
        
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_emission(self, *args, **kwargs):
        """
        Compute the dust emission for a given set of parameters.
        
        This method must be implemented by subclasses.
        """
        pass

    def apply_extinction(self, wavelength, spectra, axis=-1, **kwargs):
        """
        Apply the dust extinction model to a given spectra.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array. Can be either a numpy array of floats or an `astropy.Quantity`
            with associated units (e.g., Angstroms).
        spectra : np.ndarray or astropy.Quantity
            Array of spectra to which the extinction will be applied.
        axis : int, optional
            The axis of the spectra array corresponding to the wavelength dimension. Default is -1.
        **kwargs
            Additional keyword arguments passed to the `get_extinction` method.

        Returns
        -------
        reddened_spectra : np.ndarray
            The spectra array with dust extinction applied.
        """
        ext = self.get_extinction(wavelength, **kwargs)
        if ext.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            ext = np.expand_dims(ext, new_dims)
        return spectra * ext

    def apply_emission(self, wavelength, spectra, axis=-1, **kwargs):
        """
        Add the predicted dust emission to a given spectra.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array. Can be either a numpy array of floats or an `astropy.Quantity`
            with associated units.
        spectra : np.ndarray or astropy.Quantity
            Array of spectra to which the dust emission will be added.
        axis : int, optional
            The axis of the spectra array corresponding to the wavelength dimension. Default is -1.
        **kwargs
            Additional keyword arguments passed to the `get_emission` method.

        Returns
        -------
        spectra_with_emission : np.ndarray
            The spectra array with dust emission added.
        """
        emission = self.get_emission(wavelength, **kwargs)
        if emission.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            emission = np.expand_dims(emission, new_dims)
        return spectra + emission
        
    def redden_ssp_model(self, ssp_model, **kwargs):
        """
        Apply extinction to a simple stellar population (SSP) model.

        Parameters
        ----------
        ssp_model : `pst.SSPBase` object
            A simple stellar population (SSP) model instance.
        **kwargs
            Additional keyword arguments passed to the `apply_extinction` method.

        Returns
        -------
        reddened_ssp_model : `pst.SSPBase` object
            The SSP model with dust extinction applied.
        """
        reddened_ssp_model = ssp_model.copy()
        reddened_ssp_model.L_lambda = self.apply_extinction(
            ssp_model.wavelength, reddened_ssp_model.L_lambda, axis=-1, **kwargs)
        return reddened_ssp_model


class DustScreen(DustModelBase):
    """
    Dust screen extinction model.

    Implements a simple dust screen model where dust extinction is applied
    to spectra using a specified extinction law and R_V parameter.

    Attributes
    ----------
    extinction_law_name : str
        The name of the extinction law from the `extinction` library (e.g., 'ccm89', 'odonnell94').
    r_extinction : float
        The R_V value for the extinction law. Default is 3.1.
    """
    def __init__(self, extinction_law_name, r_extinction=3.1):
        # super().__init__(extinction_law)
        self.extinction_law_name = extinction_law_name
        self.r_extinction = r_extinction

        self.extinction_law = getattr(_extinction_lib, self.extinction_law_name)

    def get_extinction(self, wavelength, a_v=1.0):
        """
        Compute the dust extinction.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array in Angstroms.
        a_v : float, optional
            The V-band extinction (in magnitudes). Default is 1.0.

        Returns
        -------
        extinction_curve : np.ndarray
            Dimensionless extinction factor to be applied to the spectra.
        """
        return 10**(-0.4 * self.extinction_law(
            np.array(wavelength.to_value("angstrom"), dtype=float),
            a_v, self.r_extinction)) <<  u.dimensionless_unscaled

    def get_emission(self, wavelength):
        """
        Compute the dust emission.

        For this model, no dust emission is included, so this method returns zeros.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array.

        Returns
        -------
        emission : np.ndarray
            An array of zeros with the same shape as `wavelength`.
        """
        return np.zeros(wavelength.size)

class CF03DustScreen(DustScreen):
    """
    Charlot & Fall (2000) dust screen model for young and old stellar populations.

    This model applies different extinction curves to young and old populations
    based on their ages.

    Parameters
    ----------
    extinction_law_name : str
        The name of the extinction law from the `extinction` library.
    young_ssp_age : astropy.Quantity
        The age threshold for defining young populations (in years).
    r_extinction : float, optional
        The R_V value for the extinction law. Default is 3.1.
    """
    def __init__(self, extinction_law_name, young_ssp_age, r_extinction=3.1):
        assert isinstance(young_ssp_age, u.Quantity), "young_ssp_age must be an astropy.Quantity"
        self.young_ssp_age = young_ssp_age
        super().__init__(extinction_law_name, r_extinction=r_extinction)
    
    def get_extinction(self, wavelength, age, a_v_young=1.0, a_v_old=0.3):
        """
        Compute the dust extinction for young and old stellar populations.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array.
        age : np.ndarray or astropy.Quantity
            Array of stellar population ages.
        a_v_young : float, optional
            V-band extinction for young populations. Default is 1.0.
        a_v_old : float, optional
            V-band extinction for old populations. Default is 0.3.

        Returns
        -------
        extinction_curve : np.ndarray
            2D array of extinction factors with shape (age.size, wavelength.size).
        """
        age = np.atleast_1d(age)
        young = age < self.young_ssp_age
        ext = np.zeros((age.size, wavelength.size))
        ext[young] = super().get_extinction(wavelength, a_v_young) 
        ext[~young] = super().get_extinction(wavelength, a_v_old)
        return ext

        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # Redden some set of spectra using Charlote and Fall 03 model
    dust_model = CF03DustScreen("ccm89", young_ssp_age=10 * u.yr)
    
    wavelength = np.linspace(1000, 10000) * u.angstrom
    spectra = np.ones((10, wavelength.size))
    ages = np.linspace(5, 15, 10) * u.yr
    reddened_spectra = dust_model.apply_extinction(wavelength, spectra,
                                                   age=ages,
                                                   a_v_young=1.0, a_v_old=0.3)
    
    plt.figure()
    plt.title("Charlote and Fall 03 dust extinction model")
    plt.plot(wavelength, spectra[0], label=f'Unreddened')
    plt.plot(wavelength, reddened_spectra[0], label=f'Age={ages[0]}')
    plt.plot(wavelength, reddened_spectra[-1], label=f'Age={ages[-1]}')
    plt.legend()
    
    # Apply the extinction to a given SSP model
    from pst.SSP import PopStar
    ssp = PopStar(IMF='cha')
    dust_model = DustScreen("ccm89",)
    red_ssp = dust_model.redden_ssp_model(ssp, a_v=1.0)
    
    plt.figure()
    plt.title("Redden SSP model")
    plt.loglog(ssp.wavelength, ssp.L_lambda[3, -1])
    plt.loglog(ssp.wavelength, red_ssp.L_lambda[3, -1])
    plt.xlim(800, 1e5)
    plt.ylim(1e-8, 1e-4)
    
    # Little performance test
    from time import time
    a_v = np.linspace(0.1, 3, 1)
    tstart = time()
    ssps = [dust_model.redden_ssp_model(ssp, a_v=av) for av in a_v]
    tend = time()
    print(f"Time for generating {a_v.size} SSP models: {tend - tstart}")

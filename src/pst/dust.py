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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from astropy import units as u
from astropy import constants as const
from astropy.modeling.physical_models import BlackBody
import numpy as np

import extinction as _extinction_lib

from pst.model import Parameter, ModelBase
from pst.utils import check_unit, broadcast_to_axis
from pst.sed import SedComponent, StellarComponent

## Utils ###


def modified_blackbody(
    lam: u.Quantity,
    T: u.Quantity,
    beta: float,
    lam_0: Optional[u.Quantity] = None,
    lam_ref: u.Quantity = 100 * u.um,
    per_freq=False
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
    if per_freq:
        return Bnu * factor
    else:
        return (Bnu * factor).to("erg / (Angstrom s sr cm2)", 
                                 u.spectral_density(lam))


### Attenuation curve (wavelength dependence) ###

class AttenuationCurve(ABC, ModelBase):
    """
    Base class for dust attenuation curves.

    This interface defines a wavelength dependent attenuation curve expressed
    as A_lambda in magnitudes for a given V band attenuation a_v. Implementations
    must provide a_lambda, and the base class provides convenience conversions
    to optical depth and multiplicative attenuation factors.

    Notes
    -----
    The primary method a_lambda returns attenuation in magnitudes. The helper
    methods implement common conversions:

    - tau_lambda converts magnitudes to optical depth using A = 1.086 * tau
    - attenuation_factor returns the multiplicative factor applied to spectra
      as 10**(-0.4 * A_lambda)

    Attributes
    ----------
    name : str
        Name identifying the curve or law.
    """

    name: str

    @abstractmethod
    def a_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """
        Compute the attenuation A_lambda in magnitudes.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float
            V band attenuation in magnitudes.
        **params
            Additional curve parameters. Subclasses may define parameters such
            as r_v, bump strength, or slope modifiers.

        Returns
        -------
        a_lam : astropy.units.Quantity
            Attenuation in magnitudes with the same shape as wavelength.
        """
        raise NotImplementedError

    def tau_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """
        Convert attenuation in magnitudes to optical depth.

        Uses the relation A = 1.086 * tau.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the optical depth is evaluated.
        a_v : float
            V band attenuation in magnitudes.
        **params
            Additional curve parameters forwarded to a_lambda.

        Returns
        -------
        tau : astropy.units.Quantity
            Optical depth as a dimensionless quantity with the same shape as
            wavelength.
        """
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        return (a_lam / 1.086) << u.dimensionless_unscaled

    def attenuation_factor(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """
        Compute the multiplicative attenuation factor.

        The factor is defined as 10**(-0.4 * A_lambda), suitable for multiplying
        spectra expressed as flux or luminosity densities.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float
            V band attenuation in magnitudes.
        **params
            Additional curve parameters forwarded to a_lambda.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor with the same shape as wavelength.
        """
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        f = 10.0 ** (-0.4 * a_lam)
        return f << u.dimensionless_unscaled


@dataclass
class PowerLawAttenuationCurve(AttenuationCurve):
    """
    Power law attenuation curve.

    This class implements a simple attenuation law where A_lambda follows a
    power law in wavelength relative to a reference wavelength.

    The curve is commonly used as a building block in two component attenuation
    prescriptions such as Charlot and Fall 2000, where different normalizations
    are applied to young and old stellar populations.

    Parameters
    ----------
    alpha : float
        Power law slope. For alpha < 0 the attenuation decreases with
        increasing wavelength.
    turn_over_wl : astropy.units.Quantity
        Reference wavelength that sets the normalization point of the power law.
        The ratio wavelength / turn_over_wl is made dimensionless before raising
        to alpha.

    Notes
    -----
    The attenuation in magnitudes is computed as:

    A_lambda = a_v * (wavelength / turn_over_wl) ** alpha

    where a_v is interpreted as the attenuation at the reference wavelength
    turn_over_wl. This differs from laws that interpret a_v strictly as V band
    attenuation. Choose turn_over_wl accordingly if you want a_v to correspond
    to A_V.

    The method returns A_lambda as an astropy Quantity with units of magnitudes
    and the same shape as the input wavelength.

    See Also
    --------
    AttenuationCurve : Base interface for attenuation curves.
    CharlotFall00Attenuation : Two component attenuation model that can use
        different curves for young and old populations.
    """

    alpha: float
    turn_over_wl: u.Quantity

    def __post_init__(self):
        if self.alpha > 0:
            raise ValueError("alpha must be negative")

    def a_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """
        Compute attenuation A_lambda in magnitudes.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float
            Attenuation normalization in magnitudes at turn_over_wl.
        **params
            Additional parameters reserved for future extensions. Currently
            unused.

        Returns
        -------
        a_lam : astropy.units.Quantity
            Attenuation in magnitudes with the same shape as wavelength.
        """
        ratio = (wavelength / self.turn_over_wl).decompose()
        return (a_v * np.power(ratio, self.alpha)) << u.mag


@dataclass
class ExtinctionLibCurve(AttenuationCurve):
    """
    Attenuation curve wrapper for the ``extinction`` python package.

    This class wraps laws implemented by the extinction package such as ccm89,
    odonnell94, and calzetti00.

    The extinction package functions typically accept wave, a_v, r_v, and unit,
    where wave is a float array in Angstrom and the return value is A_lambda in
    magnitudes.

    Parameters
    ----------
    name : str
        Name of the extinction law function in the extinction package.

    Raises
    ------
    ValueError
        If the requested law name is not present in the extinction package.

    Notes
    -----
    The extinction functions are expected to have a signature compatible with:

    - wave: float array in Angstrom
    - a_v: float
    - r_v: float
    - return: A_lambda in magnitudes

    The parameter r_v is read from params with a default value of 3.1.

    See Also
    --------
    AttenuationCurve : Base interface for attenuation curves.
    """

    name: str

    def __post_init__(self):
        """
        Resolve the law function from the extinction package.
        """
        try:
            self._law = getattr(_extinction_lib, self.name)
        except AttributeError as e:
            raise ValueError(f"Unknown extinction law '{self.name}' in extinction package.") from e

    def a_lambda(self, wavelength: u.Quantity, *, a_v: float, **params) -> u.Quantity:
        """
        Compute A_lambda using the extinction package law.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float
            V band attenuation in magnitudes.
        **params
            Additional parameters. The recognized parameter is r_v.

            - r_v: float, optional
              Total to selective extinction ratio. Default is 3.1.

        Returns
        -------
        a_lam : astropy.units.Quantity
            Attenuation in magnitudes with the same shape as wavelength.
        """
        wav = check_unit(wavelength).to_value(u.AA)
        r_v = params.get("r_v", 3.1)
        a_lam = self._law(np.array(wav, dtype=float), float(a_v), float(r_v))
        return a_lam << u.mag


class AttenuationModel(ModelBase, ABC):
    """
    Base class for geometry dependent attenuation models.

    An attenuation model returns a multiplicative attenuation factor that can be
    applied to spectra. Models can represent different geometries such as a
    foreground screen or two component prescriptions.

    Subclasses must implement attenuation_factor.

    Notes
    -----
    The apply method multiplies the input spectra by the model attenuation
    factor, broadcasting the factor along the selected axis.

    See Also
    --------
    DustScreenAttenuation : Foreground screen model.
    CharlotFall00Attenuation : Two component model inspired by Charlot and Fall 2000.
    """

    @abstractmethod
    def attenuation_factor(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Compute the multiplicative attenuation factor.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation factor is evaluated.
        **params
            Model parameters. The required and optional parameters depend on the
            subclass.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor. The shape is model dependent. For
            most models this is (n_wave,), but some models can return multiple
            factors for different components.
        """
        raise NotImplementedError

    def apply(self, wavelength: u.Quantity, spectra, axis: int = -1, **params):
        """
        Apply attenuation to spectra.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid corresponding to the spectra.
        spectra : array-like or astropy.units.Quantity
            Input spectra to be attenuated. The wavelength axis is given by
            axis.
        axis : int, optional
            Axis in spectra corresponding to wavelength. Default is -1.
        **params
            Parameters forwarded to attenuation_factor.

        Returns
        -------
        attenuated : same type as spectra
            Spectra multiplied by the attenuation factor, broadcast to the
            selected axis.

        Notes
        -----
        This method assumes attenuation_factor returns a wavelength dependent
        factor of shape (n_wave,). For models that return multiple components,
        the caller should handle component selection or reduction before calling
        apply.
        """
        wavelength = check_unit(wavelength, u.AA)
        f = self.attenuation_factor(wavelength, **params)
        f_np = broadcast_to_axis(
            f.to_value(u.dimensionless_unscaled),
            np.ndim(spectra),
            axis=axis,
        )
        return spectra * f_np


@dataclass
class DustScreenAttenuation(AttenuationModel):
    """
    Foreground dust screen attenuation model.

    This model applies a single attenuation curve to the full spectrum as a
    multiplicative factor.

    Parameters
    ----------
    curve : AttenuationCurve or str, optional
        Attenuation curve instance or name of a law in the extinction package.
        If a string is provided, an ExtinctionLibCurve is constructed.

    Notes
    -----
    The main parameter is a_v, interpreted as V band attenuation in magnitudes.
    Additional curve parameters such as r_v are forwarded to the curve.
    """
    curve: AttenuationCurve | str = "ccm89"
    name: str = "dust_screen"

    def __post_init__(self):
        """
        Construct an ExtinctionLibCurve when curve is given by nmiame.
        """
        if isinstance(self.curve, str):
            self.curve = ExtinctionLibCurve(name=self.curve)

    def attenuation_factor(self, wavelength: u.Quantity, *, a_v: float = 0.0, **params) -> u.Quantity:
        """
        Compute the attenuation factor for a foreground screen.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float, optional
            V band attenuation in magnitudes. Default is 0.0.
        **params
            Additional parameters forwarded to the underlying curve, such as
            r_v.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor with the same shape as wavelength.
        """
        return self.curve.attenuation_factor(wavelength, a_v=float(a_v), **params)


@dataclass
class CharlotFall00Attenuation(AttenuationModel):
    """
    Two component attenuation model inspired by Charlot and Fall 2000.

    This model defines two attenuation components applied to two stellar
    populations divided by an age threshold. A common usage is:

    - young population uses a_v_young
    - old population uses a_v_old
    - the split is defined by young_age

    This class returns two wavelength dependent attenuation factors, one for the
    young component and one for the old component. The caller is responsible for
    applying these factors to the corresponding spectral components.

    Parameters
    ----------
    curve : str or list of AttenuationCurve or list of str
        If a string is provided, both components use the same extinction package
        law. If a list is provided, it must contain two curves or two names.
    young_age : astropy.units.Quantity, optional
        Age threshold separating young and old populations. Default is 10 Myr.

    Notes
    -----
    The returned attenuation factor has shape (2, n_wave). The first entry
    corresponds to the young component and the second to the old component.
    """
    curve: str | List[AttenuationCurve] | List[str] = "ccm89"
    young_age: u.Quantity = 10.0 << u.Myr
    name: str = "CF00"

    def __post_init__(self):
        """
        Normalize curve inputs to a list of two AttenuationCurve instances.
        """
        if isinstance(self.curve, str):
            self.curve = 2 * [ExtinctionLibCurve(name=self.curve)]
        elif isinstance(self.curve, list):
            self.curve = [ExtinctionLibCurve(name=c) if isinstance(c, str) else c for c in self.curve]

    def attenuation_factor(
        self,
        wavelength: u.Quantity,
        *,
        a_v_young: float = 1.0,
        a_v_old: float = 0.3,
        **params,
    ) -> u.Quantity:
        """
        Compute attenuation factors for young and old components.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v_young : float, optional
            V band attenuation in magnitudes for the young component.
            Default is 1.0.
        a_v_old : float, optional
            V band attenuation in magnitudes for the old component.
            Default is 0.3.
        **params
            Additional parameters forwarded to each attenuation curve, such as
            r_v.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factors with shape (2, n_wave). The first
            row corresponds to the young component and the second row
            corresponds to the old component.
        """
        wavelength = check_unit(wavelength, u.AA)

        f_y = self.curve[0].attenuation_factor(
            wavelength, a_v=float(a_v_young), **params
        ).to_value(u.dimensionless_unscaled)

        f_o = self.curve[1].attenuation_factor(
            wavelength, a_v=float(a_v_old), **params
        ).to_value(u.dimensionless_unscaled)

        return np.array([f_y, f_o]) << u.dimensionless_unscaled


@dataclass
class Casey2012DustComponent(SedComponent):
    """
    Dust emission component based on Casey 2012.

    This component generates a dust emission spectrum as a sum of a modified
    blackbody component and a mid infrared power law component. The shape is
    normalized to match an integrated infrared luminosity over a configurable
    wavelength range.

    Parameters
    ----------
    optically_thin : bool, optional
        If True, uses an optically thin approximation for emissivity. Default is
        False.
    ir_range : tuple of astropy.units.Quantity, optional
        Wavelength range used for luminosity normalization. Default is 8 um to
        1000 um.
    default_unit : astropy.units.Unit, optional
        Output spectral density unit. Default is Lsun / AA.
    min_wavelength : astropy.units.Quantity, optional
        Rest frame cutoff. Emission below this wavelength is set to zero.
        Default is 1 um.

    Notes
    -----
    The method emission_spectrum expects a target integrated luminosity
    ``lum_ir``. Internally the code builds a shape template on the input wavelength
    grid and then normalizes it such that the integral of L_lambda over ir_range
    equals ``lum_ir``.

    The implementation assumes integrate_sed integrates L_lambda over wavelength
    and returns units of luminosity.

    See Also
    --------
    CalorimetricDustComponent : Dust emission coupled to absorbed stellar energy.
    """
    name: str = "Casey2012"
    optically_thin: bool = False
    ir_range: Tuple[u.Quantity, u.Quantity] = (8 << u.um, 1000 << u.um)
    default_unit = u.Lsun / u.AA
    min_wavelength = 1 << u.um

    def emission_spectrum(
        self,
        wavelength: u.Quantity,
        *,
        lum_ir: u.Quantity,
        t_dust: float = 35.0,
        beta: float = 1.5,
        alpha: float = 2.0,
        lam0: Optional[u.Quantity] = 200 << u.um,
        lam_pivot: Optional[u.Quantity] = None,
        **kwargs,
    ) -> u.Quantity:
        """
        Compute the dust emission spectrum.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the emission is evaluated.
        lum_ir : astropy.units.Quantity
            Target integrated infrared luminosity used for normalization.
            Expected units are luminosity, for example Lsun.
        t_dust : float, optional
            Dust temperature in Kelvin. Default is 35.0.
        beta : float, optional
            Emissivity index. Default is 1.5.
        alpha : float, optional
            Mid infrared power law slope. Default is 2.0.
        lam0 : astropy.units.Quantity or None, optional
            Optical depth scale wavelength. If provided, uses a finite optical
            depth emissivity. If None, uses an optically thin approximation.
            Default is 200 um.
        lam_pivot : astropy.units.Quantity or None, optional
            Pivot wavelength connecting the two components. If None, it is
            computed from t_dust and alpha.
        **kwargs
            Additional unused parameters for compatibility.

        Returns
        -------
        l_lambda : astropy.units.Quantity
            Dust emission spectrum in default_unit, sampled on wavelength.

        Raises
        ------
        ValueError
            If the template has zero integrated luminosity in the IR range and
            cannot be normalized.
        """
        lam = check_unit(wavelength, u.AA).to(u.um)
        T = check_unit(t_dust, u.K)

        if lam_pivot is None:
            lam_pivot = self._lambda_pivot(T, alpha).to(u.um)
        if lam0 is not None:
            lam0 = check_unit(lam0, u.um).to(u.um)

        mbb_lam, pl_lam = self._shape_l_lambda(
            lam=lam, t_dust=T, beta=beta, alpha=alpha, lam0=lam0, lam_pivot=lam_pivot
        )
        l_lambda_shape = mbb_lam + pl_lam

        l_lambda_shape[lam < self.min_wavelength.to(u.um)] = 0.0

        if lum_ir is not None:
            lum_ir = check_unit(lum_ir, u.Lsun)
            wl_min, wl_max = self.ir_range
            l_ir_norm = self.integrate_sed(
                lam, l_lambda_shape, wl_min.to(u.um), wl_max.to(u.um)
            )
            if l_ir_norm <= 0:
                raise ValueError(
                    "Dust template has zero integrated luminosity in the IR range; cannot normalize."
                )
            l_lambda_shape = l_lambda_shape * (lum_ir / l_ir_norm).decompose()

        return l_lambda_shape.to(self.default_unit, equivalencies=u.spectral_density(wavelength))

    def _shape_l_lambda(
        self,
        lam: u.Quantity,
        *,
        t_dust: u.Quantity,
        beta: float,
        alpha: float,
        lam0: Optional[u.Quantity] = 200 << u.um,
        lam_pivot: u.Quantity = None,
    ) -> Tuple[u.Quantity, u.Quantity]:
        """
        Build the unnormalized spectral shape template in L_lambda form.

        Parameters
        ----------
        lam : astropy.units.Quantity
            Wavelength grid in microns.
        t_dust : astropy.units.Quantity
            Dust temperature.
        beta : float
            Emissivity index.
        alpha : float
            Mid infrared power law slope.
        lam0 : astropy.units.Quantity or None, optional
            Optical depth scale wavelength. If None, uses an optically thin
            approximation.
        lam_pivot : astropy.units.Quantity
            Pivot wavelength connecting the components.

        Returns
        -------
        mbb : astropy.units.Quantity
            Modified blackbody shaped component. Absolute normalization is
            arbitrary.
        pl : astropy.units.Quantity
            Mid infrared power law shaped component. Absolute normalization is
            arbitrary.

        Notes
        -----
        The absolute normalization is arbitrary. The caller normalizes the sum
        to match lum_ir using integrate_sed.
        """
        bb = BlackBody(temperature=t_dust)
        Blam = bb(lam)

        if lam0 is not None:
            tau = (lam0 / lam) ** beta
            emiss = (-np.expm1(-tau)) * u.dimensionless_unscaled
        else:
            emiss = (lam_pivot / lam) ** beta * u.dimensionless_unscaled

        mbb = Blam * emiss

        pl_shape = (lam / lam_pivot) ** alpha * np.exp(-(lam / lam_pivot) ** 2)

        mbb_piv = (
            bb(lam_pivot)
            * (
                (-np.expm1(-((lam0 / lam_pivot) ** beta)) if lam0 is not None else 1.0)
            )
            * u.dimensionless_unscaled
        )

        pl = (mbb_piv * pl_shape).to(mbb.unit)
        return mbb, pl

    def _lambda_pivot(self, t_dust: u.Quantity, alpha: float) -> u.Quantity:
        """
        Compute the pivot wavelength used to connect template components.

        Parameters
        ----------
        t_dust : astropy.units.Quantity
            Dust temperature.
        alpha : float
            Mid infrared power law slope.

        Returns
        -------
        lam_pivot : astropy.units.Quantity
            Pivot wavelength in microns.
        """
        b1, b2, b3, b4 = 26.68, 6.246, 1.905e-4, 7.243e-5
        lam_c_um = 0.75 / (
            ((b1 + b2 * alpha) ** -2) + (b3 + b4 * alpha) * t_dust.to_value(u.K)
        )
        return lam_c_um * u.um


@dataclass
class CalorimetricDustComponent(SedComponent):
    """
    Calorimetric dust emission component.

    This component couples an attenuation model and a dust emission component
    using energy balance. It computes the absorbed stellar luminosity and uses
    it as the infrared luminosity normalization of the dust emission model.

    Parameters
    ----------
    attenuation : AttenuationModel
        Attenuation model used to compute the attenuated stellar spectrum.
    dust : SedComponent
        Dust emission component that accepts lum_ir as a normalization parameter.
    default_unit : astropy.units.Unit, optional
        Output spectral density unit. Default is Lsun / AA.
    ir_range : tuple of astropy.units.Quantity, optional
        IR wavelength range used by the dust component. Default is 8 um to
        1000 um.

    Notes
    -----
    The method emission_spectrum returns three spectra:

    - Lsrc: intrinsic stellar spectrum
    - Latt: attenuated stellar spectrum
    - Ldust: dust emission spectrum normalized to absorbed energy

    For CharlotFall00Attenuation, the method expects the stellar source to
    return a binned output with shape (n_bins, n_wave), where the bins represent
    young and old populations.

    See Also
    --------
    CharlotFall00Attenuation : Two component attenuation model returning two factors.
    Casey2012DustComponent : Dust emission template normalized by lum_ir.
    """

    attenuation: AttenuationModel
    dust: SedComponent
    default_unit = u.Lsun / u.AA
    ir_range: Tuple[u.Quantity, u.Quantity] = (8 * u.um, 1000 * u.um)

    def emission_spectrum(
        self,
        wavelength: u.Quantity,
        source: StellarComponent,
        source_params: dict = None,
        **params,
    ) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:
        """
        Compute intrinsic, attenuated, and dust emission spectra.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where spectra are evaluated.
        source : StellarComponent
            Stellar emission source.
        source_params : dict, optional
            Parameters forwarded to the stellar source emission_spectrum.
        **params
            Parameters forwarded to the attenuation model and the dust emission
            component. For example a_v or a_v_young and a_v_old.

        Returns
        -------
        Lsrc : astropy.units.Quantity
            Intrinsic stellar spectrum in default_unit.
        Latt : astropy.units.Quantity
            Attenuated stellar spectrum in default_unit.
        Ldust : astropy.units.Quantity
            Dust emission spectrum in default_unit normalized to absorbed energy.

        Raises
        ------
        ValueError
            If CharlotFall00Attenuation is used and the stellar source does not
            return a 2D binned spectrum with shape (n_bins, n_wave).
        """
        lam = check_unit(wavelength, u.AA)

        f = self.attenuation.attenuation_factor(lam, **params)

        if source_params is None:
            source_params = {}

        if isinstance(self.attenuation, CharlotFall00Attenuation):
            source_params["age_bin_edges"] = u.Quantity(
                [
                    0,
                    self.attenuation.young_age.value,
                    (15 << u.Gyr).to_value(self.attenuation.young_age.unit),
                ],
                self.attenuation.young_age.unit,
            )

            Lsrc = source.emission_spectrum(lam, **source_params).to(self.default_unit)

            if Lsrc.ndim != 2:
                raise ValueError(
                    "CF00 requires binned stellar SED output with shape (n_bins, n_wave)."
                )

            Latt_bins = Lsrc * f
            Labs_spec = np.sum(Lsrc - Latt_bins, axis=0)
            Latt = np.sum(Latt_bins, axis=0)

        else:
            Lsrc = source.emission_spectrum(lam, **source_params).to(self.default_unit)
            Latt = Lsrc * f
            Labs_spec = Lsrc - Latt

        Labs = self.integrate_sed(lam, Labs_spec)
        Ldust = self.dust.emission_spectrum(lam, lum_ir=Labs, **params).to(self.default_unit)

        return Lsrc, Latt, Ldust


# Legacy code for backwards compatibility

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


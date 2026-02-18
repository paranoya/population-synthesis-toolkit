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
from pst.utils import check_unit, check_parameter, broadcast_to_axis
from pst.sed import SedComponent, StellarComponent

## Utils ###

_LOG_FLOAT_MAX = np.log(np.finfo(float).max)


def _exp_bounded(exponent: np.ndarray) -> np.ndarray:
    """Exponentiate with clipping to float64 finite range."""
    exp_arg = np.asarray(exponent, dtype=float)
    return np.exp(np.clip(exp_arg, -_LOG_FLOAT_MAX, _LOG_FLOAT_MAX))


def _pow_positive_bounded(base: np.ndarray, exponent: float) -> np.ndarray:
    """Compute base**exponent for positive bases using a log-space bounded form."""
    base_arr = np.asarray(base, dtype=float)
    base_arr = np.clip(base_arr, np.finfo(float).tiny, np.finfo(float).max)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_base = np.log(base_arr)
    return _exp_bounded(exponent * log_base)


def modified_blackbody(
    lam: u.Quantity,
    T: u.Quantity,
    beta: float,
    lam_0: Optional[u.Quantity] = None,
    lam_ref: u.Quantity = 100 * u.um,
    per_freq: bool = False,
) -> u.Quantity:
    """
    Compute a modified blackbody spectrum.

    This function returns a spectral shape given by a blackbody multiplied by
    an emissivity term. Two emissivity parameterizations are supported.

    If lam_0 is provided, the emissivity uses a finite optical depth form

    .. math::
        S_{nu} \\propto (1 - \\exp(-\\tau_{nu})) B_{nu}(T)

    with

    .. math::
        \\tau_{nu} = \\left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta}
        = \\left(\\frac{\\lambda_0}{\\lambda}\\right)^{\\beta}

    If lam_0 is None, the emissivity uses an optically thin approximation

    .. math::
        S_{nu} \\propto \\left(\\frac{\\nu}{\\nu_{ref}}\\right)^{\\beta} B_{nu}(T)

    Parameters
    ----------
    lam : astropy.units.Quantity
        Wavelength grid. Internally converted to microns.
    T : astropy.units.Quantity
        Dust temperature. Internally converted to Kelvin.
    beta : float
        Emissivity index.
    lam_0 : astropy.units.Quantity, optional
        Turnover wavelength where tau equals 1. If provided, the finite optical
        depth emissivity term is used.
    lam_ref : astropy.units.Quantity, optional
        Reference wavelength used to normalize the optically thin emissivity term.
        Only used when lam_0 is None. Default is 100 micron.
    per_freq : bool, optional
        If True, return a per-frequency spectrum proportional to B_nu.
        If False, return a per-wavelength spectrum using a spectral density
        equivalency. Default is False.

    Returns
    -------
    mbb : astropy.units.Quantity
        Modified blackbody spectrum evaluated on lam. The normalization is not set
        by this function. If per_freq is True the result is proportional to B_nu(T).
        If per_freq is False the result is returned as a per-wavelength density.

    Notes
    -----
    This function is typically used to build a template shape that is later
    normalized to an integrated luminosity.

    The per-wavelength conversion uses astropy spectral density equivalencies.
    """
    lam = check_unit(lam, u.um)
    T = check_unit(T, u.K)
    lam_ref = check_unit(lam_ref, u.um)

    nu = (const.c / lam).to(u.Hz)
    bb = BlackBody(temperature=T)
    # BlackBody evaluation is stable but can raise overflow warnings internally
    # for very large h nu / kT values where the limiting value is zero.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        Bnu = bb(nu)

    if lam_0 is not None:
        lam_0 = check_unit(lam_0, u.um).to(u.um)
        tau = _pow_positive_bounded(
            (lam_0 / lam).to_value(u.dimensionless_unscaled), beta
        )
        factor = -np.expm1(-tau) * u.dimensionless_unscaled
    else:
        nu_ref = (const.c / lam_ref).to(u.Hz)
        factor = _pow_positive_bounded(
            (nu / nu_ref).to_value(u.dimensionless_unscaled), beta
        ) * u.dimensionless_unscaled

    if per_freq:
        return Bnu * factor

    # Convert from per-frequency radiance-like shape to per-wavelength density
    return (Bnu * factor).to("erg / (Angstrom s sr cm2)", u.spectral_density(lam))


### Attenuation curve (wavelength dependence) ###


class AttenuationCurve(ModelBase):
    """
    Base class for attenuation curve shapes.

    This class defines the wavelength dependence of an attenuation law via
    a dimensionless curve k_lambda. The normalization (for example an A_V value)
    is applied separately by scaling k_lambda.

    Subclasses must implement k_lambda.

    Notes
    -----
    This interface intentionally separates shape from normalization. The helper
    methods provide common conversions:

    - a_lambda: convert k_lambda to attenuation in magnitudes given a_v
    - tau_lambda: convert magnitudes to optical depth using A = 1.086 tau
    - attenuation_factor: compute 10**(-0.4 A_lambda)

    Attributes
    ----------
    name : str
        Curve identifier.
    """

    name: str = "attenuation_curve"

    @abstractmethod
    def k_lambda(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Return the dimensionless attenuation curve shape.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the curve is evaluated.
        **params
            Optional call time parameters. Most implementations store curve
            parameters as Parameter attributes, so this is often unused.

        Returns
        -------
        k : astropy.units.Quantity
            Dimensionless curve with the same shape as wavelength.
        """
        raise NotImplementedError

    def a_lambda(
        self, wavelength: u.Quantity, *, a_v: u.Quantity, **params
    ) -> u.Quantity:
        """
        Convert curve shape to attenuation in magnitudes.

        The attenuation is computed as

        .. math::
            A_{\\lambda} = A_{v} k_{\\lambda}

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : astropy.units.Quantity
            Attenuation normalization in magnitudes. The interpretation depends
            on the curve convention. Commonly, a_v corresponds to attenuation
            at the pivot wavelength used to normalize k_lambda.
        **params
            Parameters forwarded to k_lambda.

        Returns
        -------
        A_lambda : astropy.units.Quantity
            Attenuation in magnitudes with the same shape as wavelength.
        """
        a_v = check_unit(a_v, u.mag)
        k = self.k_lambda(wavelength, **params).to_value(u.dimensionless_unscaled)
        return (a_v.to_value(u.mag) * k) << u.mag

    def tau_lambda(
        self, wavelength: u.Quantity, *, a_v: float = None, **params
    ) -> u.Quantity:
        """
        Convert attenuation in magnitudes to optical depth.

        Uses the relation

        .. math::
            A_{\\lambda} = 1.086 \\tau_{\\lambda}

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the optical depth is evaluated.
        a_v : astropy.units.Quantity
            Attenuation normalization in magnitudes.
        **params
            Parameters forwarded to a_lambda.

        Returns
        -------
        tau : astropy.units.Quantity
            Optical depth as a dimensionless quantity with the same shape as
            wavelength.
        """
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        return (a_lam / 1.086) << u.dimensionless_unscaled

    def attenuation_factor(
        self, wavelength: u.Quantity, *, a_v: u.Quantity, **params
    ) -> u.Quantity:
        """
        Compute the multiplicative attenuation factor.

        The factor is defined as

        .. math::
            f_{\\lambda} = 10^{-0.4 A_{\\lambda}}

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : astropy.units.Quantity
            Attenuation normalization in magnitudes.
        **params
            Parameters forwarded to a_lambda.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor with the same shape as wavelength.
        """
        a_lam = self.a_lambda(wavelength, a_v=a_v, **params).to_value(u.mag)
        exponent = -0.4 * np.log(10.0) * a_lam
        return _exp_bounded(exponent) << u.dimensionless_unscaled


@dataclass(kw_only=True)
class PowerLawAttenuationCurve(AttenuationCurve):
    """
    Power law attenuation curve shape.

    This curve defines the dimensionless shape

    .. math::
        k_{\\lambda} = \\left(\\frac{\\lambda}{\\lambda_{pivot}}\\right)^{\\alpha}

    where lambda_pivot is the pivot wavelength and alpha is the power law exponent.

    Parameters
    ----------
    name : str, optional
        Curve identifier.
    alpha : pst.model.Parameter
        Power law exponent. Commonly negative for dust attenuation.
    pivot : pst.model.Parameter
        Pivot wavelength used to normalize the curve so that k_lambda at pivot is 1.

    Notes
    -----
    This class returns k_lambda. The normalization A_v is applied by
    AttenuationCurve.a_lambda or by an AttenuationModel.

    See Also
    --------
    AttenuationCurve
        Base interface for attenuation curve shapes.
    """

    name: str = "powerlaw_attenuation_curve"
    alpha: Parameter = field(
        default_factory=lambda: Parameter(
            -0.7,
            unit=u.dimensionless_unscaled,
            doc="Attenuation slope alpha in k(lambda) = (lambda/lambda_pivot)^alpha",
        )
    )
    pivot: Parameter = field(
        default_factory=lambda: Parameter(
            5500.0,
            unit=u.AA,
            doc="Reference wavelength where the attenuation curve is normalized to unity",
        )
    )

    def __post_init__(self):
        """
        Validate curve parameters.

        Raises
        ------
        ValueError
            If alpha is positive.
        """
        if self.alpha.to_value() > 0:
            raise ValueError("alpha must be negative")

    def k_lambda(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Evaluate the curve shape.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the curve is evaluated.
        **params
            Optional call time parameters. Reserved for future extensions.

        Returns
        -------
        k : astropy.units.Quantity
            Dimensionless curve with the same shape as wavelength.
        """
        lam = check_unit(wavelength, u.AA)
        pivot = check_unit(self.pivot.q, u.AA)
        alpha = float(self.alpha.to_value())
        k = _pow_positive_bounded(
            (lam / pivot).to_value(u.dimensionless_unscaled), alpha
        )
        return k << u.dimensionless_unscaled


@dataclass(kw_only=True)
class ExtinctionLibCurve(AttenuationCurve):
    """
    Attenuation curve shape based on the extinction package.

    This class wraps an extinction law implemented by the extinction package
    and returns a dimensionless curve k_lambda computed from the law evaluated
    at A_v equals 1 magnitude.

    Parameters
    ----------
    name : str, optional
        Curve identifier.
    law : str, optional
        Name of the extinction law function in the extinction package.
    r_v : pst.model.Parameter
        Total to selective extinction ratio.

    Raises
    ------
    ValueError
        If the requested law is not present in the extinction package.

    Notes
    -----
    The extinction package functions typically accept wavelength, a_v, and r_v
    and return A_lambda in magnitudes. This class calls the selected law with
    a_v set to 1 and returns the resulting A_lambda as the dimensionless curve
    shape.

    If you want k_lambda normalized such that k_lambda at a pivot wavelength is 1,
    normalize externally using a_v or implement a pivot normalization in this class.

    See Also
    --------
    AttenuationCurve
        Base interface for attenuation curve shapes.
    """

    name: str = "extinction_lib"
    law: str = "ccm89"
    r_v: Parameter = field(
        default_factory=lambda: Parameter(
            3.1,
            unit=u.dimensionless_unscaled,
            doc="Total-to-selective extinction ratio R_V used by the extinction law",
        )
    )

    def __post_init__(self):
        """
        Resolve the extinction law from the extinction package.

        Raises
        ------
        ValueError
            If the law is not found.
        """
        try:
            self._law = getattr(_extinction_lib, self.law)
        except AttributeError as e:
            raise ValueError(
                f"Unknown extinction law '{self.law}' in extinction package."
            ) from e

    def k_lambda(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Evaluate the curve shape using the selected extinction law.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the curve is evaluated.
        **params
            Optional call time parameters. Reserved for future extensions.

        Returns
        -------
        k : astropy.units.Quantity
            Dimensionless curve with the same shape as wavelength.
        """
        lam = check_unit(wavelength, u.AA).to_value(u.AA)
        r_v = self.r_v.to_value()
        # compute A_lambda for a_v=1 mag
        k = self._law(lam, 1.0, r_v)  # magnitudes
        return k << u.dimensionless_unscaled


class AttenuationModel(ModelBase, ABC):
    """
    Base class for geometry dependent attenuation models.

    An attenuation model returns a multiplicative attenuation factor that can be
    applied to spectra. Models represent different geometries such as a
    foreground screen or multi component prescriptions.

    Subclasses must implement attenuation_factor.

    Notes
    -----
    The apply method multiplies the input spectra by the attenuation factor,
    broadcasting the factor along the wavelength axis.

    Attributes
    ----------
    name : str
        Model identifier.
    """

    name: str = "attenuation_model"

    @abstractmethod
    def attenuation_factor(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Compute the multiplicative attenuation factor.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation factor is evaluated.
        **params
            Optional call time parameters. Most implementations store parameters
            as Parameter attributes, so this is often unused.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor. The shape is model dependent.
            For a single screen this is (n_wave,). For a multi component model
            this may be (n_comp, n_wave).
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
            Input spectra to be attenuated.
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
        If attenuation_factor returns multiple components, the returned factor
        cannot be directly applied to a single spectrum. In that case, the caller
        should select or combine components before calling apply.
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

    This model applies a single attenuation curve with a single normalization
    parameter A_v.

    Parameters
    ----------
    name : str, optional
        Model identifier.
    curve : AttenuationCurve, optional
        Attenuation curve shape used by the model.
    a_v : pst.model.Parameter
        Attenuation normalization in magnitudes.

    Notes
    -----
    The attenuation factor is computed as

    .. math::
        f_{\\lambda} = 10^{-0.4 A_{v} k_{\\lambda}}

    where k_lambda is provided by the curve.

    See Also
    --------
    AttenuationCurve
        Curve interface used to compute k_lambda.
    """

    name: str = "dust_screen"
    curve: AttenuationCurve = field(
        default_factory=lambda: ExtinctionLibCurve(law="ccm89")
    )
    a_v: Parameter = field(
        default_factory=lambda: Parameter(
            0.0,
            unit=u.mag,
            vrange=(0.0 * u.mag, np.inf * u.mag),
            doc="V-band attenuation normalization A_V for the foreground dust screen",
        )
    )

    def __post_init__(self):
        """
        Normalize the curve input.

        Notes
        -----
        If curve is provided as a string, it is interpreted as an extinction
        package law name and converted into an ExtinctionLibCurve.
        """
        self.a_v = check_parameter(
            self.a_v, u.mag, doc="V-band attenuation normalization A_V for the foreground dust screen"
        )

        if isinstance(self.curve, str):
            self.curve = ExtinctionLibCurve(name=self.curve)

    def attenuation_factor(
        self, wavelength: u.Quantity, *, a_v: float = None, **params
    ) -> u.Quantity:
        """
        Compute the attenuation factor for a foreground screen.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid where the attenuation is evaluated.
        a_v : float, optional
            Unused. The model uses the internal Parameter a_v. This argument is
            kept for backward compatibility with older call sites.
        **params
            Additional parameters forwarded to the curve. Reserved for future
            extensions.

        Returns
        -------
        factor : astropy.units.Quantity
            Dimensionless attenuation factor with the same shape as wavelength.
        """
        a_v = self.a_v.q
        return self.curve.attenuation_factor(wavelength, a_v=a_v)


@dataclass(kw_only=True)
class CharlotFall00Attenuation(AttenuationModel):
    """
    Two component attenuation model inspired by Charlot and Fall 2000.

    The model returns two attenuation factors corresponding to young and old
    stellar populations.

    Parameters
    ----------
    name : str, optional
        Model identifier.
    curve_young : AttenuationCurve
        Curve used for young component.
    curve_old : AttenuationCurve
        Curve used for old component.
    a_v_young : pst.model.Parameter
        Normalization for young component in magnitudes.
    a_v_old : pst.model.Parameter
        Normalization for old component in magnitudes.
    young_age : pst.model.Parameter
        Age threshold used by callers that split stellar emission into components.

    Returns
    -------
    factor : astropy.units.Quantity
        Dimensionless array with shape (2, n_wave). The first row corresponds to
        young component and the second row corresponds to old component.

    Notes
    -----
    This method returns factors only. The logic that splits stellar emission
    into young and old components is implemented elsewhere, for example in the
    stellar emission component or in a calorimetric dust model.

    See Also
    --------
    CalorimetricDustComponent
        Uses the returned factors to compute absorbed luminosity by component.
    """

    name: str = "CF00"
    curve_young: AttenuationCurve = field(
        default_factory=lambda: PowerLawAttenuationCurve()
    )
    curve_old: AttenuationCurve = field(
        default_factory=lambda: PowerLawAttenuationCurve()
    )
    a_v_young: Parameter = field(
        default_factory=lambda: Parameter(
            1.0,
            unit=u.mag,
            vrange=(0.0 << u.mag, np.inf << u.mag),
            doc="V-band attenuation A_V applied to the young stellar component",
        )
    )
    a_v_old: Parameter = field(
        default_factory=lambda: Parameter(
            0.3,
            unit=u.mag,
            vrange=(0.0 << u.mag, np.inf << u.mag),
            doc="V-band attenuation A_V applied to the old stellar component",
        )
    )
    young_age: Parameter = field(
        default_factory=lambda: Parameter(
            10.0,
            unit=u.Myr,
            doc="Age threshold separating young and old stellar populations",
        )
    )

    def __post_init__(self):
        self.a_v_young = check_parameter(
            self.a_v_young,
            u.mag,
            doc="V-band attenuation A_V applied to the young stellar component",
        )
        self.a_v_old = check_parameter(
            self.a_v_old,
            u.mag,
            doc="V-band attenuation A_V applied to the old stellar component",
        )

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
        lam = check_unit(wavelength, u.AA)
        fy = self.curve_young.attenuation_factor(lam, a_v=self.a_v_young.q).to_value(
            u.dimensionless_unscaled
        )
        fo = self.curve_old.attenuation_factor(lam, a_v=self.a_v_old.q).to_value(
            u.dimensionless_unscaled
        )
        return np.stack([fy, fo], axis=0) << u.dimensionless_unscaled


@dataclass(kw_only=True)
class Casey2012DustComponent(SedComponent):
    """
    Dust emission component inspired by Casey 2012.

    This component builds an unnormalized template shape as the sum of a modified
    blackbody and a mid infrared power law. The shape is then normalized so that
    the integrated luminosity over a configurable infrared wavelength range
    matches lum_ir.

    Parameters
    ----------
    name : str, optional
        Component identifier.
    t_dust : pst.model.Parameter
        Dust temperature.
    beta : pst.model.Parameter
        Emissivity index.
    alpha : pst.model.Parameter
        Mid infrared power law exponent.
    lam0 : pst.model.Parameter
        Optical depth scale wavelength.
    min_wavelength : astropy.units.Quantity, optional
        Emission below this wavelength is set to zero.
    ir_range : tuple of astropy.units.Quantity, optional
        Integration bounds used to normalize the template.
    default_unit : astropy.units.Unit, optional
        Output spectral density unit.

    Notes
    -----
    The output is a spectral density in default_unit. The overall normalization
    is set by lum_ir.

    See Also
    --------
    modified_blackbody
        Function used to compute the modified blackbody component.
    """

    name: str = "Casey2012"

    t_dust: Parameter = field(
        default_factory=lambda: Parameter(
            35.0,
            unit=u.K,
            vrange=(1.0 << u.K, 200.0 << u.K),
            doc="Characteristic dust temperature of the modified blackbody component",
        )
    )
    beta: Parameter = field(
        default_factory=lambda: Parameter(
            1.5,
            unit=u.dimensionless_unscaled,
            vrange=(0.5, 3.0),
            doc="Dust emissivity spectral index beta",
        )
    )
    alpha: Parameter = field(
        default_factory=lambda: Parameter(
            2.0,
            unit=u.dimensionless_unscaled,
            vrange=(0.1, 3.0),
            doc="Mid-infrared power-law slope for the warm dust tail",
        )
    )
    lam0: Parameter = field(
        default_factory=lambda: Parameter(
            200.0,
            unit=u.um,
            vrange=(1.0 << u.um, 1e6 << u.um),
            doc="Wavelength where optical depth reaches unity in the modified blackbody term",
        )
    )
    min_wavelength: u.Quantity = 1.0 << u.um
    ir_range: Tuple[u.Quantity, u.Quantity] = (8.0 << u.um, 1000.0 << u.um)
    default_unit = u.Lsun / u.AA

    def emission_spectrum(
        self,
        wavelength: u.Quantity,
        *,
        lum_ir: u.Quantity,
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
        T = self.t_dust.q.to(u.K)
        beta = float(self.beta.to_value())
        alpha = float(self.alpha.to_value())
        lam0 = self.lam0.q.to(u.um) if self.lam0 is not None else None

        if lam_pivot is None:
            lam_pivot = self._lambda_pivot(T, alpha).to(u.um)

        # shape template in L_lambda-like units (arbitrary normalization)
        mbb = modified_blackbody(lam, T, beta, lam_0=lam0, per_freq=False)
        lam_ratio = (lam / lam_pivot).to_value(u.dimensionless_unscaled)
        pl = _pow_positive_bounded(lam_ratio, alpha) * _exp_bounded(
            -_pow_positive_bounded(lam_ratio, 2.0)
        )
        pl = mbb[np.argmin(np.abs(lam - lam_pivot))] * pl  # match scale near pivot

        shape = mbb + pl
        shape[lam < self.min_wavelength.to(u.um)] = 0.0

        # normalize to lum_ir
        lum_ir = check_unit(lum_ir, u.Lsun)
        wl_min, wl_max = self.ir_range
        L_ir = self.integrate_sed(lam, shape, wl_min.to(u.um), wl_max.to(u.um))
        if L_ir <= 0:
            raise ValueError(
                "Dust template has zero integrated luminosity in IR range; cannot normalize."
            )
        shape = shape * (lum_ir / L_ir).decompose()

        return shape.to(self.default_unit, equivalencies=u.spectral_density(wavelength))

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
        return lam_c_um << u.um


@dataclass
class CalorimetricDustComponent(SedComponent):
    """
    Calorimetric dust emission component.

    This component couples an attenuation model and a dust emission component
    using energy balance. It computes the stellar luminosity absorbed by dust
    and uses it as the infrared luminosity normalization for the dust emission
    component.

    Parameters
    ----------
    attenuation : AttenuationModel
        Attenuation model used to compute attenuation factors.
    dust_sed_component : SedComponent
        Dust emission component that accepts lum_ir as normalization.
    default_unit : astropy.units.Unit, optional
        Output spectral density unit.
    name : str, optional
        Component identifier.

    Returns
    -------
    Lsrc : astropy.units.Quantity
        Intrinsic stellar spectrum.
    Latt : astropy.units.Quantity
        Attenuated stellar spectrum.
    Ldust : astropy.units.Quantity
        Dust emission spectrum normalized to absorbed luminosity.

    Notes
    -----
    If the attenuation model returns multiple factors (for example two component
    CF00), the stellar source must return a matching binned output.

    See Also
    --------
    CharlotFall00Attenuation
        Two component attenuation model.
    Casey2012DustComponent
        Dust emission model normalized by lum_ir.
    """

    attenuation: AttenuationModel
    dust_sed_component: SedComponent
    default_unit = u.Lsun / u.AA
    name: str = "calorimetric_dust_sed"

    def emission_spectrum(
        self,
        wavelength: u.Quantity,
        *,
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
        if source_params is None:
            source_params = {}

        f = self.attenuation.attenuation_factor(lam, **params)

        if isinstance(self.attenuation, CharlotFall00Attenuation):
            # request binned output for CF00
            source_params["age_bin_edges"] = u.Quantity(
                [0, self.attenuation.young_age.to_value(u.Myr), 15e3], u.Myr
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
        Ldust = self.dust_sed_component.emission_spectrum(
            lam, lum_ir=Labs, **params
        ).to(self.default_unit)

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
            ssp_model.wavelength, reddened_ssp_model.L_lambda, axis=-1, **kwargs
        )
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
        return (
            _exp_bounded(
                -0.4
                * np.log(10.0)
                * self.extinction_law(
                    np.array(wavelength.to_value("angstrom"), dtype=float),
                    a_v,
                    self.r_extinction,
                )
            )
            << u.dimensionless_unscaled
        )

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
        assert isinstance(
            young_ssp_age, u.Quantity
        ), "young_ssp_age must be an astropy.Quantity"
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

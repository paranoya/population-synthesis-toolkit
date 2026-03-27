from __future__ import annotations
from typing import Any, Dict, Mapping, List, Union

import numpy as np

from astropy import units as u
from astropy.cosmology import WMAP9

from pst.utils import check_parameter, flux_conserving_interpolation
from pst import observables

from pst.sed import SedComponent
from pst.dust import AttenuationModel, CalorimetricDustComponent
from pst.model import Parameter

class GalaxySED(SedComponent):
    """
    Composite galaxy SED model (stellar emission + attenuation + dust emission).

    This component combines:
    - a stellar emission component (typically driven by a CEM/SSP model),
    - an optional attenuation model applied to the stellar emission, and
    - an optional dust emission component (optionally calorimetric / energy-balance).

    Parameters are handled via :class:`pst.model.Parameter` objects and may be
    updated at runtime using dotted parameter paths (see :meth:`build_param_index`
    and :meth:`update_parameters`), enabling integration with samplers/fitters
    such as BESTA.

    Notes
    -----
    - The internal spectral grid is ``target_wavelength`` in the rest frame.
    - If ``to_obs_frame=True``, spectra are converted to observed-frame fluxes
      using luminosity distance and redshift and resampled onto ``lambda_obs``.
    """
    name: str = "galaxy_sed"
    rest_unit = u.Lsun / u.AA  # Rest-frame default units
    obs_unit = u.erg / (u.s * u.cm**2 * u.AA)  # Observed-frame default units

    @property
    def redshift(self) -> Parameter:
        """Cosmological redshift parameter used for observation-frame mapping."""
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        """
        Set redshift and update derived cosmology-dependent quantities.

        This updates ``t_obs``, luminosity distance, and ``distance_factor``.
        """
        value = check_parameter(
            value,
            u.dimensionless_unscaled,
            doc="Cosmological redshift of the model galaxy",
        )
        z = np.asarray(value.to_value(u.dimensionless_unscaled), dtype=float)
        if not np.all(np.isfinite(z)):
            raise ValueError("redshift must be finite.")
        if np.any(z < 0.0):
            raise ValueError("redshift must be non-negative.")
        self._redshift = value
        self.t_obs = self.cosmology.age(self._redshift.q)
        # Ensure that the stellar SFH is set to same today
        if self.stars is not None:
            self.stars.sfh.today = self.t_obs
        self.dl = self.cosmology.luminosity_distance(
            self._redshift.q).clip(10 << u.pc)
        self.distance_factor = 4 * np.pi * self.dl.to("cm")**2 * (
            1 + self._redshift.q)

    def __init__(self, *,
                 stellar_model: SedComponent=None,
                 dust_attenuation_model: AttenuationModel=None,  # Fix typing
                 dust_model: SedComponent=None,
                 target_wavelength: u.Quantity=None,
                 redshift: Union[Parameter, float]=0.0,
                 cosmology=None,
                 filters: observables.FilterList=None):
        print("Initialising Galaxy model")
        self._param_index: Dict[str, Parameter] = {}
        # Setup components
        self.stars = stellar_model
        self.dust_attenuation = dust_attenuation_model
        self.dust = dust_model
        # Target wavelength range
        if target_wavelength is not None:
            self.target_wavelength = target_wavelength
        else:
            if self.dust is None:
                self.target_wavelength = self.stars.ssp.wavelength
            else:
                if self.stars is not None:
                    dust_wl = np.geomspace(1, 1000, 100) << u.um
                    wl = np.concatenate(
                        (self.stars.ssp.wavelength.to_value(u.AA),
                        dust_wl.to_value(u.AA)))
                else:
                    wl = np.geomspace(0.1, 1e3, 1000) << u.um
                self.target_wavelength = np.unique(wl) << u.AA
            print(" - Target wavelength range",
                  self.target_wavelength[[0, -1]],
                  "\n - No. pixels:", self.target_wavelength.size)
        # Interpolate components libraries to target wavelength
        if self.stars is not None:
            self.stars.ssp.interpolate_sed(self.target_wavelength)
        # Setup component transformers
        self.energy_balance = True if isinstance(
            self.dust, CalorimetricDustComponent) else False
        # Setup observation properties
        if cosmology is None:
            self.cosmology = WMAP9
        else:
            self.cosmology = cosmology
        self.redshift = redshift
        # Setup observables
        if filters is not None:
            print("Interpolating input filters to target wavelength")
            filters.interpolate(self.target_wavelength)
            self.filters = filters

    def build_param_index(self, *, include_fixed: bool = True, prefix: str = "") -> Dict[str, Parameter]:
        """
        Build and cache a mapping from parameter path to Parameter object.

        Parameters
        ----------
        include_fixed : bool, optional
            If True include fixed parameters in the index.
        prefix : str, optional
            Optional path prefix.

        Returns
        -------
        index : dict
            Mapping from dotted parameter path to Parameter.
        """
        self._param_index = self.parameters_recursive(prefix=prefix, include_fixed=include_fixed)
        return self._param_index

    def update_parameters(self, values: Mapping[str, Any], *, validate: bool = True, strict: bool = True) -> None:
        """
        Update parameters from a dict.

        Parameters
        ----------
        values : mapping
            Mapping from dotted parameter path to value. Values may be floats
            or Quantities.
        validate : bool, optional
            If True, checks fixed and vrange constraints.
        strict : bool, optional
            If True, unknown keys raise KeyError. If False, unknown keys are ignored.

        Notes
        -----
        This uses a cached parameter index when available. If the cache has not
        been built, it is built on the fly.
        """
        if not self._param_index:
            self.build_param_index(include_fixed=True)

        for path, val in values.items():
            p = self._param_index.get(path, None)
            if p is None:
                if strict:
                    raise KeyError(f"Unknown parameter path '{path}'")
                continue
            p.set(val, validate=validate)

    def emission_components(self, stars_params=None, dust_att_params=None,
                            dust_em_params=None):
        """
        Compute individual SED components on the rest-frame wavelength grid.

        Returns a dict with (when available):
        - ``stellar_sed_unatt`` : intrinsic stellar emission (no attenuation)
        - ``stellar_sed``       : attenuated stellar emission (if attenuation model is set)
        - ``dust_sed``          : dust emission component (if present)

        If ``self.dust`` is calorimetric (energy-balance), the dust component is
        computed consistently from the absorbed stellar energy.
        """
        components = {
            "stellar_sed": None,
            "stellar_sed_unatt": None,
            "dust_sed": None,
            # "nebular_sed": None,
                      }

        stars_params = stars_params or {}
        stars_params["t_obs"] = self.t_obs
        dust_att_params = dust_att_params or {}
        dust_em_params = dust_em_params or {}

        if self.energy_balance:
            stellar_sed_unatt, stellar_sed, dust_sed = self.dust.emission_spectrum(
                self.target_wavelength,
                source=self.stars,
                source_params=stars_params,
                **{**dust_att_params, **dust_em_params}
                )
            components["stellar_sed"] = stellar_sed.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            components["stellar_sed_unatt"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            components["dust_sed"] = dust_sed.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
        else:
            if self.stars is not None:
                stellar_sed_unatt = self.stars.emission_spectrum(
                    self.target_wavelength, **stars_params)
                components["stellar_sed_unatt"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            if self.dust_attenuation is not None:
                att_factor = self.dust_attenuation.attenuation_factor(
                    self.target_wavelength, **dust_att_params)
                stellar_sed = stellar_sed_unatt * att_factor
                components["stellar_sed"] = stellar_sed.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            else:
                components["stellar_sed"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))

            if self.dust is not None:
                dust_sed = self.dust.emission_spectrum(
                    self.target_wavelength, **dust_em_params)
                components["dust_sed"] = dust_sed.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))

        return components

    def emission_spectrum(self,
                          stars_params=None, dust_att_params=None,
                          dust_em_params=None, to_obs_frame=False):
        """
        Compute the composite SED (stellar + dust).

        Parameters
        ----------
        stars_params : dict, optional
            Parameters forwarded to the stellar component.
        dust_att_params : dict, optional
            Parameters forwarded to the attenuation model.
        dust_em_params : dict, optional
            Parameters forwarded to the dust emission component.
        to_obs_frame : bool
            If True, returns observed-frame flux density and shifts/resamples the
            spectrum to ``lambda_obs = (1+z) lambda_rest``.

        Returns
        -------
        sed : astropy.units.Quantity
            Composite spectrum in ``rest_unit`` (rest frame) or ``obs_unit``
            (observed frame when ``to_obs_frame=True``).
        """
        components = self.emission_components(
            stars_params=stars_params,
            dust_att_params=dust_att_params,
            dust_em_params=dust_em_params)

        composite_sed = np.zeros(self.target_wavelength.size) << self.rest_unit
        if components["stellar_sed"] is not None:
            composite_sed += components["stellar_sed"]
        if components["dust_sed"] is not None:
            composite_sed += components["dust_sed"]
        # TODO: Neb continuum
        # TODO: AGN / non-thermal emission
        if to_obs_frame:
            flux = (composite_sed / self.distance_factor).to(
                    self.obs_unit, u.spectral_density(self.target_wavelength))
            # redshift spectra
            return flux_conserving_interpolation(
                self.target_wavelength,
                self.target_wavelength * (1 + self.redshift.q),
                flux)
        return composite_sed

    def emission_photometry(self, stars_params=None, dust_att_params=None,
                            dust_em_params=None, to_obs_frame=False):
        """
        Compute synthetic photometry from the composite spectrum.

        Parameters
        ----------
        stars_params : dict, optional
            Parameters forwarded to the stellar component.
        dust_att_params : dict, optional
            Parameters forwarded to the attenuation model.
        dust_em_params : dict, optional
            Parameters forwarded to the dust emission component.
        to_obs_frame : bool, optional
            If True, compute photometry from observed-frame fluxes.

        Returns
        -------
        photometry : astropy.units.Quantity
            Band-integrated synthetic flux densities for configured filters.

        Notes
        -----
        Requires filters to be provided at initialization (or set later). The filter
        container must support evaluation of fluxes from a spectrum sampled on
        ``target_wavelength``.
        """
        spec = self.emission_spectrum(stars_params=stars_params,
                                      dust_att_params=dust_att_params,
                                      dust_em_params=dust_em_params,
                                      to_obs_frame=to_obs_frame)
        return self.filters.get_fnu(spec)[0]

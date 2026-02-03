from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.cosmology import WMAP9

from pst.utils import check_unit, flux_conserving_interpolation
from pst import observables

from pst.sed import SedComponent
from pst.dust import AttenuationModel, CalorimetricDustComponent


class GalaxySED(SedComponent):
    rest_unit = u.Lsun / u.AA  # Rest-frame default units
    obs_unit = u.erg / (u.s * u.cm**2 * u.AA)  # Observed-frame default units

    def __init__(self, *,
                 stellar_model: SedComponent=None,
                 dust_attenuation_model: AttenuationModel=None,  # Fix typing
                 dust_model: SedComponent=None,
                 target_wavelength: u.Quantity=None,
                 redshift: float=0.0, cosmology=None,
                 filters: List(observables.Filter)=None):
        # Setup components
        self.stellar_em = stellar_model
        self.dust_attenuation = dust_attenuation_model
        self.dust_em = dust_model
        # Target wavelength range
        if target_wavelength is not None:
            self.target_wavelength = target_wavelength
        else:
            if self.dust_em is None:
                self.target_wavelength = self.stellar_em.ssp.wavelength
            else:
                dust_wl = np.geomspace(1, 1000, 100) << u.um
                wl = np.concatenate(
                    (self.stellar_em.ssp.wavelength.to_value(u.AA),
                    dust_wl.to_value(u.AA)))
                self.target_wavelength = np.unique(wl) << u.AA
            print("Target wavelength range", self.target_wavelength[[0, -1]],
                   "\nNo. pixels:", self.target_wavelength.size)
        self.stellar_em.ssp.interpolate_sed(self.target_wavelength)
        # Setup component transformers
        self.energy_balance = True if isinstance(
            self.dust_em, CalorimetricDustComponent) else False
        # Setup observation properties
        self.redshift = redshift
        if cosmology is None:
            self.cosmology = WMAP9
        self.t_obs = self.cosmology.age(self.redshift)
        # Ensure that the stellar SFH is set to same today
        self.stellar_em.sfh.today = self.t_obs
        self.dl = self.cosmology.luminosity_distance(
            self.redshift).clip(10 << u.pc)
        self.distance_factor = 4 * np.pi * self.dl.to("cm")**2 * (
            1 + self.redshift)
        
        # Setup observables
        if filters is not None:
            print("Interpolating filters to target wavelength")
            filters.interpolate(self.target_wavelength)
            self.filters = filters

    def emission_components(self, stellar_em_params=None, dust_att_params=None,
                            dust_em_params=None):
        """TODO"""
        components = {
            "stellar_sed": None,
            "stellar_sed_unatt": None,
            "dust_sed": None,
            # "nebular_sed": None,
                      }

        stellar_em_params = stellar_em_params or {}
        stellar_em_params["t_obs"] = self.t_obs
        dust_att_params = dust_att_params or {}
        dust_em_params = dust_em_params or {}

        if self.energy_balance:
            stellar_sed_unatt, stellar_sed, dust_sed = self.dust_em.emission_spectrum(
                self.target_wavelength,
                source=self.stellar_em,
                source_params=stellar_em_params,
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
            if self.stellar_em is not None:
                stellar_sed_unatt = self.stellar_em.emission_spectrum(
                    self.target_wavelength, **stellar_em_params)
                components["stellar_sed_unatt"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            if self.dust_attenuation is not None:
                att_factor = self.dust_attenuation.attenuation_factor(
                    self.target_wavelength, **dust_att_params)
                stellar_sed = stellar_sed_unatt * att_factor
                components["stellar_sed"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))
            else:
                components["stellar_sed"] = stellar_sed_unatt.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))

            if self.dust_em is not None:
                dust_sed = self.dust_em.emission_spectrum(
                    self.target_wavelength, **dust_em_params)
                components["dust_sed"] = dust_sed.to(
                    self.rest_unit,
                    u.spectral_density(self.target_wavelength))

        return components

    def emission_spectrum(self,
                          stellar_em_params=None, dust_att_params=None,
                          dust_em_params=None, to_obs_frame=False):
        """TODO"""
        components = self.emission_components(
            stellar_em_params=stellar_em_params,
            dust_att_params=dust_att_params,
            dust_em_params=dust_em_params)

        composite_sed = np.zeros(self.target_wavelength.size) << self.rest_unit
        composite_sed += components.get("stellar_sed", 0)
        composite_sed += components.get("dust_sed", 0)

        if to_obs_frame:
            flux = (composite_sed / self.distance_factor).to(
                    self.obs_unit, u.spectral_density(self.target_wavelength))
            return flux_conserving_interpolation(
                self.target_wavelength,
                self.target_wavelength * (1 + self.redshift),
                flux)
        return composite_sed

    def emission_photometry(self, stellar_em_params=None, dust_att_params=None,
                            dust_em_params=None, to_obs_frame=False):
        spec = self.emission_spectrum(stellar_em_params=stellar_em_params,
                                      dust_att_params=dust_att_params,
                                      dust_em_params=dust_em_params,
                                      to_obs_frame=to_obs_frame)
        return self.filters.get_fnu(spec)[0]


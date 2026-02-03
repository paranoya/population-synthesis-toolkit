from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy import constants as const

from pst.utils import check_unit, flux_conserving_interpolation

class SedComponent(ABC):
    """
    Dust emission component.

    Returns an additive spectrum on the wavelength grid.
    """

    default_unit: u.Unit

    @abstractmethod
    def emission_spectrum(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Returns emission spectrum as Quantity.

        The *units* are determined by your chosen normalization parameterization
        (e.g., L_IR scaling or amplitude in L_lambda). Keep this consistent with PST.
        """
        raise NotImplementedError

    @staticmethod
    def integrate_sed(wavelength, sed, wl_min: u.Quantity=None, wl_max: u.Quantity=None):
        mask = np.ones(sed.size, dtype=bool)
        if wl_min is not None:
            mask &= (wavelength >= wl_min)
        if wl_max is not None:
            mask &= (wavelength <= wl_max)
        return np.trapz(sed[mask].value, wavelength[mask].value) << (sed.unit * wavelength.unit)

    @classmethod
    def q_ionizing_photons(cls, wavelength, sed):
        return cls.integrate_sed(wavelength=wavelength,
                                 sed=sed / (const.h * const.c / wavelength),
                                 wl_min=None, wl_max=912 << u.AA)

class TabularSedComponent(SedComponent):

    @property
    @abstractmethod
    def default_unit(self):
        pass

    @abstractmethod
    def load_table(self):
        pass

    @abstractmethod
    def emission_spectrum(self, wavelength: u.Quantity, **params) -> u.Quantity:
        """
        Returns emission spectrum as Quantity.

        The *units* are determined by your chosen normalization parameterization
        (e.g., L_IR scaling or amplitude in L_lambda). Keep this consistent with PST.
        """
        raise NotImplementedError

@dataclass
class StellarComponent(SedComponent):
    ssp: "SSP"
    sfh: "ChemicalEvolutionModel"
    default_unit = u.Lsun / u.AA

    def emission_spectrum(self, wavelength: u.Quantity, **params):
        sed = self.sfh.compute_SED(self.ssp, **params)
        if wavelength.size != self.ssp.wavelength.size or not (
            np.allclose(self.ssp.wavelength.to_value(wavelength.unit),
            wavelength.value)):
            sed = flux_conserving_interpolation(wavelength, self.ssp.wavelength, sed)
        return sed.to(self.default_unit, u.spectral_density(wavelength))

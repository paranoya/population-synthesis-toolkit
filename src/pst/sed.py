from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

from astropy import units as u
from astropy import constants as const

from pst.utils import check_unit, flux_conserving_interpolation
from pst.model import Parameter, ModelBase
from pst.SSP import SSPBase
from pst.cem import ChemicalEvolutionModel

class SedComponent(ModelBase, ABC):
    """
    Base class for spectral energy distribution components.

    Subclasses implement :meth:`emission_spectrum` and return a spectrum
    sampled on the requested wavelength grid.
    """

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
        """
        Integrate a spectrum over wavelength with optional bounds.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid associated with ``sed``.
        sed : astropy.units.Quantity
            Spectral density values sampled on ``wavelength``.
        wl_min : astropy.units.Quantity, optional
            Lower integration limit.
        wl_max : astropy.units.Quantity, optional
            Upper integration limit.

        Returns
        -------
        integral : astropy.units.Quantity
            Integrated quantity with units ``sed.unit * wavelength.unit``.
        """
        mask = np.ones(sed.size, dtype=bool)
        if wl_min is not None:
            mask &= (wavelength >= wl_min)
        if wl_max is not None:
            mask &= (wavelength <= wl_max)
        return np.trapz(sed[mask].value, wavelength[mask].value) << (sed.unit * wavelength.unit)

    @classmethod
    def q_ionizing_photons(cls, wavelength, sed):
        """
        Compute ionizing photon production integrated below 912 Angstrom.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Wavelength grid.
        sed : astropy.units.Quantity
            Spectral energy density.

        Returns
        -------
        q_ion : astropy.units.Quantity
            Integrated ionizing photon rate-like quantity.
        """
        return cls.integrate_sed(wavelength=wavelength,
                                 sed=sed / (const.h * const.c / wavelength),
                                 wl_min=None, wl_max=912 << u.AA)

class TabularSedComponent(SedComponent):
    """Base interface for SED components backed by tabulated data."""

    @property
    @abstractmethod
    def default_unit(self):
        """Return the native output spectral unit for this component."""
        pass

    @abstractmethod
    def load_table(self):
        """Load the underlying tabulated model data."""
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
    """Stellar emission component built from an SSP model and an SFH/CEM."""

    ssp: SSPBase
    sfh: ChemicalEvolutionModel
    name: str = "stellar_emission"
    default_unit = u.Lsun / u.AA

    def emission_spectrum(self, wavelength: u.Quantity, **params):
        """
        Compute stellar emission on the requested wavelength grid.

        Parameters
        ----------
        wavelength : astropy.units.Quantity
            Target wavelength grid.
        **params
            Parameters forwarded to ``sfh.compute_SED``.

        Returns
        -------
        sed : astropy.units.Quantity
            Stellar SED in ``default_unit``.
        """
        sed = self.sfh.compute_SED(self.ssp, **params)
        if wavelength.size != self.ssp.wavelength.size or not (
            np.allclose(self.ssp.wavelength.to_value(wavelength.unit),
            wavelength.value)):
            sed = flux_conserving_interpolation(wavelength, self.ssp.wavelength, sed)
        return sed.to(self.default_unit, u.spectral_density(wavelength))

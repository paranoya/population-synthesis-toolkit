"""
This module contains some tools for computing observable quantities
(e.g. photometry, equivalent widths) from spectra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import os
from astropy import units as u
from astropy import constants as const
from astropy import constants
import requests
import json
from matplotlib import pyplot as plt

from pst.utils import check_unit, flux_conserving_interpolation

ArrayLike = Union[np.ndarray, u.Quantity]
PST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def list_of_available_filters():
    """List the currently available filters in the default directory."""
    filter_dir = os.path.join(PST_DATA_DIR, "filters")
    return os.listdir(filter_dir)

def load_photometric_filters(filters, to_filter_list=False):
    """Convenience function for constructing a list of photometric filters.
    
    Parameters
    ----------
    filters: list of str
        List of filters to load. The list might contain the absolute path to
        a filter response file, or just the filter name following the SVO
        convention.
    
    Returns
    -------
    filters_out: list of :class:`pst.observables.Filter`
        List of filters.
    """
    filters_out = []
    for f in filters:
        if os.path.exists(f):
            filters_out.append(Filter.from_text_file(f))
        else:
            filters_out.append(Filter.from_svo(f))
    if to_filter_list:
        return FilterList(filters_out)
    return filters_out

def download_svo_filter(name: str, dest_dir: str, verbose=True):
    """Download a filter from the Spanish Virtual Observatory (SVO) Filter Profile Service.
    
    Parameters
    ----------
    name : str
        SVO-compliant filename. The naming convention for SVO filters is 
        TELESC_INSTRUMENT.BAND (e.g. WISE_WISE.W1, Subaru_HSC.g)
    dest_dir : str
        Path to the directory where to store the data.
    
    Returns
    -------
    file_path : str
        Path to the downloaded filter file.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # create folder if it does not exist
    name = name.strip(".dat")
    base_url="http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id="
    url = base_url + name.replace("_", "/")
    filename = name + ".dat"
    file_path = os.path.join(dest_dir, filename)
    if verbose:
        print(f"Querying SVO Filter: {url}")
    r = requests.get(url, stream=True)
    if len(r.text) > 0:
        if verbose:
            print(f"Saving new filter {name} to ", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return file_path
    else:
        raise FileNotFoundError("Query to {url} was unsucessful")


class Filter(object):
    """A photometric filter.
    
    Attributes
    ----------
    filter_resp: np.ndarray
        Original photometric passband response curve.
    filter_wavelength : :class:`numpy.ndarray` or :class:`astropy.units.Quantity`
        Original wavelength associated to ``filter_resp``. If type is
        ``numpy.ndarray``, the value is converted in to an ``astropy.units.Quantity`
        in Angstrom.
    response: np.ndarray
        Filter passband response curve after interpolation.
    wavelength: :class:`numpy.ndarray` or :class:`astropy.units.Quantity`
        Wavelength vector associated to ``response``. If type is
        ``numpy.ndarray``, the value is converted in to an ``astropy.units.Quantity`
        in Angstrom.
    default_dir : str
        Default directory containing filter files.

    Example
    -------
    >>> from pst.observables import Filter
    >>> ps_r_filter = Filter("PANSTARRS_PS1.r")
    """

    default_dir = os.path.join(PST_DATA_DIR, "filters")

    def __init__(self, wavelength=None, response=None,
                 filter_wavelength=None, filter_response=None, name=None):

        self.wavelength = wavelength
        self.response = response
        self.filter_wavelength = filter_wavelength
        self.filter_resp = filter_response
        if filter_wavelength is None and filter_response is None and wavelength is not None:
            self.filter_wavelength = self.wavelength
            self.filter_resp = self.response
        elif filter_wavelength is None and wavelength is None:
            raise NameError("wavelength or filter_wavelength must be provided")
        self.norm_photons = None
        self.name = name

    @property
    def wavelength(self):
        """Interpolated wavelength grid used by ``response``."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if not isinstance(value, u.Quantity) and value is not None:
            self._wavelength = value << u.angstrom
        else:
            self._wavelength = value

    @property
    def filter_wavelength(self):
        """Native wavelength grid associated with ``filter_resp``."""
        return self._filter_wavelength

    @filter_wavelength.setter
    def filter_wavelength(self, value):
        if not isinstance(value, u.Quantity) and value is not None:
            self._filter_wavelength = value << u.angstrom
        else:
            self._filter_wavelength = value

    @classmethod
    def from_text_file(cls, path, wavelength_unit=u.angstrom, **kwargs):
        """Load a :class:``Filter`` from an input text file.

        Parameters
        ---------
        path : str
            Path to the text file containing the filter information.
            The first and second columns must correspond to the
            wavelength and passband curve, respectively.
        wavelength_unit : :class:``astropy.units.Unit``
            Unit associated with wavelength values in the input file.
        **kwargs : 
            Arguments to be passed to :func:`numpy.loadtxt`

        Returns
        -------
        filter : :class:`Filter`
            The ``Filter`` containing the input information.
        """
        wavelength, response = np.loadtxt(path, usecols=(0, 1), unpack=True,
                                          **kwargs)
        name = os.path.basename(path).replace(".dat", "")
        return cls(filter_wavelength=wavelength * wavelength_unit,
                   filter_response=response, name=name)

    @classmethod
    def from_svo(cls, name, destination_dir=None):
        """Load a :class:``Filter`` from the Spanish Vitural Observatory archive.

        Parameters
        ---------
        name : str
            SVO filter name. If the filter is not found locally, it will be
            downloaded from the archive.

        Returns
        -------
        filter : :class:`Filter`
            The ``Filter`` containing the input information.

        Example
        -------
        >>> from pst.observables import Filter
        >>> panstarrs_r_filter = Filter.from_svo("PANSTARRS_PS1.r")
        """
        if destination_dir is None:
            destination_dir = cls.default_dir
        path = cls._isfilter(name)
        if path is not None:
            return cls.from_text_file(path)
        else:
            path = download_svo_filter(name, dest_dir=destination_dir)
            return cls.from_text_file(path)

    @classmethod
    def _isfilter(cls, name):
        path = os.path.join(cls.default_dir, name.strip(".dat") + ".dat")
        if os.path.isfile(path):
            return path
        else:
            return None

    def effective_wavelength(self):
        r"""Compute the effective wavelength of the filter.
        
        Description
        -----------
        The effective wavelength is computed as

        .. math::
            \lambda_{\rm eff} = \frac{\int{R(\lambda) \cdot \lambda d\lambda}}{\int{R(\lambda) d\lambda}}
        
        Returns
        -------
        eff_wl : :class:`astropy.units.Quantity`
            The effective wavelength of the filter.
        """
        return np.sum(self.filter_wavelength*self.filter_resp)/np.sum(self.filter_resp)

    def effective_bandwidth(self):
        r"""Compute the effective bandwidth of the filter.
        
        Description
        -----------
        The effective bandwith is computed as

        .. math::
            \Delta \lambda_{\rm BW} = \sqrt{8\log(2)} \left(\frac{\int{R(\lambda) \cdot \lambda^2 d\lambda}}{\int{R(\lambda) d\lambda}} - \lambda_{\rm eff}\right)^{1/2}

        Returns
        -------
        eff_bw : :class:`astropy.units.Quantity`
            The effective bandwidth of the filter.
        
        See also
        --------
        :func:`effective_wavelength`
        """
        return np.sqrt(8*np.log(2)*(
            np.sum(self.filter_wavelength**2*self.filter_resp)/np.sum(self.filter_resp)
            - self.effective_wavelength()**2))

    def effective_transmission(self):
        r"""Compute the effective bandwidth of the filter.
        
        Description
        -----------
        The effective transmission is computed as

        .. math::
            R_{\rm eff} = \frac{\int{R(\lambda)^2 d\lambda}}{\int{R(\lambda) d\lambda}}

        Returns
        -------
        eff_tr : float
            The effective transmission of the filter.
        """
        return np.sum(self.filter_resp**2)/np.sum(self.filter_resp)

    def interpolate(self, wavelength=None):
        """Interpolate and update the Filter response curve to an input wavelength.
        
        Description
        -----------
        Interpolate linearly the Filter response curve to an input wavelength vector.
        The result will update the exising values of ``wavelength`` and ``response``.

        Parameters
        ----------
        wavelength: :class:`numpy.ndarray` or :class:`astropy.units.Quantity`
            Wavelength vector to interpolate ``filt_resp``. If type is
            ``numpy.ndarray``, the value is converted in to an ``astropy.units.Quantity`
            in Angstrom.
        
        Returns
        -------
        response : np.ndarray
            Filter response curve interpolated to the input values of ``wavelength``.
        """
        if not hasattr(wavelength, "unit"):
            wavelength = wavelength << u.angstrom

        self.response = flux_conserving_interpolation(
            wavelength, self.filter_wavelength, self.filter_resp)
        self.wavelength= wavelength

        self.norm_photons, _ = self.get_photons(
            3631 * u.Jy * np.ones(self.wavelength.shape) * constants.c / self.wavelength**2,
            mask_nan=False)

        return self.response

    def get_photons(self, spectra, spectra_err=None, mask_nan=True):
        r"""Compute the photon flux from an input spectra.
        
        Description
        -----------
        The photon flux associated to the filter is computed by numerically integrating
        the input ``spectra`` with the filter ``response``, using the trapezid method.

        .. :math:
            phot_flux = \int{F_\lambda \cdot frac{\lambda}{hc} \cdot R(\lambda) d\lambda}

        Parameters
        ----------
        spectra : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Input spectra (flux density per wavelength unit) with same
            dimensions as the Filter ``wavelength``.
        spectra_err : :class:`np.ndarray` or :class:``astropy.units.Quantity``, optional
            Input spectra associated error.
        mask_nan : bool, optional
            If True, NaN values are masked.
        
        Returns
        -------
        photon_flux : :class:``astropy.units.Quantity``
            Filter photon flux.
        photon_flux_err : :class:``astropy.units.Quantity``
            Filter photon flux associated error.
        """
        spectra = check_unit(spectra, default_unit=u.Lsun / u.angstrom / u.cm**2,
                                   equivalence=u.spectral_density, wav=self.wavelength)

        if mask_nan:
            mask = np.isfinite(spectra)
            photon_flux = np.trapz(
                spectra[mask] / (constants.h * constants.c / self.wavelength[mask]
                                   ) * self.response[mask],
                x=self.wavelength[mask])
        else:
            photon_flux = np.trapz(
                spectra / (constants.h * constants.c / self.wavelength
                                   ) * self.response,
                x=self.wavelength)

        if spectra_err is not None:

            spectra_err = check_unit(spectra_err,
                                           default_unit=u.Lsun / u.angstrom / u.cm**2,
                                           equivalence=u.spectral_density, wav=self.wavelength)
            if mask_nan:
                mask = mask & np.isfinite(spectra_err)
            else:
                mask = np.ones_like(spectra_err, dtype=bool)

            photon_flux_err = np.trapz(
                spectra_err[mask] / (constants.h * constants.c / self.wavelength[mask]
                                       ) * self.response[mask],
                x=self.wavelength[mask])
        else:
            photon_flux_err = None
        return photon_flux, photon_flux_err
    
    def get_ab(self, spectra, spectra_err=None, mask_nan=True):
        r"""Compute the synthetic AB magnitude from an input spectra.
        
        Description
        -----------
        The AB magnitude associated to the filter is computed by numerically integrating
        the input ``spectra`` with the filter ``response``, using the trapezid method.

        .. :math:
            phot_flux = -2.5 \cdot \log_{10}\left(\frac{N_{phot}(spectra)}{N_{phot}(3631)}\right)

        Parameters
        ----------
        spectra : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Input spectra (flux density per wavelength unit) with same
            dimensions as the Filter ``wavelength``.
        spectra_err : :class:`np.ndarray` or :class:``astropy.units.Quantity``, optional
            Input spectra associated error.
        mask_nan : bool, optional
            If True, NaN values are masked.
        
        Returns
        -------
        mag_ab : :class:``astropy.units.Quantity``
            AB magnitude.
        mag_ab_err : :class:``astropy.units.Quantity``
            AB magnitude associated error.
        
        See also
        --------
        :func:`get_photons`
        """
        n_photons, n_photons_err = self.get_photons(spectra, spectra_err, mask_nan=mask_nan)
        mag_ab = - 2.5 * np.log10(n_photons / self.norm_photons)
        if n_photons_err is None:
            mag_ab_err = None
        else:
            mag_ab_err = 2.5 / np.log(10) * n_photons_err / n_photons
        return mag_ab, mag_ab_err

    def get_fnu(self, spectra, spectra_err=None, mask_nan=True):
        """Compute synthetic flux density per frequency unit from a spectrum.

        Parameters
        ----------
        spectra : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Input spectra (flux density per wavelength unit) with same
            dimensions as the Filter ``wavelength``.
        spectra_err : :class:`np.ndarray` or :class:``astropy.units.Quantity``, optional
            Input spectra associated error.
        mask_nan : bool, optional
            If True, NaN values are masked.

        Returns
        -------
        f_nu : :class:``astropy.units.Quantity``
            Synthetic flux density in Jy.
        f_nu_err : :class:``astropy.units.Quantity``
            Associated uncertainty in Jy.

        See also
        --------
        :func:`get_photons`
        """
        n_photons, n_photons_err = self.get_photons(spectra, spectra_err, mask_nan=mask_nan)
        f_nu = n_photons / self.norm_photons * 3631 * u.Jy
        if spectra_err is None:
            f_nu_err = None
        else:
            f_nu_err = n_photons_err / self.norm_photons * 3631 * u.Jy
        return f_nu, f_nu_err

    def get_flambda_vegamag(self, spectra, spectra_err=None, mask_nan=True):
        """Compute synthetic flux density per wavelength unit from a spectrum.

        Parameters
        ----------
        spectra : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Input spectra (flux density per wavelength unit) with same
            dimensions as the Filter ``wavelength``.
        spectra_err : :class:`np.ndarray` or :class:``astropy.units.Quantity``, optional
            Input spectra associated error.
        mask_nan : bool, optional
            If True, NaN values are masked.

        Returns
        -------
        f_lambda : :class:``astropy.units.Quantity``
            Synthetic flux density per wavelength.
        f_lambda_err : :class:``astropy.units.Quantity``
            Associated uncertainty estimate.

        See also
        --------
        :func:`get_photons`
        """
        spectra = spectra = check_unit(spectra, default_unit=u.Lsun / u.angstrom / u.cm**2,
                                   equivalence=u.spectral_density, wav=self.wavelength)

        if mask_nan:
            mask = np.isfinite(spectra)
        else:
            mask = np.ones_like(spectra, dtype=bool)

        f_lambda = np.trapz(spectra[mask] * self.wavelength[mask] * self.response[mask], x=self.wavelength[mask]
                               ) / np.trapz(self.response[mask] * self.wavelength[mask], x=self.wavelength[mask])

        if spectra_err is not None:

            spectra_err = check_unit(spectra_err, default_unit=u.Lsun / u.angstrom / u.cm**2,
                                           equivalence=u.spectral_density, wav=self.wavelength)
            if mask_nan:
                mask = mask & np.isfinite(spectra_err)
            else:
                mask = np.ones_like(spectra_err, dtype=bool)

            f_lambda_err = np.trapz(
            spectra_err[mask] / (constants.h * constants.c / self.wavelength[mask]
                                   ) * self.response[mask],
            x=self.wavelength[mask])
        else:
            f_lambda_err = None

        return f_lambda, f_lambda_err

    def plot(self, add_props=False, ax=None, show=False):
        """Plot the filter response curve.
        
        Plot the original filter response curve together with the interpolated
        version computed using a new grid of wavelengths.

        Parameters
        ----------
        show: bool
            If True
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        ax.step(self.filter_wavelength, self.filter_resp, label='Original',
                color='k', where="mid")
        ax.plot(self.filter_wavelength, self.filter_resp, '.', color='k')
        ax.set_xlabel(f"Wavelength ({self.filter_wavelength.unit})")
        ax.set_ylabel("Filter response")
        if self.wavelength is not None:
            ax.step(self.wavelength, self.response, label='Interpolated',
                      color='r', where="mid")
        ax.legend()
        if add_props:
            eff_wl = self.effective_wavelength()
            eff_bw = self.effective_bandwidth()
            ax.axvline(eff_wl.value - eff_bw.value / 2, c="k", ls=":")
            ax.axvline(eff_wl.value + eff_bw.value / 2, c="k", ls=":")
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax


class TopHatFilter(Filter):
    """Top hat photometric filter
    
    See also
    --------
    :class:`Filter`
    """
    def __init__(self, central_wave, width, **kwargs):
        central_wave = check_unit(central_wave, u.Angstrom)
        width = check_unit(width, u.Angstrom)

        self.wavelength = kwargs.get('wavelength', None)
        if self.wavelength is None:
            self.filter_wavelength = np.linspace(central_wave - width,
                                    central_wave + width,
                                    50)
        else:
            self.wavelength = check_unit(self.wavelength, u.Angstrom)
            self.filter_wavelength = self.wavelength.copy()

        self.filter_resp = np.ones(self.filter_wavelength.size)
        self.filter_resp[self.filter_wavelength < central_wave - width / 2] = 0
        self.filter_resp[self.filter_wavelength > central_wave + width / 2] = 0
        if self.wavelength is None:
            self.response = self.filter_resp.copy()


@dataclass
class FilterList:
    """A list of :class:`Filter` instances for faster computations.
    
    #TODO
    """

    filters: List["Filter"]
    wavelength: Optional[u.Quantity] = None
    response: Optional[np.ndarray] = None
    kernel_phot: Optional[u.Quantity] = None
    dlambda: Optional[u.Quantity] = None
    norm_phot: Optional[u.Quantity] = None
    names: Optional[List[str]] = None

    def __post_init__(self):
        self.filters = list(self.filters)
        self.names = [getattr(f, "name", f"band{ib}") for ib, f in enumerate(self.filters)]

    @property
    def n_bands(self) -> int:
        """Number of filters contained in the list."""
        return len(self.filters)

    @property
    def effective_wavelength(self) -> u.Quantity:
        """Array of effective wavelengths, one per filter."""
        return u.Quantity([f.effective_wavelength() for f in self.filters], u.AA)

    def wavelength_range(self, kappa_bw=2.0) -> [u.Quantity, u.Quantity]:
        """Get the net wavelength coverage by the filters."""
        min_wl = 1e6 << u.AA
        max_wl = 1 << u.AA

        for f in self.filters:
            eff_wl, eff_bw = f.effective_wavelength(), f.effective_bandwidth()
            low, up = eff_wl - eff_bw * kappa_bw, eff_wl + eff_bw * kappa_bw
            if low < min_wl:
                min_wl = low
            if up > max_wl:
                max_wl = up
        return [min_wl, max_wl]

    def interpolate(self, wavelength: ArrayLike) -> "FilterList":
        """
        Interpolate all filter responses to a common wavelength grid.

        Parameters
        ----------
        wavelength : array or Quantity
            Target wavelength grid. Must be 1D and monotonic increasing.

        Returns
        -------
        self
        """
        wl = check_unit(wavelength)

        if wl.ndim != 1:
            raise ValueError("wavelength must be 1D")

        # Enforce monotonic increasing for stable integration
        if np.any(np.diff(wl.to_value(wl.unit)) <= 0):
            raise ValueError("wavelength grid must be strictly increasing")

        self.wavelength = wl

        # Build (B, W) response matrix
        resp = np.empty((self.n_bands, wl.size), dtype=float)
        for i, f in enumerate(self.filters):
            # Use your flux-conserving interpolation (same behavior as Filter.interpolate)
            resp[i] = flux_conserving_interpolation(wl, f.filter_wavelength, f.filter_resp)

        self.response = resp

        # Cash delta_lambda for integral
        dl = np.empty_like(wl)
        dl[0] = 0.5 * (wl[1] - wl[0])
        dl[-1] = 0.5 * (wl[-1] - wl[-2])
        if wl.size > 2:
            dl[1:-1] = 0.5 * (wl[2:] - wl[:-2])
        self.dlambda = dl

        # Photon kernel: (lambda / (h c)) * R(lambda)
        self.kernel_phot = (wl / (const.h * const.c)) * resp

        # AB normalization per band
        fnu0 = 3631 * u.Jy
        f_lambda_ref = (fnu0 * const.c / wl**2).to(
            u.erg / (u.s * u.cm**2 * wl.unit),
            equivalencies=u.spectral_density(wl),
        )
        self.norm_phot = u.Quantity(
            np.einsum("i,bi,i->b", f_lambda_ref, self.kernel_phot, self.dlambda),
            copy=False,
        )

        return self

    def _require_interpolated(self):
        if self.wavelength is None or self.response is None or self.kernel_phot is None or self.dlambda is None:
            raise RuntimeError("Call interpolate(wavelength) before computing photometry.")

    def get_photons(self, spectra: ArrayLike, spectra_err: Optional[ArrayLike] = None,
                mask_nan: bool = True) -> Tuple[u.Quantity, Optional[u.Quantity]]:
        """
        Vectorized photon flux through all bandpasses.

        Parameters
        ----------
        spectra : array or Quantity
            Shape (..., n_wave). Flux density per wavelength.
        spectra_err : array or Quantity, optional
            Same shape as spectra.
        mask_nan : bool
            If True, NaNs in spectra are treated as zero contribution.

        Returns
        -------
        n_phot : Quantity
            Shape (..., n_bands)
        n_phot_err : Quantity or None
            Shape (..., n_bands)
        """
        self._require_interpolated()
        wl = self.wavelength

        F = check_unit(
            spectra,
            default_unit=u.Lsun / u.angstrom / u.cm**2,
            equivalence=u.spectral_density,
            wav=wl,
        )

        if F.shape[-1] != wl.size:
            raise ValueError(f"spectra last axis must be n_wave={wl.size}, got {F.shape[-1]}")

        # Mask NaNs
        if mask_nan:
            finite = np.isfinite(F)
            F = u.Quantity(np.where(finite, F.value, 0.0), unit=F.unit, copy=False)

        # Integrate along the last axis for each band
        n_phot = u.Quantity(
            np.einsum("...i,bi,i->...b", F, self.kernel_phot, self.dlambda),
            copy=False,
        )

        if spectra_err is None:
            return n_phot, None

        Ferr = check_unit(
            spectra_err,
            default_unit=u.Lsun / u.angstrom / u.cm**2,
            equivalence=u.spectral_density,
            wav=wl,
        )
        if Ferr.shape != F.shape:
            raise ValueError("spectra_err must have the same shape as spectra")

        if mask_nan:
            finite_e = np.isfinite(Ferr)
            Ferr = u.Quantity(np.where(finite_e, Ferr.value, 0.0),
            unit=Ferr.unit, copy=False)

        n_phot_err = u.Quantity(
            np.einsum("...i,bi,i->...b", Ferr, self.kernel_phot, self.dlambda),
            copy=False,
        )
        return n_phot, n_phot_err

    def get_fnu(self, spectra: ArrayLike, spectra_err: Optional[ArrayLike] = None,
            mask_nan: bool = True) -> Tuple[u.Quantity, Optional[u.Quantity]]:
        """
        Compute synthetic f_nu in AB system for all bands.

        Returns
        -------
        fnu : Quantity
            Shape (..., n_bands) in Jy
        fnu_err : Quantity or None
            Shape (..., n_bands) in Jy
        """
        self._require_interpolated()

        n_phot, n_phot_err = self.get_photons(spectra,
                spectra_err=spectra_err, mask_nan=mask_nan)

        fnu0 = 3631 * u.Jy
        # Per band
        fnu = (n_phot / self.norm_phot) * fnu0

        if n_phot_err is None:
            return fnu, None
        fnu_err = (n_phot_err / self.norm_phot) * fnu0
        return fnu, fnu_err

    def abmag(self, spectra: ArrayLike, spectra_err: Optional[ArrayLike] = None,
              mask_nan: bool = True) -> Tuple[u.Quantity, Optional[u.Quantity]]:
        """
        Compute AB magnitudes for all bands.
        """
        self._require_interpolated()
        n_phot, n_phot_err = self.get_photons(spectra, spectra_err=spectra_err,
                                              mask_nan=mask_nan)
        ratio = n_phot / self.norm_phot
        mag = -2.5 * np.log10(ratio)

        if n_phot_err is None:
            return mag, None

        mag_err = (2.5 / np.log(10)) * (n_phot_err / n_phot)
        return mag, mag_err

    def plot(self, add_props=False, ax=None, show=False):
        """Plot the filters response curve.

        Parameters
        ----------
        show: bool
            If True
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        for f in self.filters:
            f.plot(ax=ax, add_props=add_props, show=False)

        limits = self.wavelength_range()
        ax.set_xlim(limits[0].value, limits[1].value)
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

class EquivalentWidth(object):
    r"""Equivalent width of an spectral region.
    
    Description
    -----------
    Given a stellar spectra spectra :math:`F_\lambda`, the equivalent width is
    defined as the area of the spectral line, defined over a given spectral
    region (:math:`\lambda_{\rm C,\,min}, \lambda_{\rm C,\,max}`), divided by
    the average flux of the continuum in a given spectral region.
    It is computed as: 

    .. math::
        EW = \int_{\lambda_{C,\,\rm min}}^{\lambda_{\rm C,\,max}} \left(1 - \frac{F_\lambda(\lambda)}{F_{\rm cont}(\lambda)}\right) d\lambda
    where:

    .. math::
        F_{cont}(\lambda) = \frac{F_{\rm B} \lambda_{\rm R} - F_{\rm R} \lambda_{\rm B}}{\lambda_{\rm R} - \lambda_{\rm B}} + \lambda \frac{F_{\rm R} - F_{\rm B}}{\lambda_{\rm R} - \lambda_{\rm B}}
        
    and :math:`F_{\rm B}` and :math:`F_{\rm R}` are the average flux in the left
    and right spectral windows, respectively, defined as:

    .. math::
        F_{\rm B} = \frac{1}{\lambda_{\rm B,\,max} - \lambda_{\rm B,\,min}} \int_{\lambda_{\rm B,\,min}}^{\lambda_{\rm B,\,max}} F_\lambda(\lambda) d\lambda
    
    .. math::
        F_{\rm R} = \frac{1}{\lambda_{\rm R,\,max} - \lambda_{\rm R,\,min}} \int_{\lambda_{\rm R,\,min}}^{\lambda_{\rm R,\,max}} F_\lambda(\lambda) d\lambda

    where :math:`\lambda_{\rm B,\,min}` and :math:`\lambda_{\rm B,\,max}` are the
    left spectral window boundaries, and :math:`\lambda_{\rm R,\,min}` and
    :math:`\lambda_{\rm R,\,max}` are the right spectral window boundaries.

    Example
    -------
    >>> from pst.observables import EquivalentWidth
    >>> ew = EquivalentWidth(left_wl_range=(3700, 3900),
    ...                      central_wl_range=(4000, 4100),
    ...                      right_wl_range=(4200, 4400))
    >>> wavelength = np.linspace(3600, 4600, 1000) * u.angstrom
    >>> spectra = np.random.normal(1, 0.1, size=wavelength.size) * u.erg / u.s / u.cm**2 / u.angstrom
    >>> ew_value, ew_err = ew.compute_ew(wavelength, spectra)
    """
    def __init__(self, left_wl_range, central_wl_range, right_wl_range):
        self.left_wl_range = np.array(left_wl_range)
        self.central_wl_range = np.array(central_wl_range)
        self.right_wl_range = np.array(right_wl_range)
    
    @property
    def left_wl_range(self) -> u.Quantity:
        r"""Spectral range defining the left pseudocontinuum window :math:`\lambda_{\rm B,\,min}, \lambda_{\rm B,\,max}`."""
        return self._left_wl_range
    
    @left_wl_range.setter
    def left_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._left_wl_range = value * u.angstrom
        else:
            self._left_wl_range = value

    @property
    def right_wl_range(self) -> u.Quantity:
        r"""Spectral range defining the right pseudocontinuum window :math:`\lambda_{\rm R,\,min}, \lambda_{\rm R,\,max}`."""
        return self._right_wl_range

    @right_wl_range.setter
    def right_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._right_wl_range = value * u.angstrom
        else:
            self._right_wl_range = value
    
    @property
    def central_wl_range(self) -> u.Quantity:
        r"""Spectral range defining the equivalent width window :math:`\lambda_{\rm C,\,min}, \lambda_{\rm C,\,max}`."""
        return self._central_wl_range

    @central_wl_range.setter
    def central_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._central_wl_range = value * u.angstrom
        else:
            self._central_wl_range = value

    def compute_ew(self, wavelength, spectra, spectra_err=None):
        """Compute the equivalent width of a given input spectra.
        
        Description
        -----------
        The equivalent width is computed using the definition given in the class
        description. Positive values of the equivalent width indicate an absorption
        line, while negative values indicate an emission line. The error on the
        equivalent width is computed using the error propagation formula, assuming
        null covariance between the spectral points.

        Parameters
        ----------
        spectra : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Input spectra. If the array is multidimensional, the first axis must
            correspond to the spectral direction.
        wavelength : :class:`np.ndarray` or :class:``astropy.units.Quantity``
            Wavelength array associated to ``spectra``.
        spectra_err : :class:`np.ndarray` or :class:``astropy.units.Quantity``, optional
            If provided, computed the associated error of the equivalent width.
        
        Returns
        -------
        ew : np.ndarray
            The equivalent width of the input spectra.
        ew_err : np.ndarray
            The associated error of the equivalent width.
        """

        left_pts = np.where(np.searchsorted(self.left_wl_range, wavelength) == 1)[0]
        right_pts = np.where(np.searchsorted(self.right_wl_range, wavelength) == 1)[0]
        lick_pts = np.where(np.searchsorted(self.central_wl_range, wavelength) == 1)[0]
        delta_ew = self.central_wl_range[1] - self.central_wl_range[0]
        # pseudo-continuum interpolation
        right_weight = (self.central_wl_range.mean() - self.left_wl_range.mean()
                        ) / (self.right_wl_range.mean() - self.left_wl_range.mean())

        if spectra.ndim > 1:
            left_cont = np.nanmean(spectra[left_pts, :], axis=0)
            right_cont = np.nanmean(spectra[right_pts, :], axis=0)
            pseudocont = (1 - right_weight) * left_cont + right_weight * right_cont
            flux = np.nanmean(spectra[lick_pts], axis=0)
            ew = delta_ew * (1 - flux/pseudocont)
            if spectra_err is None:
                ew_err = np.nan
            else:
                left_cont_var = np.nanmean(spectra_err[left_pts, :]**2,
                                        axis=0) / left_pts.size
                right_cont_var = np.nanmean(spectra_err[right_pts, :]**2,
                                            axis=0) / right_pts.size
                pseudocont_var = ((1 - right_weight) * left_cont_var
                                + right_weight * right_cont_var)
                flux_var = np.nanmean(spectra_err[lick_pts]**2, axis=0)
                ew_var = ((delta_ew / pseudocont)**2 * flux_var
                        + (delta_ew * flux/pseudocont**2)**2 * pseudocont_var)
                ew_err = np.sqrt(ew_var)
        else:
            left_cont = np.nanmean(spectra[left_pts])
            right_cont = np.nanmean(spectra[right_pts])
            pseudocont = (1 - right_weight) * left_cont + right_weight * right_cont
            flux = np.nanmean(spectra[lick_pts])
            ew = delta_ew * (1 - flux/pseudocont)

            if spectra_err is None:
                ew_err = np.nan
            else:
                left_cont_var = np.nanmean(spectra_err[left_pts]**2
                                        ) / left_pts.size
                right_cont_var = np.nanmean(spectra_err[right_pts]**2
                                            ) / right_pts.size
                pseudocont_var = ((1 - right_weight) * left_cont_var
                                + right_weight * right_cont_var)
                flux_var = np.nanmean(spectra_err[lick_pts]**2)
                ew_var = delta_ew**2 * (
                    flux_var / lick_pts.size / pseudocont**2
                    + flux**2 * pseudocont_var / pseudocont**4)
                ew_err = np.sqrt(ew_var)
        return ew, ew_err

    @classmethod
    def from_json(cls, path):
        """Load a :class:`EquivalentWidth` from a JSON file.
        
        Parameters
        ----------
        path : str
            Path to the JSON file.
        
        Returns
        -------
        ew : :class:`EquivvalentWidth`
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_name(cls, name):
        """Load a :class:`EquivalentWidth` from a JSON file.
        
        Parameters
        ----------
        name : str
            Name of the Lick index.

        Returns
        -------
        ew : :class:`EquivvalentWidth`
        """
        json_file = os.path.join(PST_DATA_DIR, "lick", name + ".json")
        if os.path.isfile(json_file):
            return cls.from_json(json_file)
        else:
            raise FileNotFoundError(f"There is no JSON file\n -{json_file}"
                                    f"associated to input name {name}")

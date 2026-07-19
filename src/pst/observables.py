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
from astropy.io import ascii
import requests
import json
from matplotlib import pyplot as plt

from pst.utils import check_unit, flux_conserving_interpolation, trapz

ArrayLike = Union[np.ndarray, u.Quantity]
PST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_EW_ATLAS = os.path.join(PST_DATA_DIR, "lick", "atlas_spectral_indices.csv")

def _load_ew_atlas(atlas=DEFAULT_EW_ATLAS, **kwargs_atlas):
    """Load the built-in equivalent-width atlas.

    Parameters
    ----------
    atlas : str
        Path to the atlas file.

    Returns
    -------
    table : astropy.table.Table
        Table containing the equivalent-width indices.
    """
    atlas = ascii.read(atlas, **kwargs_atlas)
    if not kwargs_atlas.get("keep_emission", False):
        # Filter out emission lines
        atlas = atlas[atlas["catalog"] != "Emission"]
    return atlas

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

def download_svo_filter(name: str, dest_dir: str, verbose=True, retry=3):
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
    url = base_url + name.replace("_", "/", 1) # replace first underscore with slash to match SVO URL structure
    filename = name + ".dat"
    file_path = os.path.join(dest_dir, filename)
    if verbose:
        print(f"Querying SVO Filter: {url}")
    try:
        r = requests.get(url, stream=True, timeout=30.0)
    except requests.exceptions.ConnectTimeout as e:
        if retry:
            print(f"Connection timed out. Retrying... ({retry} attempts left)")
            download_svo_filter(name, dest_dir, verbose=verbose, retry=retry-1)
        else:
            raise e

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
    >>> wl = np.linspace(5000, 7000, 1000) * u.angstrom
    >>> ps_r_filter.interpolate(wl)
    >>> ps_r_filter.plot(add_props=True, show=True)
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

        if self.wavelength is not None:
            if self.response is None:
                self.interpolate(self.wavelength)
            else:
                self.norm_photons = self.get_photons(
                    3631 * u.Jy * np.ones(self.wavelength.shape) * constants.c / self.wavelength**2,
                    mask_nan=False)[0]
        else:
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
            photon_flux = trapz(
                spectra[mask] / (constants.h * constants.c / self.wavelength[mask]
                                   ) * self.response[mask],
                x=self.wavelength[mask])
        else:
            photon_flux = trapz(
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

            photon_flux_err = trapz(
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

        f_lambda = trapz(spectra[mask] * self.wavelength[mask] * self.response[mask], x=self.wavelength[mask]
                               ) / trapz(self.response[mask] * self.wavelength[mask], x=self.wavelength[mask])

        if spectra_err is not None:

            spectra_err = check_unit(spectra_err, default_unit=u.Lsun / u.angstrom / u.cm**2,
                                           equivalence=u.spectral_density, wav=self.wavelength)
            if mask_nan:
                mask = mask & np.isfinite(spectra_err)
            else:
                mask = np.ones_like(spectra_err, dtype=bool)

            f_lambda_err = trapz(
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
        add_props: bool
            If True, add vertical lines indicating the effective wavelength and
            bandwidth of the filter. Default is False.
        ax: :class:`matplotlib.axes.Axes`, optional
            Matplotlib axis to plot on. If None, a new figure and axis are created.
        show: bool
            If True, display the plot by calling ``plt.show()``. This requires
            an interactive matplotlib session. Default is False.
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

        self.interpolate(self.wavelength)


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
        wl = check_unit(wavelength, u.AA)

        if wl.ndim != 1:
            raise ValueError("wavelength must be 1D")
        if wl.size < 2:
            raise ValueError("wavelength grid must contain at least two points")

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

        # Cache delta_lambda for integral
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
    def __init__(self, left_wl_range, central_wl_range, right_wl_range, name=""):
        self.left_wl_range = np.array(left_wl_range)
        self.central_wl_range = np.array(central_wl_range)
        self.right_wl_range = np.array(right_wl_range)
        self.name = name

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

        wavelength = check_unit(wavelength, u.angstrom)

        if wavelength.ndim != 1:
            raise ValueError("wavelength must be 1D")
        if spectra.shape[0] != wavelength.size:
            raise ValueError("spectra first axis must match wavelength size")

        left_mask = ((wavelength >= self.left_wl_range[0])
                     & (wavelength <= self.left_wl_range[1]))
        right_mask = ((wavelength >= self.right_wl_range[0])
                      & (wavelength <= self.right_wl_range[1]))
        central_mask = ((wavelength >= self.central_wl_range[0])
                        & (wavelength <= self.central_wl_range[1]))

        if not np.any(left_mask):
            raise ValueError(f"No overlap between left_wl_range {self.left_wl_range} and wavelength grid {wavelength[0]}-{wavelength[-1]}")
        if not np.any(right_mask):
            raise ValueError(f"No overlap between right_wl_range {self.right_wl_range} and wavelength grid {wavelength[0]}-{wavelength[-1]}")
        if not np.any(central_mask):
            raise ValueError(f"No overlap between central_wl_range {self.central_wl_range} and wavelength grid {wavelength[0]}-{wavelength[-1]}")

        # Reshape spectra to 2D for vectorized computation.
        original_shape = spectra.shape[1:] if spectra.ndim > 1 else None
        spectra_2d = spectra.reshape(wavelength.size, -1)

        # Pseudocontinuum anchors from side bands.
        left_mean_wl = self.left_wl_range.mean()
        right_mean_wl = self.right_wl_range.mean()
        central_wl = wavelength[central_mask]
        t = (central_wl - left_mean_wl) / (right_mean_wl - left_mean_wl)

        left_cont = np.nanmean(spectra_2d[left_mask, :], axis=0)
        right_cont = np.nanmean(spectra_2d[right_mask, :], axis=0)
        central_flux = spectra_2d[central_mask, :]
        # Interpolate pseudocontinuum across the central band.
        pseudocont = ((1 - t)[:, np.newaxis] * left_cont[np.newaxis, :]
                      + t[:, np.newaxis] * right_cont[np.newaxis, :])

        # Compute EW
        integrand = 1 - central_flux / pseudocont
        ew = trapz(integrand, x=central_wl, axis=0)

        if spectra_err is None:
            ew_err = np.nan
        else:
            if spectra_err.shape != spectra.shape:
                raise ValueError("spectra_err must have the same shape as spectra")

            spectra_err_2d = spectra_err.reshape(wavelength.size, -1)

            left_var_samples = spectra_err_2d[left_mask, :] ** 2
            right_var_samples = spectra_err_2d[right_mask, :] ** 2
            central_var_samples = spectra_err_2d[central_mask, :] ** 2

            # Variance associated to the mean, sum(sigma_i^2) / N^2
            n_left = np.sum(np.isfinite(left_var_samples), axis=0)
            n_right = np.sum(np.isfinite(right_var_samples), axis=0)

            left_cont_var = np.where(
                n_left > 0,
                np.nansum(left_var_samples, axis=0) / n_left**2,
                np.nan,
            )
            right_cont_var = np.where(
                n_right > 0,
                np.nansum(right_var_samples, axis=0) / n_right**2,
                np.nan,
            )
            # Propagate the error associated to the continuum
            pseudocont_var = ((1 - t)[:, np.newaxis]**2 * left_cont_var[np.newaxis, :]
                              + t[:, np.newaxis]**2 * right_cont_var[np.newaxis, :])

            # Trapezoidal integration weights to propagate per-pixel variance.
            if central_wl.size < 2:
                trapz_w = np.zeros_like(central_wl)
            else:
                trapz_w = np.empty_like(central_wl)
                trapz_w[0] = 0.5 * (central_wl[1] - central_wl[0])
                trapz_w[-1] = 0.5 * (central_wl[-1] - central_wl[-2])
                if central_wl.size > 2:
                    trapz_w[1:-1] = 0.5 * (central_wl[2:] - central_wl[:-2])

            flux_term = ((trapz_w[:, np.newaxis] / pseudocont)**2) * central_var_samples
            cont_term = ((trapz_w[:, np.newaxis] * central_flux / pseudocont**2)**2) * pseudocont_var
            ew_var = np.nansum(flux_term + cont_term, axis=0)
            ew_err = np.sqrt(ew_var)

        if spectra.ndim == 1:
            ew = ew[0]
            if spectra_err is not None:
                ew_err = ew_err[0]
        else:
            ew = ew.reshape(original_shape)
            if spectra_err is not None:
                ew_err = ew_err.reshape(original_shape)

        return ew, ew_err

    def plot_ew(self, wavelength, spectra, spectra_err=None, ax=None, show=False):
        """Plot the equivalent width computation.
        
        Parameters
        ----------
        wavelength : array or Quantity
            Wavelength grid associated with ``spectra``.
        spectra : array or Quantity
            Input spectra with shape ``(n_wave, ...)``.
        spectra_err : array or Quantity, optional
            Uncertainty array with same shape as ``spectra``.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to plot on. If None, a new figure and axis are created.
        show : bool, optional
            If True, display the plot by calling ``plt.show()``. This requires
            an interactive matplotlib session. Default is False.
        """
        ew, ew_err = self.compute_ew(wavelength, spectra, spectra_err=spectra_err)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        # Remove units for plotting
        wl = wavelength.to_value(u.angstrom) if isinstance(wavelength, u.Quantity) else wavelength
        first_idx = np.searchsorted(wl, self.left_wl_range[0].value)
        first_idx = max(first_idx - 2, 0)
        last_idx = np.searchsorted(wl, self.right_wl_range[1].value)
        last_idx = min(last_idx + 2, wl.size - 1)
        wl = wl[first_idx:last_idx]
        spec = spectra.value[first_idx:last_idx] if isinstance(spectra, u.Quantity) else spectra[first_idx:last_idx]

        ax.plot(wl, spec, label="Spectra")
        # add the error
        if spectra_err is not None:
            spec_err = spectra_err.value[first_idx:last_idx] if isinstance(spectra_err, u.Quantity) else spectra_err[first_idx:last_idx]
            ax.fill_between(wl, spec - spec_err, spec + spec_err, color='gray', alpha=0.3,
                            label="Spectra error")

        ax.axvspan(self.left_wl_range[0].value, self.left_wl_range[1].value, color='blue', alpha=0.3,
                    label="Left continuum")
        ax.axvspan(self.central_wl_range[0].value, self.central_wl_range[1].value, color='green', alpha=0.3,
                    label="Central band")
        ax.axvspan(self.right_wl_range[0].value, self.right_wl_range[1].value, color='red', alpha=0.3,
                    label="Right continuum")
        ax.annotate(f"{self.name} EW = {ew:.2f} ± {ew_err:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        ax.set_xlabel(f"Wavelength ({wavelength.unit})")
        ax.set_ylabel(f"Spectra ({spectra.unit})")
        ax.set_xlim(self.left_wl_range[0].value - 10, self.right_wl_range[1].value + 10)
        ax.set_ylim()
        ax.legend()
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def to_json(self, path):
        """Save the :class:`EquivalentWidth` to a JSON file.
        
        Parameters
        ----------
        path : str
            Path to the JSON file.
        """
        data = {
            "left_wl_range": self.left_wl_range.value.tolist(),
            "central_wl_range": self.central_wl_range.value.tolist(),
            "right_wl_range": self.right_wl_range.value.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

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

    @classmethod
    def from_atlas(cls, name, atlas=DEFAULT_EW_ATLAS, **kwargs_atlas):
        """Load a :class:`EquivalentWidth` from an atlas file.
        
        Parameters
        ----------
        atlas : str
            Path to the atlas file.

        Returns
        -------
        ew : :class:`EquivalentWidth`
        """
        # Load the atlas file
        atlas = _load_ew_atlas(atlas, **kwargs_atlas)
        if "id" in atlas.colnames:
            position = name == np.asarray(atlas["id"]).astype(str)
        else:
            position = name == np.asarray(atlas["name"]).astype(str)
        if not np.any(position) and "catalog" in atlas.colnames:
            joined_name = np.asarray(atlas["catalog"]).astype(str) + "_" + np.asarray(atlas["name"]).astype(str)
            position = name == joined_name
        if not np.any(position):
            raise ValueError(f"Index {name} not found in atlas {atlas}")
        row = atlas[position][-1]
        return cls(
            left_wl_range=(row["left_wl_begin"], row["left_wl_end"]),
            central_wl_range=(row["central_wl_begin"], row["central_wl_end"]),
            right_wl_range=(row["right_wl_begin"], row["right_wl_end"]),
            name=name
        )

class EquivalentWidthList(object):
    """A list of :class:`EquivalentWidth` instances for faster computations."""

    def __init__(self, equivalent_widths):
        self.equivalent_widths = list(equivalent_widths)
        if len(self.equivalent_widths) == 0:
            raise ValueError("equivalent_widths cannot be empty")
        self.names = [ew.name for ew in self.equivalent_widths]

    @property
    def n_indices(self):
        """Number of equivalent-width indices in the list."""
        return len(self.equivalent_widths)

    @classmethod
    def from_names(cls, names):
        """Build a list from built-in JSON index names in ``data/lick``."""
        return cls([EquivalentWidth.from_name(name) for name in names])

    @classmethod
    def from_atlas(cls, names, atlas=DEFAULT_EW_ATLAS, **kwargs_atlas):
        """Build a list from index names in an atlas table."""
        return cls(
            [EquivalentWidth.from_atlas(name, atlas=atlas, **kwargs_atlas) for name in names],
        )

    def compute_ew(self, wavelength, spectra, spectra_err=None):
        """Compute equivalent widths for all indices in the list.

        Parameters
        ----------
        wavelength : array or Quantity
            Wavelength grid associated with ``spectra``.
        spectra : array or Quantity
            Input spectra with shape ``(n_wave, ...)``.
        spectra_err : array or Quantity, optional
            Uncertainty array with same shape as ``spectra``.

        Returns
        -------
        ew : Quantity
            Equivalent widths with shape ``(..., n_indices)``.
        ew_err : Quantity or float
            Equivalent-width uncertainties with shape ``(..., n_indices)``, or
            ``np.nan`` if ``spectra_err`` is not provided.
        """
        wl = check_unit(wavelength, u.angstrom)
        if wl.ndim != 1:
            raise ValueError("wavelength must be 1D")

        ew_all = []
        ew_err_all = []

        for ew in self.equivalent_widths:
            ew_idx, ew_err_idx = ew.compute_ew(
                wl,
                spectra,
                spectra_err=spectra_err,
            )
            ew_all.append(ew_idx)
            ew_err_all.append(ew_err_idx)

        ew = u.Quantity(ew_all, copy=False)
        if spectra_err is None:
            ew_err = np.nan
        else:
            ew_err = u.Quantity(ew_err_all, copy=False)

        original_shape = spectra.shape[1:] if spectra.ndim > 1 else ()

        if original_shape == ():
            ew = ew
            if spectra_err is not None:
                ew_err = ew_err
        else:
            ew = np.moveaxis(ew.reshape((self.n_indices,) + original_shape), 0, -1)
            if spectra_err is not None:
                ew_err = np.moveaxis(ew_err.reshape((self.n_indices,) + original_shape), 0, -1)

        return ew, ew_err


def show_available_equivalent_widths():
    """Print the list of available equivalent-width indices in the built-in atlas."""
    atlas = _load_ew_atlas()
    print("Available equivalent-width indices:")
    for row in atlas:
        index_name = row["id"] if "id" in atlas.colnames else f"{row['catalog']}_{row['name']}"
        print(f" - name: {index_name}, left_wl_range: ({row['left_wl_begin']}, {row['left_wl_end']}), right_wl_range: ({row['right_wl_begin']}, {row['right_wl_end']}), central_wl_range: ({row['central_wl_begin']}, {row['central_wl_end']})")

class FluxRatio(object):
    r"""Flux ratio between two spectral regions.

    Description
    -----------
    Given a stellar spectra :math:`F_\lambda`, the flux ratio is defined as the
    ratio of the average flux in two spectral regions, defined by their wavelength
    ranges. It is computed as:

    .. math::
        R = \frac{\langle F_\lambda \rangle_{\rm red}}{\langle F_\lambda \rangle_{\rm blue}}

    where :math:`\langle F_\lambda \rangle` is the average flux in the
    specified spectral region.

    Example
    -------
    >>> from pst.observables import FluxRatio
    >>> fr = FluxRatio(red_wl_range=(4000, 4100),
    ...                blue_wl_range=(4200, 4300))
    >>> wavelength = np.linspace(3900, 4400, 1000) * u.angstrom
    >>> spectra = np.random.normal(1, 0.1, size=wavelength.size) * u.erg / u.s / u.cm**2 / u.angstrom
    >>> fr_value, fr_err = fr.compute_flux_ratio(wavelength, spectra)
    """
    def __init__(self, blue_wl_range=None, red_wl_range=None, name=None,
                 region1_wl_range=None, region2_wl_range=None):
        # Backward compatibility with previous argument names.
        if red_wl_range is None and region1_wl_range is not None:
            red_wl_range = region1_wl_range
        if blue_wl_range is None and region2_wl_range is not None:
            blue_wl_range = region2_wl_range

        if red_wl_range is None or blue_wl_range is None:
            raise ValueError("Both red_wl_range and blue_wl_range must be provided")

        self.red_wl_range = np.array(red_wl_range)
        self.blue_wl_range = np.array(blue_wl_range)
        self.name = name

    @property
    def red_wl_range(self) -> u.Quantity:
        """Spectral window used for the numerator average flux."""
        return self._red_wl_range

    @red_wl_range.setter
    def red_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._red_wl_range = value * u.angstrom
        else:
            self._red_wl_range = value

    @property
    def blue_wl_range(self) -> u.Quantity:
        """Spectral window used for the denominator average flux."""
        return self._blue_wl_range

    @blue_wl_range.setter
    def blue_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._blue_wl_range = value * u.angstrom
        else:
            self._blue_wl_range = value

    @property
    def region1_wl_range(self) -> u.Quantity:
        """Backward compatible alias for ``red_wl_range``."""
        return self.red_wl_range

    @region1_wl_range.setter
    def region1_wl_range(self, value):
        self.red_wl_range = value

    @property
    def region2_wl_range(self) -> u.Quantity:
        """Backward compatible alias for ``blue_wl_range``."""
        return self.blue_wl_range

    @region2_wl_range.setter
    def region2_wl_range(self, value):
        self.blue_wl_range = value

    def compute_flux_ratio(self, wavelength, spectra, spectra_err=None):
        """Compute the flux ratio between two spectral windows.

        Parameters
        ----------
        wavelength : array or Quantity
            Wavelength grid associated with ``spectra``.
        spectra : array or Quantity
            Input spectra with shape ``(n_wave, ...)``.
        spectra_err : array or Quantity, optional
            Uncertainty array with same shape as ``spectra``.

        Returns
        -------
        ratio : Quantity or ndarray
            Flux ratio with shape ``...``.
        ratio_err : Quantity or ndarray or float
            Flux-ratio uncertainty with shape ``...`` if ``spectra_err`` is
            provided, otherwise ``np.nan``.
        """
        wl = check_unit(wavelength, u.angstrom)
        if wl.ndim != 1:
            raise ValueError("wavelength must be 1D")
        if spectra.shape[0] != wl.size:
            raise ValueError("spectra first axis must match wavelength size")

        red_mask = ((wl >= self.red_wl_range[0])
                    & (wl <= self.red_wl_range[1]))
        blue_mask = ((wl >= self.blue_wl_range[0])
                     & (wl <= self.blue_wl_range[1]))

        if not np.any(red_mask):
            raise ValueError("red_wl_range does not overlap wavelength grid")
        if not np.any(blue_mask):
            raise ValueError("blue_wl_range does not overlap wavelength grid")

        original_shape = spectra.shape[1:] if spectra.ndim > 1 else ()
        spectra_2d = spectra.reshape(wl.size, -1)

        flux_red = np.nanmean(spectra_2d[red_mask, :], axis=0)
        flux_blue = np.nanmean(spectra_2d[blue_mask, :], axis=0)
        ratio = flux_red / flux_blue

        if spectra_err is None:
            ratio_err = np.nan
        else:
            if spectra_err.shape != spectra.shape:
                raise ValueError("spectra_err must have the same shape as spectra")

            spectra_err_2d = spectra_err.reshape(wl.size, -1)
            red_var_samples = spectra_err_2d[red_mask, :] ** 2
            blue_var_samples = spectra_err_2d[blue_mask, :] ** 2

            n_red = np.sum(np.isfinite(red_var_samples), axis=0)
            n_blue = np.sum(np.isfinite(blue_var_samples), axis=0)

            flux_red_var = np.where(
                n_red > 0,
                np.nansum(red_var_samples, axis=0) / n_red**2,
                np.nan,
            )
            flux_blue_var = np.where(
                n_blue > 0,
                np.nansum(blue_var_samples, axis=0) / n_blue**2,
                np.nan,
            )

            ratio_var = ratio**2 * (flux_red_var / flux_red**2 + flux_blue_var / flux_blue**2)
            ratio_err = np.sqrt(ratio_var)

        if original_shape == ():
            ratio = ratio[0]
            if spectra_err is not None:
                ratio_err = ratio_err[0]
        else:
            ratio = ratio.reshape(original_shape)
            if spectra_err is not None:
                ratio_err = ratio_err.reshape(original_shape)

        return ratio, ratio_err

    def to_json(self, path):
        """Save the :class:`FluxRatio` definition to a JSON file."""
        data = {
            "red_wl_range": self.red_wl_range.value.tolist(),
            "blue_wl_range": self.blue_wl_range.value.tolist(),
            "name": self.name,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path):
        """Load a :class:`FluxRatio` from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Backward compatibility with old JSON keys.
        if "region1_wl_range" in data and "red_wl_range" not in data:
            data["red_wl_range"] = data.pop("region1_wl_range")
        if "region2_wl_range" in data and "blue_wl_range" not in data:
            data["blue_wl_range"] = data.pop("region2_wl_range")
        return cls(**data)

class D4000Index(FluxRatio):
    r"""D4000 index, a specific flux ratio between two spectral regions.

    Description
    -----------
    The D4000 index is defined as the ratio of the average flux in the red and blue
    spectral regions around 4000 Angstroms. It is computed as:

    .. math::
        D4000 = \frac{\langle F_\lambda \rangle_{\rm red}}{\langle F_\lambda \rangle_{\rm blue}}

    where the blue region is defined by :math:`3750-3950` Angstroms and the red
    region by :math:`4050-4250` Angstroms.

    Example
    -------
    >>> from pst.observables import D4000Index
    >>> d4000 = D4000Index()
    >>> wavelength = np.linspace(3800, 4200, 1000) * u.angstrom
    >>> spectra = np.random.normal(1, 0.1, size=wavelength.size) * u.erg / u.s / u.cm**2 / u.angstrom
    >>> d4000_value, d4000_err = d4000.compute_flux_ratio(wavelength, spectra)
    """
    def __init__(self):
        super().__init__(red_wl_range=(4050, 4250), blue_wl_range=(3750, 3950), name="D4000")

class HKIndex(FluxRatio):
    r"""HK index, a specific flux ratio between two spectral regions.

    Description
    -----------
    The HK index is defined as the ratio of the average flux in the red and blue
    spectral regions around 4000 Angstroms. It is computed as:

    .. math::
        HK = \frac{\langle F_\lambda \rangle_{\rm red}}{\langle F_\lambda \rangle_{\rm blue}}

    where the blue region is defined by :math:`3920-3945` Angstroms and the red
    region by :math:`3955-3980` Angstroms.

    Example
    -------
    >>> from pst.observables import HKIndex
    >>> hk = HKIndex()
    >>> wavelength = np.linspace(3800, 4200, 1000) * u.angstrom
    >>> spectra = np.random.normal(1, 0.1, size=wavelength.size) * u.erg / u.s / u.cm**2 / u.angstrom
    >>> hk_value, hk_err = hk.compute_flux_ratio(wavelength, spectra)
    """
    def __init__(self):
        super().__init__(red_wl_range=(3955., 3980.), blue_wl_range=(3920, 3945.), name="HK")
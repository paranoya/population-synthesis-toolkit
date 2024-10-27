"""
This module contains some tools for computing observable quantities
(e.g. photometry, equivalent widths) from spectra.
"""

import numpy as np
import os
from astropy import units as u
from astropy import constants
import requests
import json
from matplotlib import pyplot as plt
from . import utils

PST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def list_of_available_filters():
    """List the currently available filters in the default directory."""
    filter_dir = os.path.join(PST_DATA_DIR, "filters")
    return os.listdir(filter_dir)

def load_photometric_filters(filters):
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
                 filter_wavelength=None, filter_response=None):

        self.wavelength = wavelength
        self.response = response
        self.filter_wavelength = filter_wavelength
        self.filter_resp = filter_response
        if filter_wavelength is None and filter_response is None and wavelength is not None:
            self.filter_wavelength = self.wavelength
            self.filter_resp = self.response
        elif filter_wavelength is None and wavelength is None:
            raise NameError("wavelength or filter_wavelength must be provided")

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if not isinstance(value, u.Quantity) and value is not None:
            self._wavelength = value << u.angstrom
        else:
            self._wavelength = value

    @property
    def filter_wavelength(self):
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
        wavelength_units : :class:``astropy.units.Quantity``
        **kwargs : 
            Arguments to be passed to :func:`numpy.loadtxt`

        Returns
        -------
        filter : :class:`Filter`
            The ``Filter`` containing the input information.
        """
        wavelength, response = np.loadtxt(path, usecols=(0, 1), unpack=True,
                                          **kwargs)
        return cls(filter_wavelength=wavelength * wavelength_unit,
                   filter_response=response)

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
        """Compute the effective wavelength of the filter.
        
        Description
        -----------
        The effective wavelength is computed as
        .. :math:
            eff_wl = \frac{\int{R(\lambda) \cdot \lambda d\lambda}}{\int{R(\lambda) d\lambda}}
        
        Returns
        -------
        eff_wl : :class:`astropy.units.Quantity`
            The effective wavelength of the filter.
        """
        return np.sum(self.filter_wavelength*self.filter_resp)/np.sum(self.filter_resp)

    def effective_bandwidth(self):
        """Compute the effective bandwidth of the filter.
        
        Description
        -----------
        The effective bandwith is computed as
        .. :math:
            eff_bw = \sqrt{8\log(2)} \left(\frac{\int{R(\lambda) \cdot \lambda^2 d\lambda}}{\int{R(\lambda) d\lambda}} - eff_wl\right)^{1/2}

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
        """Compute the effective bandwidth of the filter.
        
        Description
        -----------
        The effective transmission is computed as
        .. :math:
            \frac{\int{R(\lambda)^2 d\lambda}}{\int{R(\lambda) d\lambda}}

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

        self.response = utils.flux_conserving_interpolation(
            wavelength, self.filter_wavelength, self.filter_resp)
        self.wavelength= wavelength
        return self.response

    def _check_spectra(self, spectra, default_unit=u.Lsun / u.angstrom / u.cm**2):
        if spectra is not None and not isinstance(spectra, u.Quantity):
            return  spectra * default_unit
        else:
            return spectra

    def get_photons(self, spectra, spectra_err=None, mask_nan=True):
        """Compute the photon flux from an input spectra.
        
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
        spectra = self._check_spectra(spectra)
        spectra_err = self._check_spectra(spectra_err)
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
        """Compute the synthetic AB magnitude from an input spectra.
        
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
        norm_photons, _ = self.get_photons(
            3630.781 * u.Jy * np.ones(spectra.shape) * constants.c / self.wavelength**2,
            mask_nan=False)
        mag_ab = - 2.5 * np.log10(n_photons / norm_photons)
        if n_photons_err is None:
            mag_ab_err = None
        else:
            mag_ab_err = 2.5 / np.log(10) * n_photons_err / n_photons
        return mag_ab, mag_ab_err

    def get_fnu(self, spectra, spectra_err=None, mask_nan=True):
        """Compute the  specific flux per frequency unit from a spectra.

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
        norm_photons, _ = self.get_photons(
            3630.781 * u.Jy * np.ones(spectra.shape) * constants.c / self.wavelength**2,
            mask_nan=False)
        f_nu = n_photons / norm_photons * 3630.781 * u.Jy
        if spectra_err is None:
            f_nu_err = None
        else:
            f_nu_err = n_photons_err / norm_photons * 3630.781 * u.Jy
        return f_nu, f_nu_err

    def get_flambda_vegamag(self, spectra, spectra_err=None, mask_nan=True):
        """Compute the  specific flux per wavelength unit from a spectra.

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
        spectra = self._check_spectra(spectra)
        spectra_err = self._check_spectra(spectra_err)
        if mask_nan:
            mask = np.isfinite(spectra)
        else:
            mask = np.ones_like(spectra, dtype=bool)

        f_lambda = np.trapz(spectra[mask] * self.wavelength[mask] * self.response[mask], x=self.wavelength[mask]
                               ) / np.trapz(self.response[mask] * self.wavelength[mask], x=self.wavelength[mask])

        if spectra_err is not None:
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

    def plot(self, show=False):
        """Plot the filter response curve.
        
        Plot the original filter response curve together with the interpolated
        version computed using a new grid of wavelengths.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(self.filter_wavelength, self.filter_resp, label='Original',
                color='k', where="mid")
        ax.plot(self.filter_wavelength, self.filter_resp, '.', color='k')
        ax.set_xlabel(f"Wavelength ({self.filter_wavelength.unit})")
        ax.set_ylabel("Filter response")
        if self.wavelength is not None:
            ax.step(self.wavelength, self.response, label='Interpolated',
                      color='r', where="mid")
        ax.legend()
        if show:
            plt.show()
        else:
            plt.close()
        return fig


class TopHatFilter(Filter):
    """Top hat photometric filter
    
    See also
    --------
    :class:`Filter`
    """
    def __init__(self, central_wave, width, **kwargs):
        central_wave = utils.check_unit(central_wave, u.Angstrom)
        width = utils.check_unit(width, u.Angstrom)

        self.wavelength = kwargs.get('wavelength', None)
        if self.wavelength is None:
            self.filter_wavelength = np.linspace(central_wave - width,
                                    central_wave + width,
                                    50)
        else:
            self.wavelength = utils.check_unit(self.wavelength, u.Angstrom)
            self.filter_wavelength = self.wavelength.copy()

        self.filter_resp = np.ones(self.filter_wavelength.size)
        self.filter_resp[self.filter_wavelength < central_wave - width / 2] = 0
        self.filter_resp[self.filter_wavelength > central_wave + width / 2] = 0
        if self.wavelength is None:
            self.response = self.filter_resp.copy()


class EquivalentWidth(object):
    """Equivalent width of an spectral region.
    
    Attributes
    ----------
    left_wl_range : :class:`np.ndarray` or :class:``astropy.units.Quantity``
        Spectral range defining the left pseudocontinuum window.
    right_wl_range : :class:`np.ndarray` or :class:``astropy.units.Quantity``
        Spectral range defining the right pseudocontinuum window.
    central_wl_range : :class:`np.ndarray` or :class:``astropy.units.Quantity``
        Spectral range defining the equivalent width window.
    """
    def __init__(self, left_wl_range, central_wl_range, right_wl_range):
        self.left_wl_range = np.array(left_wl_range)
        self.central_wl_range = np.array(central_wl_range)
        self.right_wl_range = np.array(right_wl_range)
    
    @property
    def left_wl_range(self):
        return self._left_wl_range
    
    @left_wl_range.setter
    def left_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._left_wl_range = value * u.angstrom
        else:
            self._left_wl_range = value

    @property
    def right_wl_range(self):
        return self._right_wl_range

    @right_wl_range.setter
    def right_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._right_wl_range = value * u.angstrom
        else:
            self._right_wl_range = value
    
    @property
    def central_wl_range(self):
        return self._central_wl_range

    @central_wl_range.setter
    def central_wl_range(self, value):
        if not isinstance(value, u.Quantity):
            self._central_wl_range = value * u.angstrom
        else:
            self._central_wl_range = value

    def compute_ew(self, wavelength, spectra, spectra_err=None):
        """Compute the equivalent width of a given input spectra.
        
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


if __name__ == '__main__':
    from pst.SSP import BaseGM
    import matplotlib.pyplot as plt
    ssp = BaseGM()
    
    filter = Filter.from_svo("PANSTARRS_PS1.r")
    filter.interpolate(ssp.wavelength)

    for sed in ssp.L_lambda.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2])):
        sed = 1 * u.Msun * sed / 4 / np.pi / (10 * u.pc)**2
        mag, mag_err = filter.get_ab(sed)
        

#        print("SSP absolute magnitude: ", mag)
    plt.figure()
    plt.plot(ssp.wavelength, filter.response)
    plt.plot(ssp.wavelength, sed / np.mean(sed))
    plt.yscale('log')
    plt.show()

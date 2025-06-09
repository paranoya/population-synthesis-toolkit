import os
from glob import glob
from copy import deepcopy
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy import units

from pst.utils import check_unit

class SSPBase(object):
    """Base class that represents a model of Simple Stellar Populations.
    
    Description
    -----------
    This class is meant for representing a model of Simple Stellar Populations
    that is mainly composed by a discreted grid of ages and metallicities and
    their associated spectral energy distributions.

    Attributes
    ----------
    ages: astropy.units.Quantity
        Ages of the SSPs.
    metallicities: astropy.units.Quantity
        Metallicities of the SSPs.
    L_lambda: astropy.units.Quantity
        Spectral energy distribution of each SSP. Each dimension correspond to
        (metallicity, ages, wavelength).
    wavelength: astroy.units.Quantity
        Wavelength array associated to the SED of the SSPs.
    """
    default_path = os.path.join(os.path.dirname(__file__), "data", "ssp")

    @property
    def ages(self):
        return self._ages

    @ages.setter
    def ages(self, ages_array):
        if not isinstance(ages_array, units.Quantity):
            raise NameError("Ages must be an astropy.Quantity")
        else:
            self._ages = ages_array

    @property
    def metallicities(self):
        return self._metallicities

    @metallicities.setter
    def metallicities(self, metallicities_array):
        if not isinstance(metallicities_array, units.Quantity):
            raise NameError("metallicities must be an astropy.Quantity")
        else:
            self._metallicities = metallicities_array

    @property
    def L_lambda(self):
        return self._L_lambda

    @L_lambda.setter
    def L_lambda(self, l_lamdba):
        if not isinstance(l_lamdba, units.Quantity):
            raise NameError("L_lambda must be an astropy.Quantity")
        else:
            self._L_lambda = l_lamdba

    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, wave):
        if not isinstance(wave, units.Quantity):
            raise NameError("wavelength must be an astropy.Quantity")
        else:
            self._wavelength = wave
       
    def get_weights(self, ages, metallicities, masses=None):
        """2D interpolation of a list of ages and metallicities.
        
        Parameters
        ----------
        ages : np.array or astropy.units.Quantity
            SSP ages to interpolate.
        metallicites : np.array or astropy.units.Quantity
            Metallicity associated to each age.
        masses : np.array, astropy.units.Quantity or None, optional
            Stellar mass corresponding to each SSP.
        """
        ages = check_unit(ages, u.Gyr)
        metallicities = check_unit(metallicities, u.dimensionless_unscaled)
        if masses is None:
            masses = np.ones(ages.size) << u.Msun
        else:
            masses = check_unit(masses, u.Msun)

        age_idx = np.clip(self.ages.searchsorted(ages), 1, self.ages.size-1)
        weights_age = np.log(ages / self.ages[age_idx-1])
        weights_age /= np.log(self.ages[age_idx] / self.ages[age_idx-1])
        weights_age = np.clip(weights_age, 0., 1.)

        z_idx = np.clip(self.metallicities.searchsorted(metallicities), 1, self.metallicities.size-1)
        weights_z = np.log(metallicities / self.metallicities[z_idx-1])
        weights_z /= np.log(self.metallicities[z_idx] / self.metallicities[z_idx-1])
        weights_z = np.clip(weights_z, 0., 1.)

        weights = np.zeros((self.metallicities.size, self.ages.size)) << masses.unit
        np.add.at(weights, (z_idx, age_idx), masses * weights_age * weights_z)
        np.add.at(weights, (z_idx-1, age_idx), masses * weights_age * (1-weights_z))
        np.add.at(weights, (z_idx-1, age_idx-1), masses * (1-weights_age) * (1-weights_z))
        np.add.at(weights, (z_idx, age_idx-1), masses * (1-weights_age) * weights_z)
        return weights

    def get_ssp_l_lambda(self, age, metallicity):
        """Compute the SED associated to an SSP of a given age and metallicity.
        
        Parameters
        ----------
        age : float or u.Quantity
            SSP age.
        metallicity : float or u.Quantity
            SSP metallicity
        
        Returns
        -------
        sed : u.Quantity
            Spectra energy distribution associated to the SSP.
        """
        age = check_unit(age, u.Gyr)
        metallicity = check_unit(metallicity, u.dimensionless_unscaled)

        age_idx = np.clip(self.ages.searchsorted(age), 1, self.ages.size-1)
        weights_age = np.log(age / self.ages[age_idx-1])
        weights_age /= np.log(self.ages[age_idx] / self.ages[age_idx-1])
        weights_age = np.clip(weights_age, 0., 1.)

        z_idx = np.clip(self.metallicities.searchsorted(metallicity), 1,
                        self.metallicities.size-1)
        weights_z = np.log(metallicity / self.metallicities[z_idx-1])
        weights_z /= np.log(self.metallicities[z_idx] / self.metallicities[z_idx-1])
        weights_z = np.clip(weights_z, 0., 1.)

        sed = self.L_lambda[z_idx, age_idx] *  weights_age * weights_z
        sed += self.L_lambda[z_idx-1, age_idx] * weights_age * (1-weights_z)
        sed += self.L_lambda[z_idx-1, age_idx-1] * (1-weights_age) * (1-weights_z)
        sed += self.L_lambda[z_idx-1, age_idx] * (1-weights_age) * weights_z
        return sed

    def get_ssp_logedges(self):
        """Get the edges of the SSP metallicities and ages."""
        logages = np.log10(self.ages / units.yr)
        lim = 1.5 * logages[[0, -1]] - 0.5 * logages[[1, -2]]
        logage_bin_edges = np.hstack(
                [lim[0], (logages[1:] + logages[:-1])/2, lim[1]])

        log_met_bins = np.log10(self.metallicities)
        lim = 1.5 * log_met_bins[[0, -1]] - 0.5 * log_met_bins[[1, -2]]
        logmet_bin_edges = np.hstack(
                [lim[0], (log_met_bins[1:] + log_met_bins[:-1])/2, lim[1]])
        return logmet_bin_edges, logage_bin_edges

    def regrid(self, age_bins, metallicity_bins, verbose=True):
        """Interpolate the SSP model to a new grid of input ages and metallicities.
        
        Parameters
        ----------
        age_bins : np.array or astropy.units.Quantity
        """
        if verbose:
            print("[SSP] Interpolating the SSP model to a new grid of ages and metallicities")
        age_bins = check_unit(age_bins, u.Gyr)
        metallicity_bins = check_unit(metallicity_bins, u.dimensionless_unscaled)
        # Bin the SED of the SSPs
        new_l_lambda = np.zeros((metallicity_bins.size, age_bins.size,
                                 self.wavelength.size)) * self.L_lambda.unit
        if verbose:
            print("New SSP SED shape: ", new_l_lambda.shape)
        for j, m_bin in enumerate(metallicity_bins):
            for i, a_bin in enumerate(age_bins):
                weights = self.get_weights(a_bin, m_bin, 1.0 * u.Msun)
                new_l_lambda[j, i] = np.sum(
                    self.L_lambda * weights[:, :, np.newaxis] / u.Msun, axis=(0, 1))

        if verbose:
            print("Updating SSP model metallicities, ages and SED")
        self.metallicities = metallicity_bins
        self.ages = age_bins
        self.L_lambda = new_l_lambda

    def cut_wavelength(self, wl_min=None, wl_max=None, verbose=True):
        """Cut model wavelength edges.
        
        Parameters
        ----------
        wl_min : float or astropy.units.Quantity, optional
            Minimum wavelength value.
        wl_max : float or astropy.units.Quantity, optional
            Maximum wavelength value.
        """
        if wl_min is None:
            wl_min = self.wavelength[0]
        else:
            wl_min = check_unit(wl_min, self.wavelength.unit)
        if wl_max is None:
            wl_max = self.wavelength[0]
        else:
            wl_max = check_unit(wl_max, self.wavelength.unit)

        cut_pts = np.where((self.wavelength >= wl_min) &
                           (self.wavelength <= wl_max))[0]
        if len(cut_pts) == 0:
            raise NameError(
                'Wavelength cuts {}, {} out of range for array with lims {} {}'
                .format(wl_min, wl_max, self.wavelength.min(),
                        self.wavelength.max())
                            )
        else:
            self.wavelength = self.wavelength[cut_pts]
            self.L_lambda = self.L_lambda[:, :, cut_pts]
            if verbose:
                print('[SSP] Models cut between {} {}'.format(wl_min, wl_max))

    def interpolate_sed(self, new_wl_edges, verbose=True):
        """Flux-conserving interpolation.

        Parameters
        ----------
        - new_wl_edges: bin edges of the new interpolated points.
        """
        if not isinstance(new_wl_edges, units.Quantity):
            new_wl_edges *= self.wavelength.unit

        new_wl = (new_wl_edges[1:] + new_wl_edges[:-1]) / 2
        dwl = np.diff(new_wl_edges)
        ori_dwl = np.hstack((np.diff(self.wavelength),
                             self.wavelength[-1] - self.wavelength[-2]))
        if verbose:
            print('[SSP] Interpolating SSP SEDs')
        new_l_lambda = np.empty(
            shape=(self.metallicities.size, self.ages.size,
                   new_wl.size), dtype=np.float32) * self.L_lambda.unit

        for i in range(self.L_lambda.shape[0]):
            for j in range(self.L_lambda.shape[1]):
                f = np.interp(new_wl_edges, self.wavelength,
                              np.cumsum(self.L_lambda[i, j] * ori_dwl))
                new_flux = np.diff(f) / dwl
                new_l_lambda[i, j] = new_flux

        self.L_lambda = new_l_lambda
        self.wavelength = new_wl

    def get_mass_lum_ratio(self, wl_range):
        """Compute the mass-to-light ratio within a wavelength range."""
        if not isinstance(wl_range, u.Quantity):
            print("Assuming that input wavelength range has same units as wavelength")
            wl_range = np.array(wl_range) * self.wavelength.unit
        pts = np.where((self.wavelength >= wl_range[0]) &
                       (self.wavelength <= wl_range[1]))[0]
        mass_to_lum = np.empty(
            (self.metallicities.size, self.ages.size)
            ) * 1 / (self.L_lambda.unit * self.wavelength.unit)

        for i in range(self.metallicities.size):
            for j in range(self.ages.size):
                mass_to_lum[i, j] = 1/np.mean(self.L_lambda[i, j][pts] * self.wavelength[pts])
        return mass_to_lum
    
    def get_specific_mass_lum_ratio(self, wl_range):
        """Compute the mass-to-light ratio per wavelength unit within a wavelength range."""
        if not isinstance(wl_range, u.Quantity):
            print("Assuming that input wavelength range has same units as wavelength")
            wl_range = np.array(wl_range) * self.wavelength.unit
        pts = np.where((self.wavelength >= wl_range[0]) &
                       (self.wavelength <= wl_range[1]))[0]
        mass_to_lum = np.empty(
            (self.metallicities.size, self.ages.size)
            ) * 1 / self.L_lambda.unit

        for i in range(self.metallicities.size):
            for j in range(self.ages.size):
                mass_to_lum[i, j] = 1/np.mean(self.L_lambda[i, j][pts])
        return mass_to_lum
    
    def compute_photometry(self, filter_list, z_obs=0.0, verbose=True):
        """Compute the SSP synthetic photometry of a set of filters.
        
        Parameters
        ----------
        filter_list: list of pst.observable.Filter
            A list of photometric filters.

        Returns
        -------
        photometry: np.ndarray
            Array storing the photometry. The dimensions correspond to
            filter, metallicity and age.
        """
        if verbose:
            print("Computing synthetic photometry for SSP model")
        self.photometry = np.zeros((len(filter_list),
                                    *self.L_lambda.shape[:-1])) * u.Jy / u.Msun
        self.photometry_filters = filter_list
        for ith, f in enumerate(filter_list):
            f.interpolate(self.wavelength * (1 + z_obs))
            flux, _ = f.get_fnu(
                    self.L_lambda  * u.Msun / 4 / np.pi / (10 * u.pc)**2,
                    mask_nan=False)
            flux /= 1 + z_obs
            self.photometry[ith] = flux.to('Jy') / u.Msun
        return self.photometry

    def copy(self):
        """Return a copy of the SSP model."""
        return deepcopy(self)


class PopStar(SSPBase):
    """
    PopStar SSP models (Mollá et al. 2009).

    This class represents the PopStar evolutionary synthesis models for simple
    stellar populations as presented in Mollá et al. (2009). It provides
    access to spectral energy distributions (SEDs) across a range of metallicities
    and ages, optionally including nebular emission.

    Parameters
    ----------
    IMF : str
        Initial Mass Function name. Must be one of the supported IMFs listed below:

        - 'sal' : Salpeter (1955)
        - 'fer' : Ferrini, Penco & Palla (1990)
        - 'kro' : Kroupa (2002)
        - 'cha' : Chabrier (2003)

        This keyword is used to select the corresponding PopStar model files.
    nebular : bool, optional
        If True, include nebular emission in the SEDs. Default is False, meaning
        only the stellar continuum is considered.
    path : str or None, optional
        Filesystem path to the PopStar model data. If None (default), the package
        default path plus 'PopStar' subdirectory is used.
    verbose : bool, optional
        If True (default), print informational messages during initialization.

    Example
    -------
    Initialize with the desired IMF and nebular option:

    >>> from pst.SSP import PopStar
    >>> help(PopStar) # Retrieve the class documentation
    >>> popstar = PopStar(IMF="cha", nebular=True) # Initialise the model
    >>> print(popstar.wavelength)
    <Quantity [91. , 94. , 96. , ..., 1200000., 1400000., 1600000.] Angstrom>
    >>> oldest_sed = popstar.L_lambda[-1, -1] # Get the SED of the oldest SSP with highest metallicity

    Notes
    -----
    - The PopStar models cover metallicities from 0.0001 to 0.05 and ages spanning from
      10^5 to ~10^10 years.
    - Nebular emission is included only if `nebular=True` is set.
    - Model files are expected to be in compressed FITS format with specific naming
      conventions as per Mollá et al. 2009.

    For more details see the :class:`SSPBase` base class documentation.

    References
    ----------
    Mollá, M., García-Vargas, M. L., & Bressan, A. (2009). PopStar I: evolutionary synthesis model description. `MNRAS, 398(1), 451-470 <https://academic.oup.com/mnras/article/398/1/451/1099698>`_.
    `Fractal data repository <https://www.fractal-es.com/PopStar/>`_
    """
    def __init__(self, IMF, nebular=False, path=None, verbose=True):
        if path is None:
            self.path = os.path.join(self.default_path, 'PopStar')
        else:
            self.path = path
        self.metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02,
                                       0.05]) * units.dimensionless_unscaled
        self.log_ages_yr = np.array([5.00, 5.48, 5.70, 5.85, 6.00, 6.10, 6.18,
                                     6.24, 6.30, 6.35, 6.40, 6.44, 6.48, 6.51,
                                     6.54, 6.57, 6.60, 6.63, 6.65, 6.68, 6.70,
                                     6.72, 6.74, 6.76, 6.78, 6.81, 6.85, 6.86,
                                     6.88, 6.89, 6.90, 6.92, 6.93, 6.94, 6.95,
                                     6.97, 6.98, 6.99, 7.00, 7.04, 7.08, 7.11,
                                     7.15, 7.18, 7.20, 7.23, 7.26, 7.28, 7.30,
                                     7.34, 7.38, 7.41, 7.45, 7.48, 7.51, 7.53,
                                     7.56, 7.58, 7.60, 7.62, 7.64, 7.66, 7.68,
                                     7.70, 7.74, 7.78, 7.81, 7.85, 7.87, 7.90,
                                     7.93, 7.95, 7.98, 8.00, 8.30, 8.48, 8.60,
                                     8.70, 8.78, 8.85, 8.90, 8.95, 9.00, 9.18,
                                     9.30, 9.40, 9.48, 9.54, 9.60, 9.65, 9.70,
                                     9.74, 9.78, 9.81, 9.85, 9.90, 9.95, 10.00,
                                     10.04, 10.08, 10.11, 10.12, 10.13, 10.14,
                                     10.15, 10.18]) * units.dimensionless_unscaled
        self.ages = 10**self.log_ages_yr * units.yr
        if verbose:
            print("> Initialising Popstar models (IMF='"+IMF+"')")
        # isochrone age in delta [log(tau)]=0.01
        if nebular:
            column = "_total"
            if verbose:
                print('--> Including NEBULAR emission')
        else:
            column = "_stellar"
            if verbose:
                print('--> Only stellar continuum')

        with fits.open(
            os.path.join(self.path, f"popstar_{IMF.lower()}_0.15_100.fits.gz")
            ) as hdul:
            self.wavelength = hdul[1].data["wavelength"] * u.Unit(
                hdul[1].header["TUNIT1"])

            self.L_lambda = np.empty(
                shape=(self.metallicities.size, self.log_ages_yr.size,
                    self.wavelength.size), dtype=np.float32
                    ) * u.Unit(hdul[2].header["TUNIT1"])

            for i, Z in enumerate(self.metallicities.value):
                table = hdul["SED_Z_0.{:04.0f}".format(Z*1e4)].data
                for j, age in enumerate(self.log_ages_yr.value):
                    self.L_lambda[i][j] = table[
                        "logage_yr_{:02.2f}".format(age)
                        + column] * self.L_lambda.unit


class PyPopStar(SSPBase):
    """PyPopStar SSP models (Millán-Irigoyen et al. 2021).

    This class represents the PyPopStar simple stellar population (SSP) models
    as described in Millán-Irigoyen et al. (2021). It provides spectral energy
    distributions (SEDs) for a range of metallicities and ages, optionally
    including nebular emission.

    Parameters
    ----------
    IMF : str
        Initial Mass Function identifier. Must be one of the supported IMFs:

        - 'KRO' : Kroupa (2002)
        - 'SAL' : Salpeter (1955)
        - 'CHA' : Chabrier (2003)
        This keyword selects the corresponding model subdirectory and file set.
    nebular : bool, optional
        If True, nebular emission is included in the SEDs. Default is False,
        meaning only the stellar continuum is included.
    path : str or None, optional
        Filesystem path to the PyPopStar model data. If None (default), the
        package default path plus 'PyPopStar' and IMF subdirectories are used.
    verbose : bool, optional
        If True (default), prints informational messages during initialization.

    Example
    -------
    Initialize the PyPopStar model with a chosen IMF and nebular emission option:

    >>> from pst.SSP import PyPopStar
    >>> psp = PyPopStar(IMF='KRO', nebular=False)
    >>> print(psp.wavelength)
    <Quantity [ ... ] Angstrom>
    >>> sed_example = psp.L_lambda[0, 0]  # SED for lowest metallicity, youngest age

    Notes
    -----
    - PyPopStar models cover metallicities from 0.004 to 0.05 and ages spanning
      approximately 10^5 to 10^10 years.
    - Nebular emission is included only if `nebular=True`.
    - Model files must follow the naming convention:
      `'SSP-{IMF}_Z{metallicity}_logt{log_age}.fits'`.

    For additional details, see the :class:`SSPBase` base class documentation.

    References
    ----------
    Millán-Irigoyen, I.; Mollá, M.; Cerviño, M.; et al. (2021). HR-PYPOPSTAR: high-wavelength-resolution stellar populations evolutionary synthesis model `MNRAS, 506 (4), 4781-4799 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4781M/abstract>`_.

    `Fractal data repository <https://www.fractal-es.com/PopStar/>`_
    """
    def __init__(self, IMF, nebular=False, path=None, verbose=True):
        if path is None:
            self.path = os.path.join(self.default_path, 'PyPopStar', IMF)
        else:
            self.path = path
        self.metallicities = np.array([0.004, 0.008, 0.02, 0.05]
                                      ) * units.dimensionless_unscaled
        self.log_ages_yr = np.array([
        5.,  5.48,  5.7 ,  5.85,  6.  ,  6.1 ,  6.18,  6.24,  6.3 ,
        6.35,  6.4 ,  6.44,  6.48,  6.51,  6.54,  6.57,  6.6 ,  6.63,
        6.65,  6.68,  6.7 ,  6.72,  6.74,  6.76,  6.78,  6.81,  6.85,
        6.86,  6.88,  6.89,  6.9 ,  6.92,  6.93,  6.94,  6.95,  6.97,
        6.98,  6.99,  7.  ,  7.04,  7.08,  7.11,  7.15,  7.18,  7.2 ,
        7.23,  7.26,  7.28,  7.3 ,  7.34,  7.38,  7.41,  7.45,  7.48,
        7.51,  7.53,  7.56,  7.58,  7.6 ,  7.62,  7.64,  7.66,  7.68,
        7.7 ,  7.74,  7.78,  7.81,  7.85,  7.87,  7.9 ,  7.93,  7.95,
        7.98,  8.  ,  8.3 ,  8.48,  8.6 ,  8.7 ,  8.78,  8.85,  8.9 ,
        8.95,  9.  ,  9.18,  9.3 ,  9.4 ,  9.48,  9.54,  9.6 ,  9.65,
        9.7 ,  9.74,  9.78,  9.81,  9.85,  9.9 ,  9.95, 10.  , 10.04,
       10.08, 10.11, 10.12, 10.13, 10.14, 10.15, 10.18]
       ) * units.dimensionless_unscaled
        self.ages = 10**self.log_ages_yr  * units.yr
        # isochrone age in delta [log(tau)]=0.01
        # self.wavelength = np.loadtxt(os.path.join(self.path, 'KRO', 'sp',
        #                                           'sp_z0.004_logt05.00.dat'),
        #                              dtype=float, usecols=(0,), unpack=True
        #                              ) * u.Angstrom
        header = 'SSP-{}'.format(IMF)
        if nebular:
            if verbose:
                print("> Initialising Popstar models (neb em) (IMF='"
                      + IMF + "')")
            column = 'flux_total'
        else:
            if verbose:
                print("> Initialising Popstar models (no neb em) (IMF='"
                      + IMF + "')")
            column = 'flux_stellar'
        with fits.open(os.path.join(self.path, header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(
                self.metallicities.value[0], self.log_ages_yr.value[0]))
                       ) as hdul:
            self.wavelength = hdul[1].data['wavelength'] * units.Angstrom # Angstrom

        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32) * u.Lsun / u.Angstrom / u.Msun

        for i, Z in enumerate(self.metallicities.value):
            for j, age in enumerate(self.log_ages_yr.value):
                filename = header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(Z, age)
                file = os.path.join(self.path, filename)
                with fits.open(file) as hdul:
                    self.L_lambda[i][j] = hdul[1].data[column] * self.L_lambda.unit
                    hdul.close()
        # Avoid 0 flux
        self.L_lambda[self.L_lambda <= 0] += self.L_lambda[self.L_lambda > 0].min()


class BC03_2003(SSPBase):
    """
    BC03 SSP models (Bruzual & Charlot 2003, original version).

    This class implements the 2003 version of the Bruzual & Charlot (BC03)
    evolutionary synthesis models. It supports both the STELIB and BaSeL stellar
    spectral libraries and various IMFs.

    Parameters
    ----------
    isochrone : str, optional
        Isochrone set to use:'Padova1994' (default) or 'Padova2000'.
    model : str, optional
        Stellar spectral library. Must be one of:

        - 'BaSeL'  : Low-resolution spectra
        - 'STELIB' : High-resolution spectra

        Case-insensitive. Default is 'BaSeL'.
    imf : str, optional
        Initial Mass Function. Must be one of:

        - 'chabrier' or 'cha' : Chabrier (2003)
        - 'salpeter' or 'sal' : Salpeter (1955)
        - 'kroupa' or 'kro'    : Kroupa (2002)

        Case-insensitive. Default is 'chabrier'.
    path : str or None, optional
        Filesystem path to the BC03 model data. If None (default), the package
        default path is used.
    verbose : bool, optional
        If True (default), print informational messages during initialization.

    Example
    -------
    Create a BC03 SSP object using the Chabrier IMF and BaSeL library:

    >>> from pst.SSP import BC03_2003
    >>> ssp = BC03_2003(model='BaSeL', imf='Chabrier')
    >>> print(ssp.wavelength)
    <Quantity [ ... ] Angstrom>
    >>> sed = ssp.L_lambda[0, -1]  # SED for lowest metallicity and oldest age

    Notes
    -----
    - This implementation loads the original 2003 release of BC03.
    - Supported metallicities (Padova1994 isochrones):

        - 'm22': Z = 0.0001
        - 'm32': Z = 0.0004
        - 'm42': Z = 0.004
        - 'm52': Z = 0.008
        - 'm62': Z = 0.02
        - 'm72': Z = 0.05

    - SEDs are read from FITS files with filenames of the form:
      `bc2003_{res}_{metallicity}_{imf}_ssp.fits`
    - Units for `L_lambda` are Lsun / Angstrom / Msun.
    - Ages are loaded from a standard time grid provided in `TIME_SCALE.DAT`.

    See also
    --------
    :class:`SSPBase` : Base class for all SSP models.

    References
    ----------
    Bruzual, G. & Charlot, S. (2003).
    Stellar population synthesis at the resolution of 2003.
    `MNRAS, 344(4), 1000–1028 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.344.1000B/abstract>`_.

    `Gustavo Bruzual page <https://www.bruzual.org/>`_
    """
    _metallicity_map = {
     # Padova1994
    'm22': 0.0001,
    'm32': 0.0004,
    'm42': 0.004,
    'm52': 0.008,
    'm62': 0.02,
    'm72': 0.05,
    #'m82': 0.1  Not available in the 2003 version
    }
    _resolution = {'BaSeL': 'lr', 'stelib': 'hr'}
    
    def __init__(self, isochrone='Padova1994', model='BaSeL',
                 imf='Chabrier', path=None, verbose=True) -> None:
        self.isochrone = isochrone
        self.model, model_key = self._parse_model(model)
        self.imf, imf_key = self._parse_imf(imf)

        if path is None:
            self.path = os.path.join(self.default_path,
                                     'BC03', 'bc03_2003ver', "bc03",
                                     self.isochrone, self.imf)
        else:
            self.path = path

        if verbose:
            print(f"> Initialising BC03 model {self.model} (IMF={self.imf})")
        self.metallicities = np.array(list(self._metallicity_map.values())
                                      ) * units.dimensionless_unscaled
        self.ages = np.loadtxt(
            os.path.join(self.default_path, 'BC03', 'TIME_SCALE.DAT')
            ) * units.yr

        self.log_ages_yr = np.log10(self.ages / units.yr)

        load_wavelength = False
        for i, metallicity_key in enumerate(self._metallicity_map.keys()):
            fits_path = os.path.join(
                self.path, f"bc2003_{self._resolution[model_key]}_{metallicity_key}_{imf_key}_ssp.fits")
            table = Table.read(fits_path)
            if not load_wavelength:
                self.wavelength = table['wavelength'].value * u.angstrom
                self.L_lambda = np.zeros((self.metallicities.size, self.ages.size,
                                          self.wavelength.size))  * u.Lsun / u.Angstrom / u.Msun
                load_wavelength = True
            table.remove_column("wavelength")
            for j, column in enumerate(table.itercols()):
                self.L_lambda[i, j] = column.value * self.L_lambda.unit

    def _parse_model(self, model):
        if 'basel' in model.lower():
            model = 'BaSeL'
            key = 'BaSeL'
        elif 'stelib' in model.lower():
            model = 'stelib'
            key = 'stelib'
        else:
            raise NameError(f"Unrecognized model: {model}.\n"
                             + "Select Basel or Stelib")
        return model, key

    def _parse_imf(self, imf):
        if 'cha' in imf.lower():
            imf = 'chabrier'
            key = 'chab'
        elif 'sal' in imf.lower():
            imf = 'salpeter'
            key = 'salp'
        elif 'kro' in imf.lower():
            imf = 'kroupa'
            key = 'kro'
        else:
            raise NameError(f"Unrecognized IMF: {imf}.\n"
                             + "Select Chabrier, Salpeter or Kroupa")
        return imf, key


class BC03_2013(SSPBase):
    """
    BC03 SSP models (Bruzual & Charlot 2003, 2013 updated version).

    This class implements the updated 2013 version of the Bruzual & Charlot (BC03)
    simple stellar population (SSP) models, using improved coverage of high
    metallicities and updated isochrone input. It supports both the STELIB and
    BaSeL stellar spectral libraries and several initial mass functions (IMFs).

    Parameters
    ----------
    isochrone : str, optional
        Isochrone set to use. Currently only 'Padova1994' is supported (default).
    model : str, optional
        Stellar spectral library. Must be one of:

        - 'BaSeL'  : Low-resolution spectra
        - 'STELIB' : High-resolution spectra

        Case-insensitive. Default is 'BaSeL'.
    imf : str, optional
        Initial Mass Function. Must be one of:

        - 'chabrier' or 'cha' : Chabrier (2003)
        - 'salpeter' or 'sal' : Salpeter (1955)
        - 'kroupa' or 'kro'    : Kroupa (2002)

        Case-insensitive. Default is 'chabrier'.
    path : str or None, optional
        Filesystem path to the BC03 model data. If None (default), the package
        default path is used.
    verbose : bool, optional
        If True (default), print informational messages during initialization.

    Example
    -------
    Create a BC03 SSP object using the Chabrier IMF and BaSeL library:

    >>> from pst.SSP import BC03_2013
    >>> ssp = BC03_2013(model='BaSeL', imf='Chabrier')
    >>> print(ssp.wavelength)
    <Quantity [ ... ] Angstrom>
    >>> sed = ssp.L_lambda[0, -1]  # SED for lowest metallicity and oldest age

    Notes
    -----
    - This version extends the original 2003 release by including an additional
      high-metallicity track (Z = 0.1).
    - Supported metallicities (Padova1994 isochrones):

        - 'm22': Z = 0.0001
        - 'm32': Z = 0.0004
        - 'm42': Z = 0.004
        - 'm52': Z = 0.008
        - 'm62': Z = 0.02
        - 'm72': Z = 0.05
        - 'm82': Z = 0.1

    - File naming convention:
      `bc2003_{res}_{model}_{metallicity}_{imf}_ssp.fits`
    - Units for `L_lambda` are Lsun / Angstrom / Msun.
    - Ages are taken from the same `TIME_SCALE.DAT` file as BC03_2003.

    See also
    --------
    :class:`SSPBase` : Common functionality for SSP models.

    References
    ----------
    Bruzual, G. & Charlot, S. (2003).
    Stellar population synthesis at the resolution of 2003.
    `MNRAS, 344(4), 1000–1028 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.344.1000B/abstract>`_.

    `Gustavo Bruzual page <https://www.bruzual.org/>`_
    """
    _metallicity_map = {
     # Padova1994
    'm22': 0.0001,
    'm32': 0.0004,
    'm42': 0.004,
    'm52': 0.008,
    'm62': 0.02,
    'm72': 0.05,
    'm82': 0.1}
    _resolution = {'BaSeL': 'lr', 'stelib': 'hr'}
    
    def __init__(self, isochrone='Padova1994', model='BaSeL',
                 imf='Chabrier', path=None, verbose=True) -> None:
        self.isochrone = isochrone
        self.model, model_key = self._parse_model(model)
        self.imf, imf_key = self._parse_imf(imf)

        if path is None:
            self.path = os.path.join(self.default_path,
                                     'BC03', 'bc03_2013ver', "bc03",
                                     self.isochrone, self.imf)
        else:
            self.path = path

        if verbose:
            print(f"> Initialising BC03 model {self.model} (IMF={self.imf})")
        self.metallicities = np.array(list(self._metallicity_map.values())
                                      ) * units.dimensionless_unscaled
        self.ages = np.loadtxt(
            os.path.join(self.default_path, 'BC03', 'TIME_SCALE.DAT')
            ) * units.yr

        self.log_ages_yr = np.log10(self.ages / units.yr)

        load_wavelength = False
        for i, metallicity_key in enumerate(self._metallicity_map.keys()):
            fits_path = os.path.join(
                self.path, f"bc2003_{self._resolution[model_key]}_{model_key}_{metallicity_key}_{imf_key}_ssp.fits")
            table = Table.read(fits_path)
            if not load_wavelength:
                self.wavelength = table['wavelength'].value * u.angstrom
                self.L_lambda = np.zeros((self.metallicities.size, self.ages.size,
                                          self.wavelength.size))  * u.Lsun / u.Angstrom / u.Msun
                load_wavelength = True
            table.remove_column("wavelength")
            for j, column in enumerate(table.itercols()):
                self.L_lambda[i, j] = column.value * self.L_lambda.unit

    def _parse_model(self, model):
        if 'basel' in model.lower():
            model = 'BaSeL'
            key = 'BaSeL'
        elif 'stelib' in model.lower():
            model = 'stelib'
            key = 'stelib'
        else:
            raise NameError(f"Unrecognized model: {model}.\n"
                             + "Select Basel or Stelib")
        return model, key

    def _parse_imf(self, imf):
        if 'cha' in imf.lower():
            imf = 'chabrier'
            key = 'chab'
        elif 'sal' in imf.lower():
            imf = 'salpeter'
            key = 'salp'
        elif 'kro' in imf.lower():
            imf = 'kroupa'
            key = 'kro'
        else:
            raise NameError(f"Unrecognized IMF: {imf}.\n"
                             + "Select Chabrier, Salpeter or Kroupa")
        return imf, key


class BC03_2016(SSPBase):
    """
    BC03 SSP models (Bruzual & Charlot 2003, 2016 updated version).

    This class implements the 2016 version of the Bruzual & Charlot (BC03)
    simple stellar population (SSP) models. It includes higher metallicity
    coverage and new spectral libraries, including MILES, STELIB, and BaSeL.
    This version uses updated file naming conventions and directory structure.

    Parameters
    ----------
    model : str, optional
        Stellar spectral library. Must be one of:

        - 'BaSeL'  : BaSeL 3.1 + ATLAS (low resolution)
        - 'STELIB' : STELIB + ATLAS (high resolution)
        - 'MILES'  : MILES + STELIB (high resolution)

        Case-insensitive. Default is 'BaSeL'.
    imf : str, optional
        Initial Mass Function. Must be one of:

        - 'chabrier' or 'cha' : Chabrier IMF
        - 'salpeter' or 'sal' : Salpeter IMF
        - 'kroupa' or 'kro'    : Kroupa IMF

        Case-insensitive. Default is 'Kroupa'.
    path : str or None, optional
        Filesystem path to the BC03 model data. If None (default), the package
        default path is used.
    verbose : bool, optional
        If True (default), print informational messages during initialization.

    Example
    -------
    Create a BC03 SSP object using the Chabrier IMF and BaSeL library:

    >>> from pst.SSP import BC03_2016
    >>> ssp = BC03_2016(model='BaSeL', imf='Chabrier')
    >>> print(ssp.wavelength)
    <Quantity [ ... ] Angstrom>
    >>> sed = ssp.L_lambda[0, -1]  # SED for lowest metallicity and oldest age

    Notes
    -----
    - This version supports extended metallicity coverage up to Z = 0.1.
    - The directory and file naming conventions follow:

        bc2003_{res}_{model_key}_{metallicity}_{imf_key}_ssp.fits

    - Supported metallicities (Padova1994 isochrones):

        - 'm22': Z = 0.0001
        - 'm32': Z = 0.0004
        - 'm42': Z = 0.004
        - 'm52': Z = 0.008
        - 'm62': Z = 0.02
        - 'm72': Z = 0.05
        - 'm82': Z = 0.1

    - MILES input is selected via `model='MILES'` (resolved as `xmiless` internally).

    See also
    --------
    :class:`SSPBase` : Base class for all SSP models.

    References
    ----------
    Bruzual, G., & Charlot, S. (2003).
    Stellar population synthesis at the resolution of 2003.
    `MNRAS, 344(4), 1000–1028 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.344.1000B/abstract>`_.

    `Gustavo Bruzual page <https://www.bruzual.org/>`_

    Gutkin, J., Charlot, S., & Bruzual, G. (2016).
    Modelling the nebular emission from primeval to present-day star-forming galaxies
    `MNRAS, 462(2), 1757–1774 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G>`_.
    """

    _metallicity_map = {
     # Padova1994
    'm22': 0.0001,
    'm32': 0.0004,
    'm42': 0.004,
    'm52': 0.008,
    'm62': 0.02,
    'm72': 0.05,
    'm82': 0.1}
    _resolution = {'BaSeL': 'lr', 'stelib': 'hr', 'xmiless': 'hr'}
    
    def __init__(self, model='BaSeL', imf='Kroupa', path=None, verbose=True) -> None:
        self.model, model_key = self._parse_model(model)
        self.imf, imf_key = self._parse_imf(imf)

        if path is None:
            self.path = os.path.join(self.default_path, 'BC03', 'bc03_2016ver',
                                     self.model, self.imf)
        else:
            self.path = path

        if verbose:
            print(f"> Initialising BC03 model {self.model} (IMF={self.imf})")
        self.metallicities = np.array(list(self._metallicity_map.values())) * units.dimensionless_unscaled
        self.ages = np.loadtxt(
            os.path.join(self.default_path, 'BC03', 'TIME_SCALE.DAT')) * units.yr
        
        self.log_ages_yr = np.log10(self.ages / units.yr)

        load_wavelength = False
        for i, metallicity_key in enumerate(self._metallicity_map.keys()):
            fits_path = os.path.join(
                self.path, f"bc2003_{self._resolution[model_key]}_{model_key}_{metallicity_key}_{imf_key}_ssp.fits")
            table = Table.read(fits_path)
            if not load_wavelength:
                self.wavelength = table['wavelength'].value * u.angstrom
                self.L_lambda = np.zeros((self.metallicities.size, self.ages.size,
                                          self.wavelength.size))  * u.Lsun / u.Angstrom / u.Msun
                load_wavelength = True
            table.remove_column("wavelength")
            for j, column in enumerate(table.itercols()):
                self.L_lambda[i, j] = column.value * self.L_lambda.unit


    def _parse_model(self, model):
        if 'basel' in model.lower():
            model = 'BaSeL3.1_Atlas'
            key = 'BaSeL'
        elif 'stelib' in model.lower():
            model = 'Stelib_Atlas'
            key = 'stelib'
        elif 'miles' in model.lower():
            model = 'Miles_Atlas'
            key = 'xmiless'
        else:
            raise NameError(f"Unrecognized model: {model}.\n"
                             + "Select Basel, Stelib, Miles")
        return model, key

    def _parse_imf(self, imf):
        if 'cha' in imf.lower():
            imf = 'Chabrier_IMF'
            key = 'chab'
        elif 'sal' in imf.lower():
            imf = 'Salpeter_IMF'
            key = 'salp'
        elif 'kro' in imf.lower():
            imf = 'Kroupa_IMF'
            key = 'kroup'
        else:
            raise NameError(f"Unrecognized IMF: {imf}.\n"
                             + "Select Chabrier, Salpeter or Kroupa")
        return imf, key

class BaseGM(SSPBase):
    """
    SSP model class for the GM (Granada-MILES) base used in the CALIFA survey.

    This class loads a combined grid of simple stellar population (SSP) models:

    - The **young stellar populations** (ages < 63 Myr) are taken from GRANADA models developed by González Delgado et al. (2005).
    - The **older populations** (ages ≥ 63 Myr) are based on the MILES spectral
      library, specifically using the models from Vazdekis et al. (2010).

    Parameters
    ----------
    path : str or None, optional
        Path to the `gsd01_156.fits` file containing the GRANADA SSP model cube.
        If None, a default path is used. The companion `fits_like_properties.dat`
        file containing the age and metallicity grid must reside in the same directory.
    verbose : bool, optional
        If True, print initialization messages. Default is True.

    Notes
    -----
    - The Granada models are provided as 156 SEDs, reshaped to a (39 × 4) grid
      of ages and metallicities.
    - All models assume a **Salpeter (1955) initial mass function (IMF)**.
    - Spectral range: **3000 Å to 7000 Å**, with **0.3 Å resolution**.

    Example
    -------
    >>> from pst.SSP import BaseGM
    >>> gm = BaseGM(verbose=False)
    >>> print(gm.wavelength[0], gm.wavelength[-1])
    3000.0 Angstrom 7000.0 Angstrom
    >>> print(gm.ages.min(), gm.ages.max())
    1e6 yr 1.3e10 yr

    References
    ----------
    González Delgado, R. M., et al. (2005).
    "Evolutionary stellar population synthesis at high spectral resolution: optical wavelengths ." MNRAS, 357, 945.
    https://ui.adsabs.harvard.edu/abs/2005MNRAS.357..945G

    Sánchez, S. F., et al. (2012).
    "CALIFA, the Calar Alto Legacy Integral Field Area survey: I. Survey presentation."
    A&A, 538, A8. https://ui.adsabs.harvard.edu/abs/2012A%26A...538A...8S
    """
    def __init__(self, path=None, verbose=True):
        if path is None:
            self.path = os.path.join(self.default_path, 'BaseGM',
                                     'gsd01_156.fits')
            self._ssp_properties_path = os.path.join(
            self.default_path,
            'BaseGM', 'fits_like_properties.dat')
        else:
            self.path = path
            self._ssp_properties_path = os.path.join(os.path.dirname(path),
                                                    "fits_like_properties.dat")

        self.metallicities = np.loadtxt(self._ssp_properties_path, usecols=(1)
                                        ) * units.dimensionless_unscaled
        self.ages = np.loadtxt(self._ssp_properties_path, usecols=(0)) * units.yr
        self.log_ages_yr = np.log10(self.ages / units.yr)

        if verbose:
            print("> Initialising GRANADA models (IMF=Salpeter)")
        ssp_fits = fits.open(self.path)
        wl0 = ssp_fits[0].header['CRVAL1']
        deltawl = ssp_fits[0].header['CDELT1']

        self._header = ssp_fits[0].header
        self.norm = np.array(
            [ssp_fits[0].header['NORM'+str(i)] for i in range(156)]
            ) * units.Lsun / units.Msun / units.Angstrom

        SED = ssp_fits[0].data * self.norm[:, np.newaxis]
        SED = SED.reshape(39, 4, -1)
        self.norm = self.norm.reshape(39, 4)
        self.age_sort = np.argsort(self.ages.reshape(39, 4), axis=0)
        SED = SED[self.age_sort[:, 0], :, :]
        self.norm = self.norm[self.age_sort[:, 0], :]
        SED = SED.transpose(1, 0, 2)
        self.norm = self.norm.transpose(1, 0)
        self.wavelength = np.arange(wl0, wl0 + deltawl * SED.shape[-1],
                                    deltawl)  * units.Angstrom # AA
        self.L_lambda = SED
        ssp_fits.close()

        self.ages = np.unique(self.ages)
        self.log_ages_yr = np.log10(self.ages.value) * units.dimensionless_unscaled
        self.metallicities = np.sort(np.unique(self.metallicities))


# class FSPS(SSPBase):
#     """Fast Stellar Population Synthesis models..."""

#     def __init__(self):
#         print("> Initialising FSPS models")
#         self.path = os.path.join(config.path_to_ssp_models, 'FSPS',
#                                  'fsps_ssp_models.hdf5')
#         file = h5py.File(self.path, 'r')
#         elements = list(file.keys())
#         log_ages = []
#         metallicities = []
#         for elem_i in elements:
#             if elem_i.find('log_age') >= 0:
#                 init = 'log_age_'
#                 end = '_Z_'
#                 log_ages.append(elem_i[elem_i.find(init)+len(init):
#                                        elem_i.find(end)])
#                 metallicities.append(elem_i[elem_i.find(end)+len(end):])
#         metallicities = np.sort(np.unique(metallicities))
#         log_ages = np.unique(log_ages)
#         self.wavelength = file['wavelength'][()]
#         self.metallicities = np.array(metallicities, dtype=float)
#         self.log_ages_yr = np.sort(np.array(log_ages, dtype=float))
#         self.ages = 10**self.log_ages_yr

#         self.L_lambda = self.L_lambda = np.empty(
#             shape=(self.metallicities.size, self.log_ages_yr.size,
#                    self.wavelength.size), dtype=np.float32)
#         for i, met in enumerate(metallicities):
#             for j, log_age in enumerate(log_ages):
#                 name = 'log_age_{}_Z_{}'.format(log_age, met)
#                 l_lambda = file[name]['L_lambda'][()]
#                 self.L_lambda[i][j] = l_lambda
#         file.close()
#          = 'Lsun/Angstrom/Msun'


class XSL(SSPBase):
    """
    X-Shooter SSP empirical models.

    These models provide spectral energy distributions for simple stellar populations
    observed with the `X-Shooter instrument <http://www.eso.org/sci/facilities/paranal/instruments/xshooter/overview.html>`_. They cover a wide range of ages and
    metallicities, using two sets of isochrones and two IMFs.

    Parameters
    ----------
    IMF : str
        Initial Mass Function to use. Options:

        - 'Kroupa' : Kroupa (2001) IMF
        - 'Salpeter' : Salpeter (1955) IMF
    ISO : str
        Set of isochrones to use. Options:

        - 'P00' : Padova 2000 isochrones
        - 'PC' : PARSEC-COLIBRI isochrones
    path : str or None, optional
        Path to directory containing the SSP FITS files.
        If None, defaults to `default_path/XSL/IMF`.
    verbose : bool, optional
        If True, prints initialization messages. Default is True.

    Notes
    -----
    - Model files must be available in the specified `path`.

    References
    ----------
    Verro, K., Trager, S.C., Peletier, R.F., et al. 2022, `A&A, 661, A50 <https://ui.adsabs.harvard.edu/abs/2022A&A...661A..50V>`_.

    Project home page: http://xsl.u-strasbg.fr/index.html

    See also
    --------
    SSPBase
        Base class for SSP models.

    Example
    -------
    >>> from pst.SSP import XSL
    >>> xsl = XSL(IMF='Kroupa', ISO='P00', verbose=False)
    >>> print(xsl.wavelength[0], xsl.wavelength[-1])
    3000.0 Angstrom 25000.0 Angstrom
    >>> print(xsl.ages.min(), xsl.ages.max())
    5e7 yr 1.6e10 yr
    >>> print(xsl.metallicities)
    [0.0004 0.001 0.004 0.008 0.019 0.03]
    >>> print(xsl.L_lambda.shape)
    (6, 26, 58642)
    """
    _c_imf = dict(salpeter=9799552.50, kroupa=5567946.09)

    _initial_mass_functions = ["Kroupa", "Salpeter"]
    _iso_ages = {
            "PC": 10**np.arange(7.7, 10.1, 0.1) << u.yr,
            "P00": np.array([8.91e8, 1e9, 1.12e9, 1.26e9, 1.41e9, 1.58e9,
                             1.78e9, 2e9, 2.24e9, 2.51e9, 2.82e9, 3.16e9, 3.55e9,
                             3.98e9, 4.47e9, 5.01e9, 5.62e9, 6.31e9,
                             7.08e9, 7.94e9, 8.91e9, 1e10, 1.12e10, 1.26e10,
                             1.41e10, 1.59e10, 1.78e10]) << u.yr
            }
    _iso_metals = {
            "P00": np.array([0.0004, 0.001, 0.004, 0.008, 0.019, 0.03]
                           ) << u.dimensionless_unscaled,
            "PC": 10**np.array([-2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8,
                             -0.6, -0.4, -0.2, 0, 0.2]) * 0.019 << u.dimensionless_unscaled
            }
    _n_wavelength = 58642

    def __init__(self, IMF, ISO, path=None, verbose=True):
        if verbose:
            print("> Initialising X-Shooter (XSL) models (IMF={}, ISO={})".format(
                IMF, ISO))
        if (IMF != 'Kroupa') & (IMF != 'Salpeter'):
            raise NameError('IMF not valid (use Kroupa | Salpeter)')
        if (ISO != 'P00') & (ISO != 'PC'):
            raise NameError('ISO not valid (use P00 for Padova2000 or PC for PARSEC/COLIBRI)')
        if path:
            self.path = path 
        else:
            self.path = os.path.join(self.default_path, 'XSL', IMF)

        self._get_ssps(ISO)
        self._fetch_files(ISO, IMF)
 
    def _get_ssps(self, iso):
        self.ages = self._iso_ages[iso]
        self.log_ages_yr = np.log10(self.ages.to_value("yr"))
        self.metallicities = self._iso_metals[iso]
        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.ages.size,
                   self._n_wavelength), dtype=np.float32
                   ) * units.Lsun / units.Msun / units.Angstrom
 
    def _fetch_files(self, iso, imf):
        c_solar = self._c_imf[imf.lower()]  # Convert to solar units
        if iso == "P00":
            file_fmt = r"XSL_SSP_T{:2.2e}_Z{}_Kroupa_P00.fits"
            for age_idx, age in enumerate(self.ages):
                for met_idx, met in enumerate(self.metallicities):
                    filename = file_fmt.format(age.to_value("yr"), met.value)
                    path_to_file = os.path.join(self.path, filename)
                    print(path_to_file)
                    if not os.path.isfile(path_to_file):
                        raise FileNotFoundError(f"{path_to_file} not found")
                    with fits.open(path_to_file) as hdul:
                        spec = hdul[0].data * c_solar
                        self.L_lambda[met_idx][age_idx] = spec * self._L_lambda.unit
        elif iso == "PC":
            file_fmt = r"XSL_SSP_logT{:.1f}_MH{:.1f}_Kroupa_PC.fits"
            for age_idx, age in enumerate(self.ages):
                for met_idx, met in enumerate(self.metallicities):
                    filename = file_fmt.format(np.log10(age.to_value("yr")),
                                               np.log10(met.value / 0.019))
                    if met == 0.019:
                        filename = file_fmt.format(np.log10(age.to_value("yr")),
                                               np.log10(met.value / 0.019 - 1e-3))
                    path_to_file = os.path.join(self.path, filename)
                    print(path_to_file)
                    if not os.path.isfile(path_to_file):
                        raise FileNotFoundError(f"{path_to_file} not found")
                    with fits.open(path_to_file) as hdul:
                        spec = hdul[0].data * c_solar
                        self.L_lambda[met_idx][age_idx] = spec * self._L_lambda.unit

        # Use the last file to load the wavelenth array
        with fits.open(path_to_file) as hdul:
            self.wavelength = 10**(
                (np.arange(0, hdul[0].data.size, 1) - hdul[0].header['CRPIX1'])
                * hdul[0].header['CDELT1'] + hdul[0].header['CRVAL1'] + 1
                ) * units.Angstrom

class EMILES(SSPBase):
    """
    E-MILES simple stellar population (SSP) models.

    These models provide high-quality empirical SSP spectra covering the
    spectral range 1680−50000 Å at moderately high resolution. The E-MILES
    library combines multiple stellar libraries and theoretical isochrones
    to deliver spectra of single-age, single-metallicity populations.

    The UV spectral range (1680–3540 Å) is computed using the `NGSL space-based
    stellar library <https://archive.stsci.edu/prepds/stisngsl/>`_, offering a significant improvement over earlier space-based
    models. The optical spectra use the MILES empirical library, and redder
    wavelengths are covered by `Indo-US <https://noirlab.edu/science/observing-noirlab/observing-kitt-peak/telescope-and-instrument-documentation/cflib>`_, `CaT <https://research.iac.es/proyecto/miles/pages/stellar-libraries/cat-library.php>`_, and `IRTF <https://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_ empirical libraries,
    all computed with consistent methods.

    The SSP spectra span metallicities from −1.79 < [M/H] < +0.26 and ages
    greater than 30 Myr, across several IMF types with varying slopes.

    Spectral resolution:

        - UV range (λ < 3060 Å): FWHM ≈ 3 Å
        - UV range (3060 Å < λ < 3540 Å): FWHM ≈ 5 Å
        - Optical range (3540 Å to 8950 Å): FWHM ≈ 2.5 Å
        - Infrared (longer wavelengths): σ = 60 km/s

    Parameters
    ----------
    iso : str
        Isochrone set to use. Options:

        - 'BASTI' : BASTI isochrones (Pietrinferni et al. 2004)
        - 'PADOVA00' : Padova 2000 isochrones (Girardi et al. 2000)
    imf : str
        Initial mass function to use. Options:

        - 'KROUPA_UNIVERSAL' : Kroupa universal IMF (Kroupa 2001)
        - 'UNIMODAL' : Unimodal IMF (single power-law, Vazdekis et al. 1996)
        - 'BIMODAL' : Bimodal IMF (Vazdekis et al. 1996)
        - 'CHABRIER' : Chabrier IMF (Chabrier 2003)
    path : str or None, optional
        Path to the directory containing the E-MILES FITS files.
        Defaults to `default_path/EMILES`.
    verbose : bool, optional
        If True, prints initialization messages. Default is True.

    Notes
    -----
    - The spectral resolution and wavelength coverage follow Vazdekis et al. (2016)
      and Röck et al. (2016).
    - The models cover a wavelength range of approximately 1680–50000 Å with
      spectral resolution ~2.5 Å (FWHM).
    - The metallicities are given as logarithmic values relative to solar,
      with solar metallicity fixed at Z=0.019.

    References
    ----------
    Vazdekis, A., Koleva, M., Ricciardelli, E., et al. 2016, `MNRAS, 463, 3409 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.3409V/abstract>`_
    Röck, B., Vazdekis, A., Ricciardelli, E., et al. 2016, `A&A, 589, A73, 8 <https://ui.adsabs.harvard.edu/abs/2016A%26A...589A..73R/abstract>`_

    See also
    --------
    SSPBase
        Base class for simple stellar population models.

    Example
    -------
    >>> from pst.SSP import EMILES
    >>> emiles = EMILES(iso='BASTI', imf='KROUPA_UNIVERSAL', verbose=False)
    >>> print(emiles.wavelength[0], emiles.wavelength[-1])
    1680.0 Angstrom 50000.0 Angstrom
    >>> print(emiles.ages.min(), emiles.ages.max())
    30.0 Myr 14.0 Gyr
    >>> print(emiles.metallicities)
    [0.0001 0.00016 ... 0.03]  # actual values depend on the isochrone
    >>> print(emiles.L_lambda.shape)
    (12, 52, 53689)
    """
    _lib_isochrones = {"BASTI": "iTp",
                  "PADOVA00": "iPp"}
    _lib_imfs = {"KROUPA_UNIVERSAL": "ku1.30",
                "UNIMODAL": "un",
                "BIMODAL": "bi",
                "CHABRIER": "ch1.30"}

    
    _n_wavelength = 53689  # Spectra size

    _iso_logmetals = {
        "BASTI": np.array([-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35,
                           -0.25, 0.06, 0.15, 0.26, 0.4]),
        "PADOVA00": np.array([-2.32, -1.71 , -1.31 , -0.71, -0.40, 0.00, 0.22])
    }
    _iso_ages = {
        "BASTI": np.array(
            [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.250,
            1.500, 1.750, 2.000, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75, 4.0,
            4.5, 5.0, 5.5, 6.0, 6.5, 7., 7.5, 8., 8.5, 9.0, 9.5, 10., 10.5, 11.,
            11.5, 12., 12.5, 13., 13.5, 14.]) << u.Gyr,
        "PADOVA00": np.array([0.063, 0.071, 0.079, 0.089, 0.10, 0.11, 0.13,
                              0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32,
                              0.35, 0.40, 0.45, 0.50, 0.56, 0.63, 0.71, 0.79,
                              0.89, 1.00, 1.12, 1.26, 1.41, 1.58, 1.78, 2.00,
                              2.24, 2.51, 2.82, 3.16, 3.55, 3.98, 4.47, 5.01,
                              5.62, 6.31, 7.08, 7.94, 8.91, 10.00, 11.22, 12.59,
                              14.13, 15.85, 17.78]) << u.Gyr
        }

    def __init__(self, iso, imf, path=None, verbose=True):
        if verbose:
            print("> Initialising E-MILES models (IMF={}, ISO={})".format(
                imf, iso))
        if path:
            self.path = path
        else:
            self.path = os.path.join(self.default_path, 'EMILES')
        model_name = self._get_models_prefix(iso, imf)
        self._load_models(model_name)

    def _get_models_prefix(self, iso, imf):
        """Get the file prefix used to load the model files."""
        model_name = os.path.join(self.path, "E")
        # IMF
        if not imf.upper() in self._lib_imfs.keys():
            raise NameError(f"Input IMF {imf} not recognized"
                            + f"\nThe available isochrones are: {self._lib_imfs.keys()}")
        else:
            model_name += self._lib_imfs[imf.upper()]
        # Metallicity and age
        model_name += r"Z{}{:.2f}T{:07.4f}_"

        # Isochrone
        if not iso.upper() in self._lib_isochrones:
            raise NameError(f"Input isochrone {iso} not recognized"
                            + f"\nThe available isochrones are: {list(self._lib_isochrones.keys())}")
        else:
            model_name += self._lib_isochrones[iso.upper()]
            self._ages = self._iso_ages[iso.upper()]
            self.log_ages_yr = np.log10(self._ages / units.yr)
            # Solar metallicity defined in V+16.
            self._logmetals = self._iso_logmetals[iso.upper()]
            self._metallicities = 10**self._logmetals * 0.019
        # Alpha over iron
        model_name += "0.00_baseFe.fits"
        return model_name

    def _load_models(self, model_name):
        """Load the SSP models from individual FITS files."""
        self.L_lambda = np.zeros(
            (self.metallicities.size, self.ages.size,
             self._n_wavelength))  * u.Lsun / u.Angstrom / u.Msun

        for met_idx, metal in enumerate(self._logmetals):
            if metal < 0:
                m_prefix = "m"
            else:
                m_prefix = "p"

            for age_idx, age in enumerate(self.ages.to_value("Gyr")):
                file = model_name.format(m_prefix, np.abs(metal), age)
                assert os.path.isfile(file), f"File {file} not found"
                with fits.open(file) as hdul:
                    self.L_lambda[met_idx, age_idx] = hdul[0].data << self.L_lambda.unit
        
        with fits.open(file) as hdul:
            wcs = WCS(hdul[0].header)
            self.wavelength = wcs.array_index_to_world_values(
                np.arange(0, self._n_wavelength)) << u.AA

if __name__ == '__main__':
    # ssp = PopStar(IMF='cha_0.15_100')
    from matplotlib import pyplot as plt
    ssp = BaseGM()

# %%                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/

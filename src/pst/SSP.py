import os
from copy import deepcopy
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from astropy import units

from scipy import interpolate

class SSPBase(object):
    """Base class that represents a model of Simple Stellar Populations.
    
    Description
    -----------
    This class is meant for representing a model of Simple Stellar Populations
    that is mainly composed by a discreted grid of ages and metallicities and
    their associated spectral energy distributions.

    Attributes
    ----------
    - ages: astropy.units.Quantity
        Ages of the SSPs.
    - metallicities: astropy.units.Quantity
        Metallicities of the SSPs.
    - L_lambda: astropy.units.Quantity
        Spectral energy distribution of each SSP. Each dimension correspond to
        (metallicity, ages, wavelength).
    - wavelength: astroy.units.Quantity
        Wavelength array associated to the SED of the SSPs.
    
    Methods
    -------
    #TODO
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
            raise NameError("Ages must be an astropy.Quantity")
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

    def compute_SED(self, time, mass, metallicity,
                    t_obs=13.7 * u.Gyr):
        """
        Compute the SED for a given star formation history.
        
        Prameters:
        ----------
        - time: (np.ndarray)
            Time steps expressed in Gyr corresponding to the age of the Universe.
        - mass: (np.ndarray)
                Cumulative stellar mass for each time step.
        - metallicity: (np.ndarray)
            Average metallicity on each time step.
        """
        # SSP time steps to interpolate (i.e. lookback time)
        age_bins = np.hstack(
            [0 * u.yr, np.sqrt(self.ages[1:] * self.ages[:-1]), 1e12 * u.yr])
        t_i = t_obs - age_bins
        t_i = t_i[t_i > 0]
        mass_interpolator = interpolate.Akima1DInterpolator(
            time.value, mass.value)
        mass_history = mass_interpolator(t_i.to(time.unit)) * mass.unit
        mass_history[t_i > time[-1]] = mass[-1]
        mass_history[t_i < time[0]] = 0
        mass_bins = np.hstack([mass_history[:-1] - mass_history[1:],
                                 mass_history[-1]])

        mass_z_interpolator = interpolate.Akima1DInterpolator(
            time.value, mass.value * metallicity.value)
        mass_z_history = mass_z_interpolator(
            t_i.to(time.unit)) * metallicity.unit * mass.unit
        mass_z_history[t_i > time[-1]] = metallicity[-1] * mass[-1]
        mass_z_history[t_i < time[-1]] = 0
        mass_z_bins = np.hstack([mass_z_history[:-1] - mass_z_history[1:],
                                 mass_z_history[-1]])
        z_history = np.clip(mass_z_bins / mass_bins,
                            self.metallicities[0], self.metallicities[-1])

        sed = np.zeros(self.wavelength.size)
        weights = np.zeros((self.metallicities.size, t_i.size)) * mass.unit
        
        z_indices = np.searchsorted(
            self.metallicities, z_history).clip(
            min=1, max=self.metallicities.size - 1)
        t_indices = np.arange(0, weights.shape[1], dtype=int)

        weight_Z = np.log(
                    z_history / self.metallicities[z_indices - 1]) / np.log(
                    self.metallicities[z_indices] / self.metallicities[z_indices-1]
                    ) * mass.unit
        weights[z_indices, t_indices] = weight_Z
        weights[z_indices - 1, t_indices] = 1 * mass.unit - weight_Z
        weights *=  mass_bins[np.newaxis, :]
        mask = (weights > 0) & np.isfinite(weights)
        sed = np.nansum(
            weights[mask, np.newaxis] * self.L_lambda[mask, :], axis=(0))

        return sed, weights

    def compute_burstSED(self, age, Z):
        """Compute the SED of a stellar burst of age t and metallicity Z."""
        log_age = np.log10(age)
        index_Z_hi = self.metallicities.searchsorted(Z).clip(
            1, len(self.metallicities)-1)
        weight_Z_hi = (np.log(Z/self.metallicities[index_Z_hi-1])
                       / np.log(self.metallicities[index_Z_hi]
                                / self.metallicities[index_Z_hi-1]))
        index_tage_hi = self.log_ages_yr.searchsorted(log_age).clip(
            1, len(self.log_ages_yr)-1)
        weight_tage_hi = (log_age - self.log_ages_yr[index_tage_hi-1])/(
            self.log_ages_yr[index_tage_hi]
            - self.log_ages_yr[index_tage_hi-1])
        sed = (
            self.L_lambda[index_Z_hi][index_tage_hi]
            * weight_Z_hi * weight_tage_hi
            + self.L_lambda[index_Z_hi][index_tage_hi - 1]
            * weight_Z_hi * (1 - weight_tage_hi)
            + self.L_lambda[index_Z_hi-1][index_tage_hi]
            * (1 - weight_Z_hi) * weight_tage_hi
            + self.L_lambda[index_Z_hi-1][index_tage_hi-1]
            * (1 - weight_Z_hi) * (1 - weight_tage_hi)
            )
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

    def regrid(self, n_logage_bin_edges, n_logmet_bin_edges):
        """Reinterpolate the SSP model to a new grid of input ages and metallicities."""
        print("[SSP] Interpolating the SSP model to a new grid of ages and metallicities")
        logmet_bin_edges, logage_bin_edges = self.get_ssp_logedges()
        
        ssp_age_idx = np.searchsorted(logage_bin_edges, n_logage_bin_edges)
        age_bins = [slice(ssp_age_idx[i], ssp_age_idx[i+1]) for i in range(len(n_logage_bin_edges) - 1)]
        ssp_met_idx = np.searchsorted(logmet_bin_edges, n_logmet_bin_edges)
        met_bins = [slice(ssp_met_idx[i], ssp_met_idx[i+1]) for i in range(len(n_logmet_bin_edges) - 1)]

        # Bin the SED of the SSPs
        previous_sed = self.L_lambda.copy()
        self.L_lambda = np.empty(
            (len(met_bins), len(age_bins), previous_sed.shape[-1]),
            dtype=float) * previous_sed.unit

        print("[SSP] New SSP grid dimensions: ", self.L_lambda.shape)
        for j, m_bin in enumerate(met_bins):
            met_av_sed = np.exp(np.mean(
                np.log(previous_sed[m_bin] / previous_sed.unit), axis=0))
            for i, a_bin in enumerate(age_bins):
                self.L_lambda[j, i, :] = np.exp(
                    np.mean(np.log(met_av_sed[a_bin]), axis=0)) * previous_sed.unit

        self.metallicities = 10**((n_logmet_bin_edges[:-1] + n_logmet_bin_edges[1:]) / 2
                                  ) * units.dimensionless_unscaled
        log_ages_yr = (n_logage_bin_edges[:-1] + n_logage_bin_edges[1:]) / 2
        self.ages = 10**log_ages_yr * u.yr

    def cut_wavelength(self, wl_min=None, wl_max=None):
        """Cut model wavelength edges."""
        if wl_min is None:
            wl_min = self.wavelength[0]
        elif not isinstance(wl_min, u.Quantity):
            print("Assuming minimum wavelenth in the same units as wavelength")
            wl_min *= self.wavelength.unit
        if wl_max is None:
            wl_max = self.wavelength[0]
        elif not isinstance(wl_max, u.Quantity):
            print("Assuming minimum wavelenth in the same units as wavelength")
            wl_max *= self.wavelength.unit

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
            print('[SSP] Models cut between {} {}'.format(wl_min, wl_max))

    def interpolate_sed(self, new_wl_edges):
        """Flux-conserving interpolation.

        params
        -----
        - new_wl_edges: bin edges of the new interpolated points.
        """
        if not isinstance(new_wl_edges, units.Quantity):
            new_wl_edges *= self.wavelength.unit

        new_wl = (new_wl_edges[1:] + new_wl_edges[:-1]) / 2
        dwl = np.diff(new_wl_edges)
        ori_dwl = np.hstack((np.diff(self.wavelength),
                             self.wavelength[-1] - self.wavelength[-2]))
        print('[SSP] Interpolating SSP SEDs')
        new_l_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
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
    
    def compute_photometry(self, filter_list, z_obs=0.0):
        """Compute the SSP synthetic photometry of a set of filters.
        
        Paramteres
        ----------
        filter_list: list of pst.observable.Filter
            A list of photometric filters.

        Returns
        -------
        photometry: np.ndarray
            Array storing the photometry. The dimensions correspond to
            filter, metallicity and age.
        """
        print("Computing synthetic photometry for SSP model")
        self.photometry = np.zeros((len(filter_list),
                                    *self.L_lambda.shape[:-1]))
        self.photometry_filters = filter_list
        for ith, f in enumerate(filter_list):
            f.interpolate(self.wavelength * (1 + z_obs))
            self.photometry[ith], _ = f.get_fnu(
                    self.L_lambda  * u.Msun / 4 / np.pi / (10 * u.pc)**2,
                    mask_nan=False)

    def copy(self):
        """Return a copy of the SSP model."""
        return deepcopy(self)

class PopStar(SSPBase):
    """PopStar SSP models (Mollá+09)."""

    def __init__(self, IMF, nebular=False, path=None):
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

        # isochrone age in delta [log(tau)]=0.01
        self.wavelength = np.loadtxt(os.path.join(
            self.path, IMF, 'SED', f'spneb_{IMF}_0.15_100_z0500_t9.95'), dtype=float,
            skiprows=0, usecols=(0,), unpack=True) * units.Angstrom

        print("> Initialising Popstar models (IMF='"+IMF+"')")
        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32
                   ) * units.Lsun / units.Angstrom / units.Msun
        if nebular:
            column = 3
            print('--> Including NEBULAR emission')
        else:
            column = 1
            print('--> Only stellar continuum')
        for i, Z in enumerate(self.metallicities.value):
            for j, age in enumerate(self.log_ages_yr.value):
                file = os.path.join(
                    self.path, IMF, 'SED',
                    'spneb_{0}_0.15_100_z{1:04.0f}_t{2:.2f}'.format(IMF, Z*1e4, age))
                spec = np.loadtxt(
                    file, dtype=float, skiprows=0, usecols=(column),
                    unpack=True)  # Lsun/Angstrom/Msun
                self.L_lambda[i][j] = spec * self.L_lambda.unit


class PyPopStar(SSPBase):
    """PyPopStar SSP models (Millán-Irigoyen+21)."""

    def __init__(self, IMF, nebular=False, path=None):
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
            print("> Initialising Popstar models (neb em) (IMF='"
                  + IMF + "')")
            column = 'flux_total'
        else:
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

# class BC03_Padova94(SSP): # TODO: CHANGE METHODS

#     def __init__(self, mode, IMF):
#         self.path = os.path.join(os.path.dirname(__file__),
#                                  'data/BC03/models/Padova1994')
#         self.metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02,
#                                        0.05])
#         self.ages = np.loadtxt(os.path.join(self.path, 'ages.dat'))
#         self.log_ages_yr = np.log10(self.ages)

#         self.L_lambda = np.empty(shape=(self.metallicities.size,
#                                         self.log_ages_yr.size),
#                                  dtype=Spectrum1D)
#         self.wavelength = np.loadtxt(os.path.join(self.path,
#                                      'wavelength_'+mode+'.dat')) * u.Angstrom
#         print("> Initialising Bruzual&Charlote2003 (P94) models (IMF='"
#               + IMF + "')")
#         self.L_lambda = np.empty(shape=(self.metallicities.size,
#                                         self.log_ages_yr.size,
#                                         self.wavelength.size),
#                                  dtype=np.ndarray)
#         for i, Z in enumerate(self.metallicities):
#             file = os.path.join(self.path, IMF, 'SED',
#                                 'bc03_{0}_Z_{1:.4f}_{2}.txt'.format(mode, Z,
#                                                                     IMF))
#             spec = np.loadtxt(file, dtype=float,
#                               usecols=(1),
#                               unpack=True
#                               ) * u.Lsun/u.Angstrom/u.Msun
#             Spectrum1D(flux=spec, spectral_axis=self.wavelength)
                
#             self.L_lambda[i][:, :] = np.loadtxt(file, dtype=float
#                                                      ) * u.Lsun/u.Angstrom/u.Msun

class BaseGM(SSPBase):
    """Granada models..."""

    def __init__(self, path=None):
        if path is None:
            self.path = os.path.join(self.default_path, 'BaseGM',
                                     'gsd01_156.fits')
            self.ssp_properties_path = os.path.join(
            self.default_path,
            'BaseGM', 'fits_like_properties.dat')
        else:
            self.path = path
            self.ssp_properties_path = os.path.join(os.path.dirname(path),
                                                    "fits_like_properties.dat")

        self.metallicities = np.loadtxt(self.ssp_properties_path, usecols=(1)
                                        ) * units.dimensionless_unscaled
        self.ages = np.loadtxt(self.ssp_properties_path, usecols=(0)) * units.yr
        self.log_ages_yr = np.log10(self.ages / units.yr)

        print("> Initialising GRANADA models (IMF=Salpeter)")
        ssp_fits = fits.open(self.path)
        wl0 = ssp_fits[0].header['CRVAL1']
        deltawl = ssp_fits[0].header['CDELT1']

        self.header = ssp_fits[0].header
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

    Input:
        - IMF: Initial mass function [KRO, SAL]
        - ISOC: Set of isochrones to use [P00, PC]

    Please cite Verro et al. (2022b), when using these data.

    XSL SSP models are in cgs flux units F [erg cm s Å].
    To convert them to L/Lsun/Msun/Å, multiply the flux array by C_{imf}
    """

    C_imf = dict(salpeter=9799552.50, kroupa=5567946.09)

    def __init__(self, IMF, ISO, path=None):
        print("> Initialising X-Shooter (XSL) models (IMF={}, ISO={})".format(
            IMF, ISO))
        if (IMF != 'Kroupa') & (IMF != 'Salpeter'):
            raise NameError('IMF not valid (use Kroupa | Salpeter)')
        if (ISO != 'P00') & (ISO != 'PC'):
            raise NameError('ISO not valid (use P00 for Padova2000 or PC for PARSEC/COLIBRI)')
        if path:
            self.path = os.path.join(path, '_'.join([IMF, ISO]))
        else:
            self.path = os.path.join(self.default_path, 'XSL',
                                     '_'.join([IMF, ISO]))
        files = os.listdir(self.path)
        if len(files) == 0:
            raise NameError('No files found at:\n {}'.format(self.path))
        ages = []
        metallicities = []
        for file in files:
            ages.append(float(file[file.find('T') + 1: file.find('_Z')]))
            metallicities.append(
                float(file[file.find('Z') + 1: file.find('_' + IMF)]))
        self.ages = np.unique(np.array(ages)) * units.yr
        self.log_ages_yr = np.log10(self.ages / units.yr)
        self.metallicities = np.unique(np.array(metallicities)
                                       ) * units.dimensionless_unscaled

        header = 'XSL_SSP_T'
        c_solar = self.C_imf[IMF.lower()]  # Convert to solar units

        with fits.open(os.path.join(self.path,
                                     header+'{:.2e}_Z{:}_{}_{}.fits'.format(
                self.ages.value[0], self.metallicities.value[0], IMF, ISO))) as hdul:
            self.wavelength = 10**(
                (np.arange(0, hdul[0].data.size, 1) - hdul[0].header['CRPIX1'])
                * hdul[0].header['CDELT1'] + hdul[0].header['CRVAL1'] + 1
                ) * units.Angstrom

        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32
                   ) * units.Lsun / units.Msun / units.Angstrom

        for i, Z in enumerate(self.metallicities.value):
            for j, age in enumerate(self.ages.value):
                filename = header+'{:.2e}_Z{:}_{}_{}.fits'.format(age, Z, IMF,
                                                                  ISO)
                file = os.path.join(self.path, filename)
                with fits.open(file) as hdul:
                    spec = hdul[0].data * c_solar
                    hdul.close()
                self.L_lambda[i][j] = spec * self._L_lambda.unit


if __name__ == '__main__':
    # ssp = PopStar(IMF='cha_0.15_100')
    from matplotlib import pyplot as plt
    ssp = BaseGM()

# %%                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/

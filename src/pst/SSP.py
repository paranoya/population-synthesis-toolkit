import os
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy import constants as c

from pst.utils import gaussian1d_conv

class SSPBase(object):
    """Base class that represents a model of Simple Stellar Populations."""
    L_lambda = None
    wavelength = None
    metallicities = None
    ages = None
    log_ages_yr = None

    default_path = os.path.join(os.path.dirname(__file__), "data", "ssp")

    def compute_SED(self, time, mass, metallicity, dust_model=None,
                    plot_interpolation=False, plot_weights=False):
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
        - dust_mode: (callable, default=None)
            A callable function that takes as input and age and a wavelength array `f(age, wave)`
            and returns an array corresponding to the extinction.
        - plot_interpolation: (bool, default=False)
            If True, plot the interpolated mass history of the galaxy to the SSP age bins.
        - plot_weights: (bool, default=False)
            If True, plot the weights of each SSP in a grid of ages and metallicity.
        
        """
        # SSP time steps to interpolate (i.e. lookback time)
        t_i = self.log_ages_yr - np.ediff1d(self.log_ages_yr, to_begin=0)/2
        t_i = np.append(t_i, 12)  # 1000 Gyr
        # Conversion to time since the onset of SF (i.e. age of the Universe)
        today = time.max()
        t_i = today - 10**t_i
        t_i[0] = today
        t_i.clip(time.min(), today, out=t_i)
        interp_M_i = np.interp(t_i, time, mass)
        M_i = -np.ediff1d(interp_M_i)
        Z_i = np.interp(t_i, time, np.log10(metallicity))
        Z_i = 10**Z_i
        # Z_i = -np.ediff1d( Z_i ) / (M_i+u.kg)
        # to prevent extrapolation
        Z_i.clip(self.metallicities[0], self.metallicities[-1], out=Z_i)
        extinction = np.ones((M_i.size,
                              self.L_lambda[0][0].size))
        if dust_model:
            log_t_mid = (np.log10(t_i[:-1])+np.log10(t_i[1:]))/2
            for ii, log_t_i in enumerate(log_t_mid):
                extinction[ii, :] = dust_model(10**log_t_i/u.yr, self.wavelength)

        sed = np.zeros(self.wavelength.size)
        weights = np.zeros((t_i.size, self.metallicities.size))
        for i, mass_i in enumerate(M_i):
            if mass_i > 0:
                index_Z_hi = self.metallicities.searchsorted(Z_i[i]).clip(
                    1, len(self.metallicities)-1)
                # log interpolation in Z
                weight_Z_hi = np.log(
                    Z_i[i] / self.metallicities[index_Z_hi-1]) / np.log(
                        self.metallicities[index_Z_hi] / self.metallicities[index_Z_hi-1])
                sed += extinction[i, :] * mass_i * (
                    self.L_lambda[index_Z_hi][i] * weight_Z_hi +
                    self.L_lambda[index_Z_hi-1][i] * (1-weight_Z_hi))
                weights[i, index_Z_hi] = weight_Z_hi * mass_i
                weights[i, index_Z_hi-1] = (1 - weight_Z_hi) * mass_i
        weights /= np.sum(M_i)

        if plot_interpolation:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.plot(np.log10(time/u.Gyr), np.log10(mass), 'k-o', label='input')
            ax.plot(np.log10(t_i/u.Gyr), np.log10(interp_M_i), 'r-o', label='input')
            ax.set_ylabel(r'$\log_{10}(M_*)$')
            ax.set_xlabel(r'$\log_{10}(t/Gyr)$')

            ax = fig.add_subplot(212)
            ax.plot(np.log10(time/u.Gyr), np.log10(metallicity/0.02),
                    'k-o', label='input')
            ax.plot(np.log10(t_i/u.Gyr), np.log10(Z_i/0.02), 'r-o', label='interp')
            ax.set_ylabel(r'$\log_{10}(Z_*/Z_\odot)$')
            ax.set_xlabel(r'$\log_{10}(t/Gyr)$')
            fig.subplots_adjust(hspace=0)
            plt.show()

        if plot_weights:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(5,5))
            plt.imshow(np.log10(weights.T), aspect='auto', origin='lower',
                       interpolation='none', cmap='rainbow', vmin=-4)
            nticks = np.arange(0, self.log_ages_yr.size, 15)
            plt.xticks(nticks, labels=np.round(self.log_ages_yr[nticks],
                                               decimals=2))
            plt.xlabel(r'$\log_{10}(age_{SSP}/yr)$')
            plt.grid(b=True)
            plt.colorbar()
            plt.show()    
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

    def regrid(self, n_logage_bin_edges, n_logmet_bin_edges):
        """Reinterpolate the SSP model to a new grid of input ages and metallicities."""
        print("[SSP] Interpolating the SSP model to a new grid of ages and metallicities")
        lim = 1.5 * self.log_ages_yr[[0, -1]] - 0.5 * self.log_ages_yr[[1, -2]]
        logage_bin_edges = np.hstack(
                [lim[0], (self.log_ages_yr[1:] + self.log_ages_yr[:-1])/2, lim[1]])

        log_met_bins = np.log10(self.metallicities)
        lim = 1.5 * log_met_bins[[0, -1]] - 0.5 * log_met_bins[[1, -2]]
        logmet_bin_edges = np.hstack(
                [lim[0], (log_met_bins[1:] + log_met_bins[:-1])/2, lim[1]])
        
        ssp_age_idx = np.searchsorted(logage_bin_edges, n_logage_bin_edges)
        age_bins = [slice(ssp_age_idx[i], ssp_age_idx[i+1]) for i in range(len(n_logage_bin_edges) - 1)]
        ssp_met_idx = np.searchsorted(logmet_bin_edges, n_logmet_bin_edges)
        met_bins = [slice(ssp_met_idx[i], ssp_met_idx[i+1]) for i in range(len(n_logmet_bin_edges) - 1)]

        # Bin the SED of the SSPs
        previous_sed = self.L_lambda.copy()
        self.L_lambda = np.empty((len(met_bins), len(age_bins), previous_sed.shape[-1]), dtype=float)
        print("[SSP] New SSP grid dimensions: ", self.L_lambda.shape)
        for j, m_bin in enumerate(met_bins):
            met_av_sed = np.exp(np.mean(np.log(previous_sed[m_bin]), axis=0))
            for i, a_bin in enumerate(age_bins):
                self.L_lambda[j, i, :] = np.exp(np.mean(np.log(met_av_sed[a_bin]), axis=0))
        self.metallicities = 10**((n_logmet_bin_edges[:-1] + n_logmet_bin_edges[1:]) / 2)
        self.log_ages_yr = (n_logage_bin_edges[:-1] + n_logage_bin_edges[1:]) / 2
        self.ages = 10**self.log_ages_yr

    def cut_wavelength(self, wl_min, wl_max):
        """Cut model wavelength edges."""
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
        new_wl = (new_wl_edges[1:] + new_wl_edges[:-1]) / 2
        dwl = np.diff(new_wl_edges)
        ori_dwl = np.hstack((np.diff(self.wavelength),
                             self.wavelength[-1] - self.wavelength[-2]))
        print('[SSP] Interpolating SSP SEDs')
        new_l_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   new_wl.size), dtype=np.float32)
        for i in range(self.L_lambda.shape[0]):
            for j in range(self.L_lambda.shape[1]):
                f = np.interp(new_wl_edges, self.wavelength,
                              np.cumsum(self.L_lambda[i, j] * ori_dwl))
                new_flux = np.diff(f) / dwl
                new_l_lambda[i, j] = new_flux
        self.L_lambda = new_l_lambda
        self.wavelength = new_wl

    def convolve_sed(self, profile=gaussian1d_conv,
                     **profile_params):
        """Convolve the SSP spectra with a given LSF."""
        print('[SSP] Convolving SSP SEDs')
        for i in range(self.L_lambda.shape[0]):
            for j in range(self.L_lambda.shape[1]):
                self.L_lambda[i, j] = profile(self.L_lambda[i, j],
                                              **profile_params)

    def get_mass_lum_ratio(self, wl_range):
        """Compute the mass-to-light ratio within a giveng wavelength range."""
        pts = np.where((self.wavelength >= wl_range[0]) &
                       (self.wavelength <= wl_range[1]))[0]
        mass_to_lum = np.empty((self.metallicities.size,
                                     self.ages.size)
                                    )
        for i in range(self.metallicities.size):
            for j in range(self.ages.size):
                mass_to_lum[i, j] = 1/np.mean(self.L_lambda[i, j][pts])
        return mass_to_lum


class PopStar(SSPBase):
    """PopStar SSP models (Mollá+09)."""

    def __init__(self, IMF, nebular=False, path=None):
        if path is None:
            self.path = os.path.join(self.default_path, 'PopStar')
        else:
            self.path = path
        self.metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02,
                                       0.05])
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
                                     10.15, 10.18])
        self.ages = 10**self.log_ages_yr
        # isochrone age in delta [log(tau)]=0.01
        self.wavelength = np.loadtxt(os.path.join(
            self.path, IMF, 'SED', f'spneb_{IMF}_0.15_100_z0500_t9.95'), dtype=float,
            skiprows=0, usecols=(0,), unpack=True)
        print("> Initialising Popstar models (IMF='"+IMF+"')")
        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32)
        if nebular:
            column = 3
            print('--> Including NEBULAR emission')
        else:
            column = 1
            print('--> Only stellar continuum')
        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.log_ages_yr):
                file = os.path.join(
                    self.path, IMF, 'SED',
                    'spneb_{0}_0.15_100_z{1:04.0f}_t{2:.2f}'.format(IMF, Z*1e4, age))
                spec = np.loadtxt(
                    file, dtype=float, skiprows=0, usecols=(column),
                    unpack=True)  # Lsun/Angstrom/Msun
                self.L_lambda[i][j] = spec
        self.sed_unit = 'Lsun/Angstrom/Msun'


class PyPopStar(SSPBase):
    """PyPopStar SSP models (Millán-Irigoyen+21)."""

    def __init__(self, IMF, nebular=False, path=None):
        if path is None:
            self.path = os.path.join(self.default_path, 'PyPopStar', IMF)
        else:
            self.path = path
        self.metallicities = np.array([0.004, 0.008, 0.02, 0.05])
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
       10.08, 10.11, 10.12, 10.13, 10.14, 10.15, 10.18])
        self.ages = 10**self.log_ages_yr
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
                self.metallicities[0], self.log_ages_yr[0]))
                       ) as hdul:
            self.wavelength = hdul[1].data['wavelength']  # Angstrom

        self.L_lambda = self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32)

        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.log_ages_yr):
                filename = header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(Z, age)
                file = os.path.join(self.path, filename)
                with fits.open(file) as hdul:
                    self.L_lambda[i][j] = hdul[1].data[column]  # Lsun/AA/Msun
                    hdul.close()
        # Avoid 0 flux
        self.L_lambda[self.L_lambda <= 0] += self.L_lambda[self.L_lambda > 0].min()
        self.sed_unit = 'Lsun/Angstrom/Msun'

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

        self.metallicities = np.loadtxt(self.ssp_properties_path, usecols=(1))
        self.ages = np.loadtxt(self.ssp_properties_path, usecols=(0))
        self.log_ages_yr = np.log10(self.ages)
        print("> Initialising GRANADA models (IMF=Salpeter)")
        ssp_fits = fits.open(self.path)
        wl0 = ssp_fits[0].header['CRVAL1']
        deltawl = ssp_fits[0].header['CDELT1']

        self.header = ssp_fits[0].header
        self.norm = np.ones(156)  # Lsun/AA/Msun
        for i in range(156):
            self.norm[i] = ssp_fits[0].header['NORM'+str(i)]

        SED = ssp_fits[0].data
        SED *= self.norm[:, np.newaxis]
        SED = SED.reshape(39, 4, -1)
        self.norm = self.norm.reshape(39, 4)
        self.age_sort = np.argsort(self.ages.reshape(39, 4), axis=0)
        SED = SED[self.age_sort[:, 0], :, :]
        self.norm = self.norm[self.age_sort[:, 0], :]
        SED = SED.transpose(1, 0, 2)
        self.norm = self.norm.transpose(1, 0)
        self.wavelength = np.arange(wl0, wl0 + deltawl * SED.shape[-1],
                                    deltawl)  # AA
        self.L_lambda = SED
        self.sed_unit = 'Lsun/Angstrom/Msun'
        ssp_fits.close()

        self.ages = np.unique(self.ages)
        self.log_ages_yr = np.log10(self.ages)
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
#         self.sed_unit = 'Lsun/Angstrom/Msun'


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
        self.ages = []
        self.metallicities = []
        for file in files:
            self.ages.append(float(file[file.find('T') + 1: file.find('_Z')]))
            self.metallicities.append(
                float(file[file.find('Z') + 1: file.find('_' + IMF)]))
        self.ages = np.unique(self.ages)
        self.log_ages_yr = np.log10(self.ages)
        self.metallicities = np.unique(self.metallicities)

        header = 'XSL_SSP_T'
        c_solar = self.C_imf[IMF.lower()]  # Convert to solar units

        with fits.open(os.path.join(self.path,
                                     header+'{:.2e}_Z{:}_{}_{}.fits'.format(
                self.ages[0], self.metallicities[0], IMF, ISO))) as hdul:
            self.wavelength = 10**(
                (np.arange(0, hdul[0].data.size, 1) - hdul[0].header['CRPIX1'])
                * hdul[0].header['CDELT1'] + hdul[0].header['CRVAL1'] + 1)

        self.L_lambda = self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size,
                   self.wavelength.size), dtype=np.float32)

        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.ages):
                filename = header+'{:.2e}_Z{:}_{}_{}.fits'.format(age, Z, IMF,
                                                                  ISO)
                file = os.path.join(self.path, filename)
                with fits.open(file) as hdul:
                    spec = hdul[0].data * c_solar
                    hdul.close()
                self.L_lambda[i][j] = spec
        self.sed_unit = 'Lsun/Angstrom/Msun'


if __name__ == '__main__':
    # ssp = PopStar(IMF='cha_0.15_100')
    from matplotlib import pyplot as plt
    ssp = BaseGM()

# %%                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/

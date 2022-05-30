import os
import numpy as np
import sys

from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
import h5py


class SSP(object):

    def compute_SED(self, time, mass, metallicity,
                    dust_model=None,
                    plot_interpolation=False, plot_weights=False):
        """
        This method computes the corresponding SED for a given star formation
        history.
        -----------------------------------------------------------------------
        params:
            -- time: Time steps expressed in Gyr corresponding to the age of
                the Universe
            -- mass: Cumulative s tellar mass for each time step
            -- metallicity: Average metallicity corresponding to each time step
        """
        # SSP time steps to interpolate (i.e. lookback time)
        t_i = self.log_ages_yr - np.ediff1d(self.log_ages_yr, to_begin=0)/2
        t_i = np.append(t_i, 12)  # 1000 Gyr
        # Conversion to time since the onset of SF (i.e. age of the Universe)
        today = time.max()
        t_i = today - (np.power(10, t_i) * u.yr)
        t_i[0] = today
        t_i.clip(time.min(), today, out=t_i)
        interp_M_i = np.interp(t_i, time, mass)
        M_i = -np.ediff1d(interp_M_i)
        print(M_i)
        Z_i = np.interp(t_i, time, np.log10(metallicity))
        Z_i = 10**Z_i
        # Z_i = -np.ediff1d( Z_i ) / (M_i+u.kg)
        # to prevent extrapolation
        Z_i.clip(self.metallicities[0], self.metallicities[-1], out=Z_i)
        extinction = np.ones((M_i.size,
                              self.L_lambda[0][0].spectral_axis.size))
        if dust_model:
            log_t_mid = (np.log10(t_i[:-1])+np.log10(t_i[1:]))/2
            for ii, log_t_i in enumerate(log_t_mid):
                extinction[ii, :] = dust_model(10**log_t_i/u.yr,
                                               self.wavelength)
        SED = np.zeros(self.L_lambda[0][0].spectral_axis.size)
        weights = np.zeros((t_i.size, self.metallicities.size))
        for i, mass_i in enumerate(M_i):
            # print(t_i[i]/u.Gyr, self.log_ages_yr[i],'\t', m/u.Msun, Z_i[i])
            if mass_i > 0:
                index_Z_hi = self.metallicities.searchsorted(Z_i[i]).clip(
                    1, len(self.metallicities)-1)
                # log interpolation in Z --> Crashes when there is only 1 met?
                weight_Z_hi = (np.log(Z_i[i]/self.metallicities[index_Z_hi-1])
                               / np.log(self.metallicities[index_Z_hi]
                                        / self.metallicities[index_Z_hi-1]))
                if np.isnan(weight_Z_hi):
                    weight_Z_hi = 0
                SED = SED + extinction[i, :] * mass_i * (
                    self.L_lambda[index_Z_hi][i].flux * weight_Z_hi +
                    self.L_lambda[index_Z_hi-1][i].flux * (1-weight_Z_hi))
                weights[i, index_Z_hi] = weight_Z_hi * mass_i.value
                weights[i, index_Z_hi-1] = (1-weight_Z_hi) * mass_i.value
                index_Z_hi = self.metallicities.searchsorted(Z_i[i]).clip(
                    1, len(self.metallicities)-1)
                weights /= np.sum(M_i.value)

        if plot_interpolation:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.plot(np.log10(time/u.Gyr), np.log10(mass), 'k-o', label='input')
            ax.plot(np.log10(t_i/u.Gyr), np.log10(interp_M_i), 'r-o',
                    label='input')
            ax.set_ylabel(r'$\log_{10}(M_*)$')
            ax.set_xlabel(r'$\log_{10}(t/Gyr)$')
            ax = fig.add_subplot(212)
            ax.plot(np.log10(time/u.Gyr), np.log10(metallicity/0.02),
                    'k-o', label='input')
            ax.plot(np.log10(t_i/u.Gyr), np.log10(Z_i/0.02), 'r-o',
                    label='interp')
            ax.set_ylabel(r'$\log_{10}(Z_*/Z_\odot)$')
            ax.set_xlabel(r'$\log_{10}(t/Gyr)$')

            fig.subplots_adjust(hspace=0)
            plt.show()
            plt.close()
        if plot_weights:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(np.log10(weights.T), aspect='auto', origin='lower',
                       interpolation='none', cmap='rainbow', vmin=-4)
            nticks = np.arange(0, self.log_ages_yr.size, 15)
            plt.xticks(nticks, labels=np.round(self.log_ages_yr[nticks],
                                               decimals=2))
            plt.xlabel(r'$\log_{10}(age_{SSP}/yr)$')
            plt.grid(b=True)
            plt.colorbar()
            return SED, weights
        else:
            return SED, weights

    def cut_models(self, wl_min, wl_max):
        cut_pts = np.where((self.wavelength >= wl_min) &
                           (self.wavelength <= wl_max))[0]
        if len(cut_pts) == 0:
            raise NameError(
                'Wavelength cuts {}, {} out of range for array with lims {} {}'
                .format(wl_min, wl_max, self.wavelength.min(),
                        self.wavelength.max())
                            )
        else:
            self.SED = self.SED[:, :, cut_pts]
            self.wavelength = self.wavelength[cut_pts]
            print('Models cut between {} {}'.format(wl_min, wl_max))

    def interpolate_sed(self, new_wl_edges):
        new_wl = (new_wl_edges[1:] + new_wl_edges[:-1]) / 2
        dwl = np.diff(new_wl_edges)
        ori_dwl = np.diff(self.L_lambda[0, 0].bin_edges)
        print(' [SSP] Interpolating SSP SEDs')
        for i in range(self.L_lambda.shape[0]):
            for j in range(self.L_lambda.shape[1]):
                f = np.interp(new_wl_edges, self.L_lambda[i, j].spectral_axis,
                              np.cumsum(self.L_lambda[i, j].flux * ori_dwl))
                new_flux = np.diff(f) / dwl
                self.L_lambda[i, j] = Spectrum1D(flux=new_flux,
                                                 spectral_axis=new_wl)
        self.wavelength = new_wl

    def get_mass_lum_ratio(self, wl_range):
        pts = np.where((self.wavelength >= wl_range[0]) &
                       (self.wavelength <= wl_range[1]))[0]
        self.mass_to_lum = np.empty((self.metallicities.size,
                                     self.ages.size)
                                    ) * 1/self.L_lambda[0, 0].flux.unit
        for i in range(self.metallicities.size):
            for j in range(self.ages.size):
                self.mass_to_lum[i, j] = 1/np.mean(
                    self.L_lambda[i, j].flux[pts])


class PopStar(SSP):

    def __init__(self, IMF, nebular=False):
        self.path = os.path.join(os.path.dirname(__file__),
                                 'data', 'PopStar')
        self.metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02,
                                       0.05])
    # Z_sun=0.0134 # FIXME: Hardcoded!!!!
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
        self.ages = 10**self.log_ages_yr * u.yr
        # isochrone age in delta [log(tau)]=0.01
        self.wavelength = np.loadtxt(os.path.join(
            self.path, 'SED', 'spneb_kro_0.15_100_z0500_t9.95'), dtype=float,
            skiprows=0, usecols=(0,), unpack=True) * u.angstrom
        print("> Initialising Popstar models (IMF='"+IMF+"')")
        self.L_lambda = np.empty(
            shape=(self.metallicities.size, self.log_ages_yr.size),
            dtype=Spectrum1D)
        if nebular:
            column = 3
            print('--> Including NEBULAR emission')
        else:
            column = 1
            print('--> Only stellar continuum')
        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.log_ages_yr):
                file = os.path.join(
                    self.path, 'SED',
                    'spneb_{0}_z{1:04.0f}_t{2:.2f}'.format(IMF, Z*1e4, age))
                spec = np.loadtxt(
                    file, dtype=float, skiprows=0, usecols=(column),
                    unpack=True) * u.Lsun/u.Angstrom/u.Msun
                # TODO: Decide Flam, Fnu, or nuFnu
                self.L_lambda[i][j] = Spectrum1D(flux=spec,
                                                 spectral_axis=self.wavelength)


class PyPopStar(SSP):

    def __init__(self, IMF, nebular=False):
        self.path = os.path.join(os.path.dirname(__file__), 'data/PyPopStar')
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
        self.ages = 10**self.log_ages_yr * u.yr
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
        self.L_lambda = np.empty(shape=(self.metallicities.size,
                                        self.log_ages_yr.size),
                                 dtype=Spectrum1D)

        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.log_ages_yr):
                filename = header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(Z, age)
                file = os.path.join(self.path, IMF, filename)
                with fits.open(file) as hdul:
                    spec = hdul[1].data[column] * u.Lsun/u.angstrom/u.Msun
                    self.wavelength = hdul[1].data['wavelength'] * u.angstrom
                    hdul.close()
                self.L_lambda[i][j] = Spectrum1D(flux=spec,
                                                 spectral_axis=self.wavelength)

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


class BaseGM(SSP):

    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__),
                                 'data/BaseGM/gsd01_156.fits')
        self.ssp_properties_path = os.path.join(
            os.path.dirname(__file__), 'data/BaseGM/fits_like_properties.dat')
        self.metallicities = np.loadtxt(self.ssp_properties_path, usecols=(1))
        self.ages = np.loadtxt(self.ssp_properties_path, usecols=(0)) * u.yr
        self.log_ages_yr = np.log10(self.ages.value)
        print("> Initialising GRANADA models (IMF=Salpeter)")
        ssp_fits = fits.open(self.path)
        self.header = ssp_fits[0].header
        self.norm = np.ones(156) * u.Lsun/u.Angstrom/u.Msun
        for i in range(156):
            self.norm[i] = (ssp_fits[0].header['NORM'+str(i)]
                            * u.Lsun/u.Angstrom/u.Msun)

        SED = ssp_fits[0].data * u.Lsun/u.Angstrom/u.Msun
        SED *= self.norm[:, np.newaxis].value
        SED = SED.reshape(39, 4, -1)
        self.norm = self.norm.reshape(39, 4)

        self.age_sort = np.argsort(self.ages.reshape(39, 4), axis=0)

        wl0 = ssp_fits[0].header['CRVAL1']
        deltawl = ssp_fits[0].header['CDELT1']
        self.wavelength = np.arange(wl0, wl0 + deltawl * SED.shape[-1],
                                    deltawl) * u.angstrom

        SED = SED[self.age_sort[:, 0], :, :]
        self.norm = self.norm[self.age_sort[:, 0], :]
        SED = SED.transpose(1, 0, 2)
        self.norm = self.norm.transpose(1, 0)

        ssp_fits.close()

        self.ages = np.unique(self.ages)
        self.log_ages_yr = np.log10(self.ages.value)
        self.metallicities = np.sort(np.unique(self.metallicities))

        self.L_lambda = np.empty(shape=(self.metallicities.size,
                                        self.log_ages_yr.size),
                                 dtype=Spectrum1D)
        for i in range(self.metallicities.size):
            for j in range(self.log_ages_yr.size):
                self.L_lambda[i][j] = Spectrum1D(flux=SED[i, j, :],
                                                 spectral_axis=self.wavelength)


class FSPS(SSP):

    def __init__(self):
        print("> Initialising FSPS models")
        file = h5py.File('data/FSPS/fsps_ssp_models.hdf5', 'r')
        elements = list(file.keys())
        log_ages = []
        metallicities = []
        for elem_i in elements:
            if elem_i.find('log_age') >= 0:
                init = 'log_age_'
                end = '_Z_'
                log_ages.append(elem_i[elem_i.find(init)+len(init):
                                       elem_i.find(end)])
                metallicities.append(elem_i[elem_i.find(end)+len(end):])
        metallicities = np.sort(np.unique(metallicities))
        log_ages = np.unique(log_ages)
        wavelength_unit = u.Unit(file['wavelength_units'][()])
        self.wavelength = file['wavelength'][()] * wavelength_unit
        l_lambda_unit = u.Unit(file['L_lambda_units'][()])

        self.metallicities = np.array(metallicities, dtype=float)
        self.log_ages_yr = np.sort(np.array(log_ages, dtype=float))
        self.ages = 10**self.log_ages_yr * u.yr

        self.L_lambda = np.empty(shape=(self.metallicities.size,
                                        self.log_ages_yr.size),
                                 dtype=Spectrum1D)
        for i, met in enumerate(metallicities):
            for j, log_age in enumerate(log_ages):
                name = 'log_age_{}_Z_{}'.format(log_age, met)
                l_lambda = file[name]['L_lambda'][()]
                self.L_lambda[i][j] = Spectrum1D(flux=l_lambda * l_lambda_unit,
                                                 spectral_axis=self.wavelength)
        file.close()


class XSL(SSP):
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

    def __init__(self, IMF, ISO, path_to_lib=None):
        print("> Initialising X-Shooter (XSL) models (IMF={}, ISO={})".format(
            IMF, ISO))
        if (IMF != 'Kroupa') & (IMF != 'Salpeter'):
            raise NameError('IMF not valid (use Kroupa | Salpeter)')
        if (ISO != 'P00') & (ISO != 'PC'):
            raise NameError('ISO not valid (use P00 for Padova2000 or PC for PARSEC/COLIBRI)')
        if path_to_lib:
            self.path = os.path.join(path_to_lib, '_'.join([IMF, ISO]))
        else:
            self.path = os.path.join(os.path.dirname(__file__), 'data/XSL',
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
        self.ages = np.unique(self.ages) * u.yr
        self.log_ages_yr = np.log10(self.ages / u.yr).value
        self.metallicities = np.unique(self.metallicities)

        self.L_lambda = np.empty(shape=(self.metallicities.size,
                                        self.log_ages_yr.size),
                                 dtype=Spectrum1D)

        header = 'XSL_SSP_T'
        c_solar = self.C_imf[IMF.lower()]  # Convert to solar units
        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.ages.value):
                filename = header+'{:.2e}_Z{:}_{}_{}.fits'.format(age, Z, IMF, ISO)
                file = os.path.join(self.path, filename)
                with fits.open(file) as hdul:
                    spec = hdul[0].data * c_solar * u.Lsun/u.angstrom/u.Msun
                    self.wavelength = 10**(
                        (np.arange(0, spec.size, 1) - hdul[0].header['CRPIX1'])
                        * hdul[0].header['CDELT1'] + hdul[0].header['CRVAL1']
                        + 1) * u.angstrom
                    hdul.close()
                self.L_lambda[i][j] = Spectrum1D(flux=spec,
                                                 spectral_axis=self.wavelength)


if __name__ == '__main__':
    # ssp = PopStar(IMF='cha_0.15_100')
    from matplotlib import pyplot as plt
    ssp = XSL(IMF='Kroupa', ISO='P00', path_to_lib='/home/pablo/SSP_TEMPLATES/XSL')

# %%                                                    ... Paranoy@ Rulz! ;^D

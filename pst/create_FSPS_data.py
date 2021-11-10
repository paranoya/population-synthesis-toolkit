import os
import numpy as np
import sys

from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from specutils import Spectrum1D


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
        cut_pts = np.where((self.wavelength>=wl_min)&(self.wavelength<=wl_max))[0]
        if len(cut_pts) == 0:
            raise NameError('Wavelength cuts {}, {} out of range for array with lims {} {}'.format(
                wl_min, wl_max, self.wavelength.min(), self.wavelength.max())
                            )
        else:
            self.SED = self.SED[:, :, cut_pts]
            self.wavelength = self.wavelength[cut_pts]
            print('Models cut between {} {}'.format(wl_min, wl_max))


class FSPS(SSP):

    def __init__(self, **kwargs):
        import fsps  # FIXME: Only available when FSPS is installed
        print("> Initialising FSPS models")
        self.metallicities = kwargs['metallicities']
        fsps_metallicities = np.log10(self.metallicities/0.02)
        self.ages = kwargs['ages']
        fsps_ages = self.ages.to(u.Gyr).value
        self.log_ages_yr = np.log10(self.ages.to('yr').value)
        print(
            " > Creating model grid with {} metallicities and {} ages".format(
                self.metallicities.size, self.ages.size))
        # FSPS parameters and first initialization
        params = {}
        params['compute_vega_mags'] = kwargs.get('compute_vega_mags', False)
        params['zcontinuous'] = kwargs.get('zcontinuous', 1)
        params['sfh'] = kwargs.get('sfh', 0)
        params['dust_type'] = kwargs.get('dust_type', 0)
        params['dust_tesc'] = kwargs.get('dust_tesc', 7)
        params['dust1'] = kwargs.get('dust1', 0)
        params['dust2'] = kwargs.get('dust2', 0)
        params['dust_index'] = kwargs.get('dust_index', -0.7)
        params['dust1_index'] = kwargs.get('dust1_index', -0.7)
        params['imf_type'] = kwargs.get('imf_type', 1)
        params['mwr'] = kwargs.get('mwr', 3.1)
        params['add_neb_emission'] = kwargs.get('add_neb_emission', False)
        params['add_dust_emission'] = kwargs.get('add_dust_emission', False)
        params['add_neb_continuum'] = kwargs.get('add_neb_continuum', False)
        # There are MANY more params...
        sp = fsps.StellarPopulation(**params)
        print(sp.libraries)
        print('  > Isochrones: {}\n  > SSP: {}'.format(sp.libraries[0],
                                                       sp.libraries[1]))
        self.L_lambda = np.empty(shape=(self.metallicities.size,
                                        self.log_ages_yr.size),
                                 dtype=Spectrum1D)
        for i, met_i in enumerate(fsps_metallicities):
            sp.params['logzsol'] = met_i
            # FIXME: Gas metallicity is fixed to Z(SSP) (recommended by FSPS)
            sp.params['gas_logz'] = met_i
            for j, age_j in enumerate(fsps_ages):
                wave, spec = sp.get_spectrum(tage=age_j, peraa=True)
                # alive_mass = sp.stellar_mass
                self.L_lambda[i, j] = Spectrum1D(
                    flux=spec * u.Lsun/u.Angstrom/u.Msun,
                    spectral_axis=wave * u.angstrom)
        self.wavelength = wave


if __name__ == '__main__':
    # ssp = PopStar(IMF='cha_0.15_100')
    import h5py
    log_age = np.array([5.00, 5.48, 5.70, 5.85, 6.00, 6.10, 6.18,
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
    metals = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
    ssp = FSPS(ages=10**log_age * u.yr, metallicities=metals)
    f = h5py.File('data/FSPS/fsps_ssp_models.hdf5', 'a')
    for i, log_age_i in enumerate(log_age):
        for j, met_j in enumerate(metals):
            print(i, j)
            grp = f.create_group('log_age_{:2.2f}_Z_{:.4f}'.format(log_age_i,
                                                                   met_j))
            grp.create_dataset('L_lambda', data=ssp.L_lambda[j, i].flux.value)

    f.create_dataset('L_lambda_units', data=str(ssp.L_lambda[j, i].flux.unit))
    f.create_dataset('wavelength', data=ssp.L_lambda[j, i].spectral_axis.value)
    f.create_dataset('wavelength_units',
                     data=str(ssp.L_lambda[j, i].spectral_axis.unit))

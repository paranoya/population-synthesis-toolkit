#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u
import pst


# %%

class Polynomial_MFH_fit:

    def __init__(self, N, SSP, obs_filters, t_obs, **kwargs):
        self.t_obs = t_obs
        # self.t_start = kwargs.get('t_start', 0*u.Gyr)
        # self.t_end = kwargs.get('t_end', t_obs)
        primordial_coeffs = []
        primordial_L = []
        for n in range(N):
            # M_hat(t_hat) = t_hat^n - t_hat^N
            c = np.zeros(N+1)
            c[n] = 1
            c[-1] = -1
            primordial_coeffs.append(c)

            L = []
            p = pst.models.Polynomial_MFH(
                # t_start=self.t_start, t_end=self.t_end,
                coeffs=c)
            sed = p.compute_SED(SSP, t_obs)
            for filter_name in obs_filters:
                photo = pst.observables.luminosity(
                    flux=sed, wavelength=SSP.wavelength, filter_name=filter_name)
                L.append(photo.integral_flux.to_value(u.Lsun))  # linalg complains about units
            primordial_L.append(np.array(L))
            # primordial_L.append(np.array(L, dtype=u.Quantity))  # with units

            # plt.figure()
            # plt.title(r'$\hat M(\hat t) = \hat t^{}-\hat t^{}$ ; coeffs={}'.format(n, N, c))
            # t = np.linspace(0, 14, 141)*u.Gyr
            # plt.plot(t, p.integral_SFR(t), 'k-', label='model')
            # plt.xlabel('t [Gyr]')
            # plt.ylabel(r'M [M$_\odot$]')
            # # plt.yscale('log')
            # plt.show()

        self.lstsq_solution = np.matmul(
            np.linalg.pinv(np.matmul(primordial_L, np.transpose(primordial_L))),
            primordial_L)
        self.primordial_coeffs = np.array(primordial_coeffs)
        self.primordial_L_Lsun = np.array(primordial_L)


    def fit(self, L_obs_Lsun):
        solution = np.matmul(self.lstsq_solution, L_obs_Lsun)
        c = np.matmul(solution, self.primordial_coeffs)
        return pst.models.Polynomial_MFH(
            # t_start=self.t_start, t_end=self.t_end,
            coeffs=c)


# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------

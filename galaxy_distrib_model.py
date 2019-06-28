#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:21:05 2019

@author: pablo
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


class Model_grid(object):

    def __init__(self, **kwargs):
        self.metallicities = [0.0004, 0.004, 0.008, 0.02, 0.05]
        self.log_Z = np.log10(self.metallicities)
        print('Metallicities :', self.metallicities)
#        self.SFH_model = kwargs['SFH']
#        self.load_models('Results/ExponentialSFRdelayed/tau_M/Mags_colours/')
        self.load_models('models/')
        self.compute_V_max()

    def load_models(self, path, n_metallicities_interpolation=0):

        tau_model = []
        u_model = []
        g_model = []
        r_model = []
        i_model = []
        z_model = []

        for Z_i in self.metallicities:  # TODO: Include IMF
            tau, u, g, r, i, z = np.loadtxt(
                    path +
                    'Magn_AB_SDSS__SED_kro_0.15_100_Z_{:.4f}_.txt'.format(Z_i),
                    skiprows=1, usecols=(0, 1, 2, 3, 4, 5), unpack=True)
            tau_model.append(tau)
            u_model.append(u)
            g_model.append(g)
            r_model.append(r)
            i_model.append(i)
            z_model.append(z)

        print('Models loaded')
        self.tau = tau_model[-1]
        self.log_tau = np.log10(self.tau)
        print('tau_min = {:.2f}, tau_max = {:.2f}'.format(
                min(self.tau), max(self.tau)))

        # Define range of metallicities
        if n_metallicities_interpolation > 0:
            new_metallicities = np.linspace(self.log_Z[0], self.log_Z[-1],
                                            n_metallicities_interpolation)

            u_model = interp1d(self.log_Z, u_model, axis=0)(new_metallicities)
            g_model = interp1d(self.log_Z, g_model, axis=0)(new_metallicities)
            r_model = interp1d(self.log_Z, r_model, axis=0)(new_metallicities)
            i_model = interp1d(self.log_Z, i_model, axis=0)(new_metallicities)
            z_model = interp1d(self.log_Z, z_model, axis=0)(new_metallicities)

            self.metallicities = 10**new_metallicities
            self.log_Z = new_metallicities
            print('New range of metallicities: \n', new_metallicities)

        # Add axis for M_inf and create model grid
        self.log_M_inf = np.linspace(6, 14, 100)
        self.M_inf = 10**self.log_M_inf

        self.u = (np.array(u_model)[:, :, np.newaxis]
                  - 2.5*self.log_M_inf[np.newaxis, np.newaxis, :])
        self.g = (np.array(g_model)[:, :, np.newaxis]
                  - 2.5*self.log_M_inf[np.newaxis, np.newaxis, :])
        self.r = (np.array(r_model)[:, :, np.newaxis]
                  - 2.5*self.log_M_inf[np.newaxis, np.newaxis, :])
        self.i = (np.array(i_model)[:, :, np.newaxis]
                  - 2.5*self.log_M_inf[np.newaxis, np.newaxis, :])
        self.z = (np.array(z_model)[:, :, np.newaxis]
                  - 2.5*self.log_M_inf[np.newaxis, np.newaxis, :])

    def compute_V_max(self):
        u_limit = 19
        g_limit = 19
        r_limit = 17.77
        z_min = 0.02
        z_max = 0.07
        H0 = 70  # km/s/Mpc
        D_z_min = z_min * 3e5/H0
        D_z_max = z_max * 3e5/H0
        survey_solid_angle = 1317  # deg**2
        survey_solid_angle *= (np.pi/180)**2  # steradian

        D_u = 10**(0.2 * (u_limit-self.u) - 5)  # Mpc
        D_g = 10**(0.2 * (g_limit-self.g) - 5)  # Mpc
        D_r = 10**(0.2 * (r_limit-self.r) - 5)  # Mpc

        D_max = np.min([D_u, D_g, D_r], axis=(0)).clip(D_z_min, D_z_max)
        self.V_max = survey_solid_angle * (D_max**3 - D_z_min**3) / 3

    def bayesian_model_assignment(self, **kwargs):
        u = kwargs['u']
        g = kwargs['g']
        r = kwargs['r']
#        i = kwargs['i']
#        z = kwargs['z']

        u_err = kwargs['u_err']
        g_err = kwargs['g_err']
        r_err = kwargs['r_err']
#        i_err = kwargs['i_err']
#        z_err = kwargs['z_err']
        Z_galaxy = []
        tau_galaxy = []
        M_inf_galaxy = []

        self.min_err = 0.
        min_err =self.min_err
        # TODO: Check number of given elements, include the possibility of varying the number of constrains

#        for galaxy in range(len(u)):
        for galaxy in range(0, 20000):
            print('# Total % {:.2f}'.format(galaxy/len(u)*100))
            likelihood = (
                    ((u[galaxy]-self.u_model_grid)**2)/(min_err+u_err[galaxy])**2 +(
                    (g[galaxy]-self.g_model_grid)**2)/(min_err+g_err[galaxy])**2 + (
                    (r[galaxy]-self.r_model_grid)**2)/(min_err+r_err[galaxy])**2) # + (
#                    (i[galaxy]-i_model_grid)**2)/(min_err+i_err[galaxy])**2) # +(
#                    (z[galaxy]-z_model_grid)**2)/(min_err+z_err[galaxy])**2 )

            likelihood = np.exp(- likelihood/.5)

            dp_dlogtau = np.sum(likelihood, axis=(0, 2))
            log_tau_i = np.sum(self.log_tau*dp_dlogtau)/np.sum(dp_dlogtau)
            tau_galaxy.append(10**log_tau_i)

            dp_dlogZ = np.sum(likelihood, axis=(1, 2))
            log_Z_i = np.sum(self.log_Z*dp_dlogZ)/np.sum(dp_dlogZ)
            Z_galaxy.append(10**log_Z_i)

            dp_dlogMinf = np.sum(likelihood, axis=(0, 1))
            log_M_inf_i = np.sum(self.log_M_inf*dp_dlogMinf)/np.sum(dp_dlogMinf)
            M_inf_galaxy.append(10**log_M_inf_i)

        print('Bayesian assignment finished!, returned: Z_galaxy[], tau_galaxy[], M_inf_galaxy[]')

        return np.array(Z_galaxy), np.array(tau_galaxy), np.array(M_inf_galaxy)


class Ansatz():


    def alpha12beta_powlaw(tau, tau_0, alpha1, alpha2, beta):

        x = tau/tau_0
        dN_dtau = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)
        dtau = np.interp(tau, tau[0:-1], np.diff(tau))
#        return  dN_dtau
        return  dN_dtau/(np.sum(dN_dtau*dtau))

    def dpdtau(M, tau):
        """The propability of presenting tau given a mass"""
        eta_tau = 0.07
        tau_c = 2.454
        tau_0 = tau_c*(M/(10**(10.13)))**eta_tau

        eta1 = -0.65
        alpha_c1 = 3.7
        alpha1 = 1 +(alpha_c1-1)*(M/10**(10.13))**(eta1)

        eta2 = 0.65
        alpha_c2 = 3.7
        alpha2 = 1 +(alpha_c2-1)*(M/10**(10.13))**(eta2)

        beta = 4

        dpdtau = Ansatz.alpha12beta_powlaw(tau,
                                          tau_0,
                                          alpha1,
                                          alpha2,
                                          beta)
        return dpdtau

    def dndMass(M, Mprime=10**(10.6), alpha= -1.25, phi=1.5*1e10):
        """schechter_mass_function"""
        return phi*np.exp(-(M/Mprime))/Mprime *(M/Mprime)**(alpha+1)

    def dNdMdtau(mass_range, tau_range):

        """CONVENTION: Mass is the first dimension"""
        """CONVENTION: Provide always logspace """

        """dpdtau"""
        eta_tau = 0.07
        tau_c = 2.454
        tau_0 = tau_c*(mass_range/(10**(10.13)))**eta_tau

        eta1 = -0.65
        alpha_c1 = 3.7
        alpha1 = 1 +(alpha_c1-1)*(mass_range/10**(10.13))**eta1

        eta2 = 0.65
        alpha_c2 = 3.7
        alpha2 = 1 +(alpha_c2-1)*(mass_range/10**(10.13))**eta2

        beta = 4

        x = tau_range[np.newaxis, :]/ tau_0[:, np.newaxis]

        dp_dtau = pow(x, alpha1[:, np.newaxis])/pow(
                1+x**beta, (
                alpha1[:, np.newaxis]+alpha2[:, np.newaxis])/beta)

        # Trying to avoid wrong normalization by increasing the number of points

        interpolator = interpolate.interp1d(tau_range, dp_dtau, axis=1)
        tau_interp =np.logspace(np.log10(tau_range[0]), np.log10(tau_range[-1]),
                                                          len(tau_range)*1000)
        dp_dtau_interp = interpolator(tau_interp)

        int_dp_dtau = np.trapz(dp_dtau_interp, tau_interp)


#        dtau = np.interp(tau_range, tau_range[0:-1], np.diff(tau_range))

        # integration along tau axis
#        int_dp_dtau = np.sum(dp_dtau*dtau[np.newaxis, :], axis=1)

        dp_dtau_norm = dp_dtau/int_dp_dtau[:, np.newaxis]


        """dNdMass"""
        Mprime=10**(10.6)
        alpha_sch= -1.25
        phi=1.5*1e10
        dndM = phi*np.exp(-(mass_range/Mprime))/Mprime *(
                mass_range/Mprime)**(alpha_sch+1)

        """dndMdtau"""

        return dp_dtau_norm*dndM[:, np.newaxis], int_dp_dtau, dp_dtau_norm, dndM
#        return dp_dtau_norm

#
#
#        dpdtau = Ansatz.alpha12beta_powlaw(tau,
#                                          tau_0,
#                                          alpha1,
#                                          alpha2,
#                                          beta)
#


if __name__ == "__main__":

    models = Model_grid()

    tau_grid = np.ones_like(models.u) * models.tau[np.newaxis, :, np.newaxis]

#    plt.figure()
#    plt.plot(tau_grid.flatten(), (models.u-models.r).flatten(), 'k,')

    plt.figure()
    plt.title(r'log( Vmax/Mpc$^3$ )')
    plt.imshow(np.log10(models.V_max[-2]+1),
               extent=[
                       models.log_M_inf[0], models.log_M_inf[-1],
                       models.log_tau[0], models.log_tau[-1]
                       ],
               vmin=3.3, vmax=6.7, cmap='gist_earth', aspect='auto'
               )
    plt.xlabel(r'log(M$_\infty$)')
    plt.ylabel(r'log($\tau$)')
    plt.colorbar()

# ... Paranoy@ rulz!

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:21:05 2019

@author: pablo
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


def V_max(u, g, r):
    u_limit = 19
    g_limit = 19
    r_limit = 17.77
    z_min = 0.02
    z_max = 0.07
    H0 = 95  # km/s/Mpc
    D_z_min = z_min * 3e5/H0
    D_z_max = z_max * 3e5/H0
    survey_solid_angle = 1317  # deg**2
    survey_solid_angle *= (np.pi/180)**2  # steradian

    D_u = 10**(0.2 * (u_limit-u) - 5)  # Mpc
    D_g = 10**(0.2 * (g_limit-g) - 5)  # Mpc
    D_r = 10**(0.2 * (r_limit-r) - 5)  # Mpc

    D_max = np.min([D_u, D_g, D_r], axis=(0)).clip(D_z_min, D_z_max)
    return survey_solid_angle * (D_max**3 - D_z_min**3) / 3


# =============================================================================
class Ansatz():

    def parameters_given_M(M_star):
        M_sym = 10**10
        tau_sym = 2.16
        alpha_sym = 3.6

        eta_tau = 0.09
        eta1 = -0.3
        eta2 = 1

        tau_0 = tau_sym*(M_star/M_sym)**eta_tau
        alpha1 = -1 + (alpha_sym+1)*(M_star/M_sym)**(eta1)
        alpha2 = 1 + (alpha_sym-1)*(M_star/M_sym)**(eta2)
        beta = 4

        return tau_0, alpha1, alpha2, beta

    def dp_dtau_given_M(M_star, tau):
        """
        Probability density of the characteristic time
        scale tau (scalar or array) for a given stellar mass (must be scalar)
        """
        tau_0, alpha1, alpha2, beta = Ansatz.parameters_given_M(M_star)

        x = tau/tau_0
        dp_dtau = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)

        # Normalize dp_dtau by integrating (do the integral in log tau)
        log_tau_interp = np.linspace(-2, 4, 1000)
        x = 10**log_tau_interp/tau_0
        dp_dtau_interp = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)
        integral_dpdtau_dtau = np.log(10)*tau_0 * np.trapz(dp_dtau_interp*x,
                                                           log_tau_interp)
        return dp_dtau / integral_dpdtau_dtau

    def dp_dtau_grid(M_star, tau):
        tau_0, alpha1, alpha2, beta = Ansatz.parameters_given_M(M_star)

        x = tau[:, np.newaxis]/tau_0[np.newaxis, :]
        dp_dtau = pow(x, alpha1[np.newaxis, :]) / pow(
                1+x**beta, (alpha1[np.newaxis, :]+alpha2[np.newaxis, :])/beta)

        # Normalize dp_dtau by integrating (do the integral in log tau)
        log_tau_interp = np.linspace(-2, 4, 1000)
        x = 10**log_tau_interp[:, np.newaxis]/tau_0[np.newaxis, :]
        dp_dtau_interp = pow(x, alpha1[np.newaxis, :]) / pow(
                1+x**beta, (alpha1[np.newaxis, :]+alpha2[np.newaxis, :])/beta)
        integral_dpdtau_dtau = np.log(10)*tau_0 * np.trapz(
                dp_dtau_interp*x, log_tau_interp, axis=0)
        return dp_dtau / integral_dpdtau_dtau[np.newaxis, :]

    def dn_dM(M, M_schechter=10**(11), alpha=-1.55, phi=0.01):
        """
        Multiplicity function (number density of galaxies
        per unit comoving volume per unit mass), described
        as a Schechter function.
        """
        return phi*np.exp(-(M/M_schechter))/M_schechter * (M/M_schechter)**alpha

    def dn_dMdtau_grid(M, tau):
        """
        Returns number density of galaxies as a function of M and tau
        """
# Alternative way of creating dp_dtau grid:
#        N_tau = len(tau)
#        N_M = len(M)
#        dp_dtau_grid = Ansatz.dp_dtau_unnorm(M.repeat(N_tau), np.tile(tau, N_M))
#        dp_dtau_grid.shape = (N_M, N_tau)
#
#        log_tau = np.log10(tau)
#        d_log_tau = np.concatenate((
#                log_tau[1]-log_tau[0],
#                (log_tau[2:]-log_tau[:-2]) / 2,
#                log_tau[-1]-log_tau[-2]
#                ))
#        norm = np.log(10) * np.sum(
#                dp_dtau_grid * d_log_tau[np.newaxis, :]*tau[np.newaxis, :],
#                axis=1)
        dn_dM = Ansatz.dn_dM(M)
        return dn_dM[np.newaxis, :] * Ansatz.dp_dtau_grid(M, tau)


# =============================================================================
class Model_grid(object):

    def __init__(self, **kwargs):
#        self.metallicities = [0.0004, 0.004, 0.008, 0.02, 0.05]
        self.metallicities = [0.004, 0.008, 0.02]
        self.log_Z = np.log10(self.metallicities)
        print('Metallicities :', self.metallicities)

        self.log_M_star = np.linspace(8, 12, 200)
        self.M_star = 10**self.log_M_star

        self.load_models('models/', n_metallicities_interpolation=10)
        self.compute_V_max()
        self.compute_N_galaxies()

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
#            new_metallicities = np.linspace(self.log_Z[0], self.log_Z[-1],
#                                            n_metallicities_interpolation)
            new_metallicities = np.log10(np.linspace(self.metallicities[0], self.metallicities[-1],
                                            n_metallicities_interpolation))
            
            u_model = interp1d(self.log_Z, u_model, axis=0)(new_metallicities)
            g_model = interp1d(self.log_Z, g_model, axis=0)(new_metallicities)
            r_model = interp1d(self.log_Z, r_model, axis=0)(new_metallicities)
            i_model = interp1d(self.log_Z, i_model, axis=0)(new_metallicities)
            z_model = interp1d(self.log_Z, z_model, axis=0)(new_metallicities)

            self.metallicities = 10**new_metallicities
            self.log_Z = new_metallicities
            print('New range of metallicities: \n', new_metallicities)

        # Add axis for M_inf and create model grid

        t0 = 13.7  # Gyr
        x = t0/self.tau
        self.M_inf = self.M_star[np.newaxis, :] / (
                1 - np.exp(-x[:, np.newaxis])*(1+x[:, np.newaxis]))
        delta_mag = 2.5 * np.log10(self.M_inf)

        self.u = np.array(u_model)[:, :, np.newaxis] - delta_mag[np.newaxis, :, :]
        self.g = np.array(g_model)[:, :, np.newaxis] - delta_mag[np.newaxis, :, :]
        self.r = np.array(r_model)[:, :, np.newaxis] - delta_mag[np.newaxis, :, :]
        self.i = np.array(i_model)[:, :, np.newaxis] - delta_mag[np.newaxis, :, :]
        self.z = np.array(z_model)[:, :, np.newaxis] - delta_mag[np.newaxis, :, :]

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

    def compute_N_galaxies(self):
        """
        Compute number density of galaxies per unit comoving volume within
        each bin of the grid, according to the ansatz, as well as
        total number of galaxies observed by the SDSS, taking into account
        Vmax (depending on metallicity).
        """
        # Cosmic number density:
        self.n_ansatz = Ansatz.dn_dMdtau_grid(self.M_star, self.tau)
        d_log_tau = np.concatenate((
                np.array([self.log_tau[1]-self.log_tau[0]]),
                (self.log_tau[2:]-self.log_tau[:-2]) / 2,
                np.array([self.log_tau[-1]-self.log_tau[-2]])
                ))
        d_log_M = np.concatenate((
                np.array([self.log_M_star[1]-self.log_M_star[0]]),
                (self.log_M_star[2:]-self.log_M_star[:-2]) / 2,
                np.array([self.log_M_star[-1]-self.log_M_star[-2]])
                ))
        d_log_tau = np.abs(d_log_tau)
        self.n_ansatz *= self.tau[:, np.newaxis]*d_log_tau[:, np.newaxis]
        self.n_ansatz *= self.M_star[np.newaxis, :]*d_log_M[np.newaxis, :]
        self.n_ansatz *= np.log(10)**2

        # Observed by SDSS:
        self.N_galaxies = self.n_ansatz[np.newaxis, :, :] * self.V_max

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


# =============================================================================
if __name__ == "__main__":

    u, g, r, SDSS_V_max = np.loadtxt('SDSS/photometry.dat',
                                     usecols=(1, 3, 5, 6), unpack=True)
    models = Model_grid()

    plt.figure()
    yago = V_max(u, g, r)
    pablo = 1/SDSS_V_max
    plt.plot(yago, pablo/yago, 'k.', alpha=.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Yago')
    plt.ylabel('Pablo')
    plt.grid(b=True)

#    tau_grid = np.ones_like(models.u) * models.tau[np.newaxis, :, np.newaxis]
#    plt.figure()
#    plt.plot(tau_grid.flatten(), (models.u-models.r).flatten(), 'k,')

#    plt.figure()
#    plt.title(r'log( Vmax/Mpc$^3$ )')
#    plt.imshow(np.log10(models.V_max[-2]+1),
#               extent=[
#                       models.log_M_star[0], models.log_M_star[-1],
#                       models.log_tau[0], models.log_tau[-1]
#                       ],
#               vmin=3.3, vmax=6.7, cmap='gist_earth',
#               aspect='auto', origin='lower',
#               )
#    plt.xlabel(r'log(M$_*$ / M$_\odot$)')
#    plt.ylabel(r'log($\tau$ / Gyr)')
#    plt.colorbar()

#    plt.figure()
#    plt.plot(models.M_star, Ansatz.dn_dM(models.M_star))
#    plt.xlabel(r'M$_*$ / M$_\odot$')
#    plt.ylabel(r'dn / dM$_*$')
#    plt.xscale('log')
#    plt.yscale('log')

#    grid_p = Ansatz.dp_dtau_grid(models.M_star, models.tau)
#    plt.figure()
#    plt.title(r'log( dp/d$\tau$ )')
#    plt.imshow(np.log10(grid_p),
#               extent=[
#                       models.log_M_star[0], models.log_M_star[-1],
#                       models.log_tau[0], models.log_tau[-1]
#                       ],
#               aspect='auto', origin='lower',
#               vmin=-3.8, vmax=-0.2, cmap='gist_earth',
##               vmin=.1, vmax=0.7, cmap='gist_earth',
#               )
#    plt.xlabel(r'log(M$_*$ / M$_\odot$)')
#    plt.ylabel(r'log($\tau$)')
#    plt.colorbar()

#    plt.figure()
#    plt.plot(models.tau, Ansatz.dp_dtau_given_M(1e13, models.tau), 'k-')
#    plt.plot(models.tau, Ansatz.dp_dtau_given_M(1e11, models.tau), 'r-')
#    plt.plot(models.tau, Ansatz.dp_dtau_given_M(1e10, models.tau), 'g-')
#    plt.plot(models.tau, Ansatz.dp_dtau_given_M(1e9, models.tau), 'b-')
#    plt.plot(models.tau, grid_p[:, 50], 'b.')
#    plt.plot(models.tau, grid_p[:, 100], 'g.')
#    plt.plot(models.tau, grid_p[:, 150], 'r.')
#    plt.xlabel(r'$\tau$')
#    plt.ylabel(r'dp / d$\tau$')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.ylim(1e-6, 1)
#    plt.grid(b=True)

#    grid_n = Ansatz.dn_dMdtau_grid(models.M_star, models.tau)
#    plt.figure()
#    plt.title(r'log( dn/dMd$\tau$ )')
#    plt.imshow(np.log10(grid_n),
#               extent=[
#                       models.log_M_star[0], models.log_M_star[-1],
#                       models.log_tau[0], models.log_tau[-1]
#                       ],
#               aspect='auto', origin='lower',
#               vmin=-20, vmax=-13, cmap='gist_earth',
#               )
#    plt.xlabel(r'log(M$_*$ / M$_\odot$)')
#    plt.ylabel(r'log($\tau$)')
#    plt.colorbar()

#    for Z in [1, 2, 3]:
#        plt.figure()
#        plt.title(r'log (N_galaxies)')
#        plt.imshow(np.log10(models.N_galaxies[Z]),
#                   extent=[
#                           models.log_M_star[0], models.log_M_star[-1],
#                           models.log_tau[0], models.log_tau[-1]
#                           ],
#                   aspect='auto', origin='lower',
#                   vmin=-1.5, vmax=1.5, cmap='gist_earth',
#                   )
#        plt.xlabel(r'log(M$_*$ / M$_\odot$)')
#        plt.ylabel(r'log($\tau$)')
#        plt.colorbar()

    alfa_models = 0.5
    plt.figure()
    for Z in [2, 4, 6]:
        plt.hist(models.u[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -16, 50),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='bar', alpha=alfa_models)
    plt.hist(u,
             bins=np.linspace(-24, -16, 50),
             label=str(models.metallicities[Z]), histtype='step', alpha=alfa_models)
    plt.xlabel(r'u')
    plt.ylabel(r'N')
    plt.grid(b=True)
    plt.legend()

    plt.figure()
    for Z in [2, 4, 6]:
        plt.hist(models.g[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -16, 50),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='bar', alpha=alfa_models)
    plt.hist(g,
             bins=np.linspace(-24, -16, 50),
             label=str(models.metallicities[Z]), histtype='step', alpha=alfa_models)
    plt.xlabel(r'g')
    plt.ylabel(r'N')
    plt.grid(b=True)
    plt.legend()

    plt.figure()
    for Z in [2, 4, 6]:
        plt.hist(models.r[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -16, 50),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='bar', alpha=alfa_models)
    plt.hist(r,
             bins=np.linspace(-24, -16, 50),
             label=str(models.metallicities[Z]), histtype='step', alpha=alfa_models)
    plt.xlabel(r'r')
    plt.ylabel(r'N')
    plt.grid(b=True)
    plt.legend()

    n_bins = 50
    plt.figure()
    for Z in [2, 4, 6]:
        u_r = models.u[Z] - models.r[Z]
        plt.hist(u_r.flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(1, 3.5, n_bins),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='bar', alpha=alfa_models)
    plt.hist(u-r,
             bins=np.linspace(1, 3.5, n_bins),
             label='SDSS', histtype='step')
    plt.xlabel(r'u-r')
    plt.ylabel(r'N')
    plt.grid(b=True)
    plt.legend()

    plt.figure()
    for Z in [2, 4, 6]:
        g_r = models.g[Z] - models.r[Z]
        plt.hist(g_r.flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(0.2, 1.2, n_bins),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='bar', alpha=alfa_models)
    plt.hist(g-r,
             bins=np.linspace(0.2, 1.2, n_bins),
             label='SDSS', histtype='step')
    plt.xlabel(r'g-r')
    plt.ylabel(r'N')
    plt.grid(b=True)
    plt.legend()

    plt.figure()
    plt.plot(g-r, u-r, 'g.', alpha=.05)
    for Z in [2, 4, 6]:
        u_r = models.u[Z] - models.r[Z]
        g_r = models.g[Z] - models.r[Z]
        plt.plot(g_r, u_r, 'k.')
    plt.xlim(0, 2)
    plt.ylim(0, 4)

    plt.figure()
    plt.hist2d(r, u-r,
               range=[[-24, -16], [.5, 3.5]], bins=30,
               cmap='gist_earth')
    plt.xlim(-24, -16)
    plt.ylim(0, 4)
    plt.colorbar()

    for Z in [2, 4, 6]:
        plt.figure()
        u_r = models.u[Z] - models.r[Z]
        plt.hist2d(models.r[Z].flatten(), u_r.flatten(),
                   weights=models.N_galaxies[Z].flatten(),
                   range=[[-24, -16], [.5, 3.5]], bins=30,
                   cmap='gist_earth')
        plt.xlim(-24, -16)
        plt.ylim(0, 4)
        plt.colorbar()

#%% -----------------------------------------------------------------------------        
    plt.rcParams['axes.edgecolor']='white'
    plt.rcParams['axes.facecolor']='black'
    plt.figure(figsize=(10,10))
    
    GOOD_MET = [2, 5, 9]
    plt.subplot(221)
    plt.hist2d(r, u-r,
               range=[[-24, -16], [.5, 3.5]], bins=30,
               cmap='Greys_r')
    plt.xlim(-16, -24)
    plt.ylim(0, 4)
#    plt.colorbar()
    
    color_met = ['blue', 'orange', 'limegreen']
    for i, Z in enumerate(GOOD_MET):
        
        u_r = models.u[Z] - models.r[Z]
        hist_u_r, r_edges, u_r_edges = np.histogram2d(
                models.r[Z].flatten(), 
                u_r.flatten(),
                weights=models.N_galaxies[Z].flatten(),
                range=[[-24, -16], [.5, 3.5]], bins=30)
        levels = [np.max(hist_u_r)*0.1, np.max(hist_u_r)*0.3,
                  np.max(hist_u_r)*0.7, np.max(hist_u_r)*0.95]
        plt.contour(r_edges[0:-1], u_r_edges[0:-1], 
                    hist_u_r.T, colors=color_met[i], 
                    levels=levels,
                    linewidths=2)
        plt.xlim(-16.5, -23.5)
        plt.ylim(0.5, 3.5)
        plt.grid(b=True)
    plt.xlabel(r'r', fontsize=12)
    plt.ylabel(r'u-r', fontsize=12)
    plt.locator_params(axis='x', nbins=5)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True,
                    top=True, labelsize=11, color='white')
# -----------------------------------------------------------------------------        
    ax = plt.subplot(222)
    plt.hist2d(r, g-r,
               range=[[-24, -16], [.1, 1.2]], bins=30,
               cmap='Greys_r')
    plt.xlim(-16, -24)
    plt.ylim(0, 4)
#    plt.colorbar()
    
    for i, Z in enumerate(GOOD_MET):
        
        g_r = models.g[Z] - models.r[Z]
        hist_g_r, r_edges, g_r_edges = np.histogram2d(
                models.r[Z].flatten(), 
                g_r.flatten(),
                weights=models.N_galaxies[Z].flatten(),
                range=[[-24, -16], [.1, 1.2]], bins=30)
        levels = [np.max(hist_g_r)*0.1, np.max(hist_g_r)*0.3,
                  np.max(hist_g_r)*0.7, np.max(hist_g_r)*0.95]
        plt.contour(r_edges[0:-1], g_r_edges[0:-1], 
                    hist_g_r.T, colors=color_met[i], 
                    levels=levels,
                    linewidths=2)
        plt.xlim(-16.5, -23.5)
        plt.ylim(0.1, 1.)
        plt.grid(b=True)
    plt.xlabel(r'r', fontsize=12)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True,
                    top=True, labelsize=11, color='white', labelright=True, labelleft=False)
    ax.yaxis.set_label_position("right")
    plt.ylabel(r'g-r', fontsize=12)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=5)
#        plt.colorbar()
# -----------------------------------------------------------------------------        
        
    alfa_models = 1
    alfa_data = 0.4
    plt.subplot(234)
    plt.hist(u,
             bins=np.linspace(-24, -15.5, 50),
             label='SDSS', histtype='bar', alpha=alfa_data, color='silver')
    for i, Z in enumerate(GOOD_MET):
        plt.hist(models.u[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -16, 50),
                 label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='step', alpha=alfa_models,
                 color=color_met[i])
    plt.ylim(0, 15000)
    plt.xlabel(r'u', fontsize=13)
    plt.ylabel(r'N', fontsize=13)
    plt.grid(b=True)
    plt.legend(facecolor='white', framealpha=1, loc='upper left')
    plt.tick_params(which='both', length=4, direction = 'in',  right = True,
                    top=True, labelsize=11, color='white')
    plt.locator_params(axis='x', nbins=5)
    
    ax2 = plt.subplot(235)
    plt.hist(g,
             bins=np.linspace(-24, -15.5, 50),
             label='SDSS', histtype='bar', alpha=alfa_data, color='silver')
    for i, Z in enumerate(GOOD_MET):
        plt.hist(models.g[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -15.5, 50),
                 label=r'{:.4} $Z_\odot$'.format(models.metallicities[Z]/0.02),
                 histtype='step', alpha=alfa_models,
                 color=color_met[i])
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.ylim(0, 15000)
    plt.xlabel(r'g', fontsize=13)
    plt.grid(b=True)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True,
                    top=True, labelsize=11, color='white')
    plt.locator_params(axis='x', nbins=5)
#    plt.legend()

    ax3 = plt.subplot(236)
    plt.hist(r,
             bins=np.linspace(-24, -15.5, 50),
             label='SDSS', histtype='bar', alpha=alfa_data, color='silver')
    for i, Z in enumerate(GOOD_MET):
        plt.hist(models.r[Z].flatten(),
                 weights=models.N_galaxies[Z].flatten(),
                 bins=np.linspace(-24, -15.5, 50),
                 label=str(models.metallicities[Z]),
                 histtype='step', alpha=alfa_models,
                 color=color_met[i])            
        
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.ylim(0, 15000)
    plt.xlabel(r'r', fontsize=13)
    plt.grid(b=True)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True,
                    top=True, labelsize=11, color='white')
    plt.locator_params(axis='x', nbins=5)
plt.subplots_adjust(hspace=0.15, wspace=0.05)

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.facecolor']='white'
#    plt.legend()       
# =============================================================================
# ... Paranoy@ rulz!

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:49:52 2019

@author: pablo
"""


import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


from galaxy_distrib_model import Model_grid
from utilities import Statistics as stat
from utilities import Models as SFH


"""

THIS CODE PROVIDES NEXT FIGURES:

    - COLOR-COLOR DIAGRAM WITH MODELS OVERPLOTED, COL-MAG DIAG WITH
    RUNNING PERCENTILES.
    - TAU-MASS DISTRIBUTION IN METALLICITY BINS WITH RUN-PERCENTS.
    - sSFR-MASS DISTRIBUTION IN METALLICITY BINS WITH RUN-PERCENTS.

For the computation of running percentiles, the code takes ~15-20 minutes to
complete all the computations.

"""

u, u_abs, g, g_abs, r, r_abs, SDSS_V_max = np.loadtxt('SDSS/photometry.dat',
                                     usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)

models = Model_grid()
GOOD_MET = [5, 10, 15, 20]
color_met = ['blue', 'limegreen', 'orange', 'red']


u_r = u_abs-r_abs
g_r = g_abs-r_abs


#%% =============================================================================
# Tau assignment for each galaxy in the SDSS sample
# =============================================================================

u_r_lim =[1.2, 3.5]
g_r_lim =[0.2, 1.1]


tau_u_r_met_bins = []
tau_g_r_met_bins = []

mass_met_bins_u_r = []
mass_met_bins_g_r = []

M_inf_met_bins_u_r = []
M_inf_met_bins_g_r = []

U = models.u[GOOD_MET, :, 0]
G = models.g[GOOD_MET, :, 0]
R = models.r[GOOD_MET, :, 0]

u_r_model = U-R
g_r_model = G-R

tau = models.tau

mass_model = models.M_star[0]

for i, Z in enumerate(GOOD_MET):

    """TAU"""
    # Using u-r ----------------------------------------------
    tau_u_r_lim = [np.interp(u_r_lim[1], u_r_model[i], tau),
                   np.interp(u_r_lim[0], u_r_model[i], tau)]

    tau_u_r_p = np.interp(u_r, u_r_model[i], tau)

    tau_u_r_p[tau_u_r_p<tau_u_r_lim[0]] = np.nan
    tau_u_r_p[tau_u_r_p>tau_u_r_lim[1]] = np.nan

    tau_u_r_met_bins.append(tau_u_r_p)

    # Using g-r ----------------------------------------------
    tau_g_r_lim = [np.interp(g_r_lim[1], g_r_model[i], tau),
                   np.interp(g_r_lim[0], g_r_model[i], tau)]

    tau_g_r_p = np.interp(g_r, g_r_model[i], tau)
    tau_g_r_p[tau_g_r_p<tau_g_r_lim[0]] = np.nan
    tau_g_r_p[tau_g_r_p>tau_g_r_lim[1]] = np.nan

    tau_g_r_met_bins.append(tau_g_r_p)

    """MASS"""

    # Using u-r ----------------------------------------------
    r_mag_p = np.interp(u_r, u_r_model[i], R[i], left=999, right=999)
    r_mag_p[r_mag_p==999] = np.nan

    mass_r_SDSS_u_r_p = mass_model*pow(10, -0.4*(r_abs-r_mag_p))

    mass_met_bins_u_r.append(mass_r_SDSS_u_r_p)


    # Using g-r ----------------------------------------------
    r_mag_p = np.interp(g_r, g_r_model[i], R[i], left=999, right=999)
    r_mag_p[r_mag_p==999] = np.nan

    mass_r_SDSS_g_r_p = mass_model*pow(10, -0.4*(r_abs-r_mag_p))

    mass_met_bins_g_r.append(mass_r_SDSS_g_r_p)

    m_inf_u_r_p = SFH.M_inf(tau_u_r_p, mass_r_SDSS_u_r_p)
    M_inf_met_bins_u_r.append(m_inf_u_r_p)

    m_inf_g_r_p = SFH.M_inf(tau_g_r_p, mass_r_SDSS_g_r_p)
    M_inf_met_bins_g_r.append(m_inf_g_r_p)

M_inf_u_r = np.array(M_inf_met_bins_u_r)
M_inf_g_r = np.array(M_inf_met_bins_g_r)

mass_met_bins_u_r = np.array(mass_met_bins_u_r)
logmass_met_u_r = np.log10(mass_met_bins_u_r)

mass_met_bins_g_r = np.array(mass_met_bins_g_r)
logmass_met_g_r = np.log10(mass_met_bins_g_r)

tau_u_r_met_bins = np.array(tau_u_r_met_bins)
tau_g_r_met_bins = np.array(tau_g_r_met_bins)


# %%


def schechter_mass_function(M, Mprime, alpha, phi=1):
    return phi * np.exp(-(M/Mprime))/Mprime * (M/Mprime)**alpha


def schechter_dn_dlog10M(M, Mprime, alpha, phi=1):
    return phi*np.log(10) * np.exp(-(M/Mprime)) * (M/Mprime)**(alpha+1)


def GAMA_dn_dlog10M(M):
    return(schechter_dn_dlog10M(M, 10**10.66, -0.35, 3.96e-3) +
           schechter_dn_dlog10M(M, 10**10.66, -1.47, 0.79e-3))


mass_vector = np.logspace(12, 8)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'white'
plt.figure(figsize=(8, 6))
#total_galaxy_density = np.sum(SDSS_V_max)
for i, Z in enumerate(GOOD_MET):
#    ax1 = plt.subplot(1, 1, 1)

    hh_u_r, logmass_bins = np.histogram(logmass_met_u_r[i], range=[8, 12],
                                        weights=SDSS_V_max, bins=30,
                                        density=True
                                        )
    logmass_bins = (logmass_bins[1:]+logmass_bins[:-1])/2
    total_galaxy_density = np.sum(SDSS_V_max[~np.isnan(logmass_met_u_r[i])])
    plt.plot(logmass_bins, hh_u_r*total_galaxy_density,
             color=color_met[i],
             label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02))

    hh_g_r, logmass_bins = np.histogram(logmass_met_g_r[i], range=[8, 12],
                                        weights=SDSS_V_max, bins=90,
                                        density=True
                                        )
    logmass_bins = (logmass_bins[1:]+logmass_bins[:-1])/2
    total_galaxy_density = np.sum(SDSS_V_max[~np.isnan(logmass_met_g_r[i])])
    plt.plot(logmass_bins, hh_g_r*total_galaxy_density,
             '--', color=color_met[i],
             label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02))

    plt.plot(np.log10(mass_vector), schechter_dn_dlog10M(mass_vector,
             10**(11), alpha=-1.55, phi=.01), '--k')
    plt.plot(np.log10(mass_vector), GAMA_dn_dlog10M(mass_vector/.7)/.7**3, '-k')

    plt.ylim(3e-5, 3)
    plt.yscale('log')
    plt.ylabel(r'$\frac{dN}{d\log(M_*)}$', fontsize=17)
    plt.xlabel(r'$\log(M/M_\odot)$', fontsize=15)
    plt.legend()
    plt.tick_params(which='both', length=4, direction='in',
                    right=True, top=True, labelsize=13, color='black')
    plt.grid(b=True)

plt.savefig('mass_distrib_schechter.png')


#

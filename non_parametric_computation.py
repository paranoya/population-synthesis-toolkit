#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:28:29 2019

@author: pablo
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from galaxy_distrib_model import Model_grid
from utilities import Statistics as stat
from utilities import Models as SFH
import utilities as ut

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
5
models = Model_grid()
GOOD_MET = [5, 10, 15, 20]
color_met = ['blue', 'limegreen', 'orange', 'red']


u_r = u_abs-r_abs
g_r = g_abs-r_abs

# =============================================================================
# PERCENTILES (WARNING: VERY SLOW PROCESS)
# =============================================================================

#u_r_p10, u_r_stdp10 = stat.Running_weighted_percentile_2d(u_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.1)
#u_r_p90, u_r_stdp90 = stat.Running_weighted_percentile_2d(u_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.9)
#u_r_p50, u_r_stdp50 = stat.Running_weighted_percentile_2d(u_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.5)
#
#g_r_p10, g_r_stdp10 = stat.Running_weighted_percentile_2d(g_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.1)
#g_r_p90, g_r_stdp90 = stat.Running_weighted_percentile_2d(g_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.9)
#g_r_p50, g_r_stdp50 = stat.Running_weighted_percentile_2d(g_r, r_abs, 
#                                                          SDSS_V_max, 5000,  0.5)


#%% =============================================================================
#   color-magnitude diagrams // color-color diagrama + percentiles
# =============================================================================

hist_colors, g_r_edges, u_r_edges = np.histogram2d(g_r, u_r, bins=40, 
                                                   weights=SDSS_V_max, 
                                                   range=[[0., 1.1],[0.8, 3]])

line_pattern = ['o-','-.v','s--','D:']


plt.rcParams['axes.edgecolor']='white'
plt.rcParams['axes.facecolor']='black'
plt.figure(figsize=(10,7))

lev  = [np.max(hist_colors)*0.01, np.max(hist_colors)*0.1, np.max(hist_colors)*0.2, 
        np.max(hist_colors)*0.3, np.max(hist_colors)*0.4, np.max(hist_colors)*0.5,
        np.max(hist_colors)*0.6, np.max(hist_colors)*0.7, np.max(hist_colors)*0.9,
        np.max(hist_colors)*1]
plt.subplot(131)

plt.contourf(g_r_edges[0:-1], u_r_edges[0:-1], hist_colors.T,
                 cmap='Greys_r', levels=10)
for i, Z in enumerate(GOOD_MET):
    u_r_mod = models.u[Z, :, 0] - models.r[Z, :, 0]
    g_r_mod = models.g[Z, :, 0] - models.r[Z, :, 0]
    l = plt.plot(g_r_mod, u_r_mod, line_pattern[i], markevery=15, color='k',                 
             label=r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02))
    plt.setp(l, 'markerfacecolor', color_met[i])
    
    plt.legend(facecolor='white', framealpha=1, loc='upper left')
plt.ylim(0.8, 2.9)    
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.grid(b=True)
plt.xlabel(r'$g-r$')
plt.ylabel(r'$u-r$')
plt.subplot(132)

hist_colmag, r_edges, u_r_edges = np.histogram2d(r_abs, u_r, bins=40, 
                                                   weights=SDSS_V_max, 
                                                   range=[[-22, -15.5],[0.8, 3]])

plt.contourf(r_edges[0:-1], u_r_edges[0:-1], hist_colmag.T,
                 cmap='Greys_r', levels=20)
plt.plot(np.sort(r_abs), u_r_p10, '-b')
plt.plot(np.sort(r_abs), u_r_p90, '-r')
plt.plot(np.sort(r_abs), u_r_p50, '-g')

plt.xlim(-15.5, -22)    
plt.ylim(0.8, 2.7)    
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.grid(b=True)
plt.xlabel(r'$M_r$')
plt.ylabel(r'$u-r$')
plt.subplot(133)

hist_colmag, r_edges, g_r_edges = np.histogram2d(r_abs, g_r, bins=40, 
                                                   weights=SDSS_V_max, 
                                                   range=[[-22, -15.5],[0., 1.1]])

plt.contourf(r_edges[0:-1], g_r_edges[0:-1], hist_colmag.T,
                 cmap='Greys_r', levels=20)
plt.plot(np.sort(r_abs), g_r_p10, '-b')
plt.plot(np.sort(r_abs), g_r_p90, '-r')
plt.plot(np.sort(r_abs), g_r_p50, '-g')
plt.xlim(-15.5, -22)    
plt.ylim(0.1, .9)    
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.grid(b=True)
plt.xlabel(r'$M_r$')
plt.ylabel(r'$g-r$')

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.facecolor']='white'

plt.subplots_adjust(hspace=0.0, wspace=0.27)
plt.savefig('colmag_colcol_diagrams.png')


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

ssfr_u_r_met_bins = []
ssfr_g_r_met_bins = []

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
    
    """SSFR"""
    
    ssfr_u_r_met_bins.append( SFH.ssfr(np.array(tau_u_r_p)*1e9) )
    ssfr_g_r_met_bins.append( SFH.ssfr(np.array(tau_g_r_p)*1e9))


ssfr_u_r_met = np.array(ssfr_u_r_met_bins)
ssfr_g_r_met = np.array(ssfr_g_r_met_bins)

M_inf_u_r = np.array(M_inf_met_bins_u_r)
M_inf_g_r = np.array(M_inf_met_bins_g_r)

mass_met_bins_u_r = np.array(mass_met_bins_u_r)
logmass_met_u_r = np.log10(mass_met_bins_u_r)

mass_met_bins_g_r = np.array(mass_met_bins_g_r)
logmass_met_g_r = np.log10(mass_met_bins_g_r)

tau_u_r_met_bins = np.array(tau_u_r_met_bins)
tau_g_r_met_bins = np.array(tau_g_r_met_bins)

#%% =============================================================================
# PERCENTILES (WARNING: VERY SLOW PROCESS)
# =============================================================================
#tau_p10_u_r = []
#tau_p90_u_r = []
#tau_p50_u_r = []
#
#tau_p10_g_r = []
#tau_p90_g_r = []
#tau_p50_g_r = []
#
#for i in range(len(tau_u_r_met_bins)):
#    tau_p10_u_r_ith, u_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_u_r_met_bins[i], mass_met_bins_u_r[i], 
#            SDSS_V_max, 5000,  0.1)
#    tau_p90_u_r_ith, u_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_u_r_met_bins[i], mass_met_bins_u_r[i], 
#            SDSS_V_max, 5000,  0.9)
#    tau_p50_u_r_ith, u_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_u_r_met_bins[i], mass_met_bins_u_r[i], 
#            SDSS_V_max, 5000,  0.5)
#    
#    tau_p10_g_r_ith, g_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_g_r_met_bins[i], mass_met_bins_g_r[i], 
#            SDSS_V_max, 5000,  0.1)
#    tau_p90_g_r_ith, g_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_g_r_met_bins[i], mass_met_bins_g_r[i], 
#            SDSS_V_max, 5000,  0.9)
#    tau_p50_g_r_ith, g_r_stdp10 = stat.Running_weighted_percentile_2d(
#            tau_g_r_met_bins[i], mass_met_bins_g_r[i], 
#            SDSS_V_max, 5000,  0.5)
#    
#    tau_p10_u_r.append(tau_p10_u_r_ith)
#    tau_p90_u_r.append(tau_p90_u_r_ith)
#    tau_p50_u_r.append(tau_p50_u_r_ith)
#    
#    tau_p10_g_r.append(tau_p10_g_r_ith)
#    tau_p90_g_r.append(tau_p90_g_r_ith)
#    tau_p50_g_r.append(tau_p50_g_r_ith)
#    

#%% =============================================================================
# Plot of tau-M distributions // ssfr-M distributions
# =============================================================================
plt.rcParams['axes.edgecolor']='white'
plt.rcParams['axes.facecolor']='black'
plt.figure(figsize=(12,10))
for i, Z in enumerate(GOOD_MET):
    ax1 = plt.subplot(2,4, i+1)
    hist_tau, logmass_edges, tau_edges = np.histogram2d(logmass_met_u_r[i],
                                                        tau_u_r_met_bins[i], 
                                                        weights = SDSS_V_max,
                                        bins=40, range=[[8,12],[0.2, 50.3]] )
    plt.contourf(logmass_edges[0:-1], tau_edges[0:-1], hist_tau.T,
                 cmap='Greys_r', levels=20)    
    plt.plot(np.sort(logmass_met_u_r[i])[:90277], tau_p10_u_r[i][:90277], '-r')
    plt.plot(np.sort(logmass_met_u_r[i])[:90277], tau_p90_u_r[i][:90277], '-b')
    u_r_median = np.interp(tau_p50_u_r[i], tau[::-1], u_r_model[i][::-1])
    plt.scatter(np.sort(logmass_met_u_r[i])[:90277], tau_p50_u_r[i][:90277], 
                c=u_r_median[:90277],
                cmap='rainbow', norm=mcolors.Normalize(vmin=1.0, vmax=3.), 
                s=0.5)
    
    plt.annotate(r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02), 
                 xy=(9.8, 30), color='white', weight='bold')
    plt.yscale('log')    
    plt.grid(b=True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='white')    
    if i==0:
        plt.ylabel(r'$\tau$', fontsize=14)
    else:        
        plt.setp(ax1.get_yticklabels(), visible=False)
    if i==3:
        cbar = plt.colorbar()
        cbar.set_label(r'$u-r$')

    ax2 = plt.subplot(2,4, i+5)
    hist_tau, logmass_edges, tau_edges = np.histogram2d(logmass_met_g_r[i],
                                                        tau_g_r_met_bins[i], 
                                                        weights = SDSS_V_max,
                                        bins=40, range=[[8,12],[0.2, 50.3]] )
    plt.contourf(logmass_edges[0:-1], tau_edges[0:-1], hist_tau.T,
                 cmap='Greys_r', levels=20)    
    plt.plot(np.sort(logmass_met_g_r[i])[:90277], tau_p10_g_r[i][:90277], '-r')
    plt.plot(np.sort(logmass_met_g_r[i])[:90277], tau_p90_g_r[i][:90277], '-b')
    g_r_median = np.interp(tau_p50_g_r[i], tau[::-1], g_r_model[i][::-1])
    plt.scatter(np.sort(logmass_met_g_r[i])[:90277], tau_p50_g_r[i][:90277], 
                c=g_r_median[:90277],
                cmap='rainbow', norm=mcolors.Normalize(vmin=.2, vmax=1.), 
                s=0.5)
    plt.yscale('log')
    plt.grid(b=True)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='white')    
    plt.xlabel(r'$\log(M/M_\odot)$ ', fontsize=14)    
    if i==0:
        plt.ylabel(r'$\tau$', fontsize=14)
    else:        
        plt.setp(ax1.get_yticklabels(), visible=False)    
    if i==3:
        cbar = plt.colorbar()
        cbar.set_label(r'$g-r$')
    
#    
#    plt.subplot(4,4, i+9)
#    hist_ssfr, logmass_edges, ssfr_edges = np.histogram2d(logmass_met_g_r[i],
#                                                        ssfr_g_r_met[i], 
#                                                        weights = SDSS_V_max,
#                                        bins=40, range=[[8,12],[0.2, -8]] )
#    plt.contourf(logmass_edges, tau_edges, hist_tau.T,
#                 cmap='Greys_r', levels=20)    
#    plt.plot(np.sort(logmass_met_g_r[i]), tau_p10_g_r[i], '-b')
#    plt.plot(np.sort(logmass_met_g_r[i]), tau_p90_g_r[i], '-r')
#    g_r_median = np.interp(tau_p50_g_r[i], tau, g_r_model[i])
#    plt.scatter(np.sort(logmass_met_g_r[i]), tau_p50_g_r[i], c=g_r_median,
#                cmap='rainbow', norm=mcolors.Normalize(vmin=.2, vmax=1.))
    
    
plt.subplots_adjust(hspace=0.0, wspace=0.0)    
plt.savefig('tau_mass_distrib_met_bins.png')
plt.close()

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.facecolor']='white'

#%% --------------------------------------------------------------------------
plt.rcParams['axes.edgecolor']='white'
plt.rcParams['axes.facecolor']='black'
plt.figure(figsize=(12,10))
for i, Z in enumerate(GOOD_MET):
    ax1 = plt.subplot(2,4, i+1)
    hist_ssfr, logmass_edges, ssfr_edges = np.histogram2d(logmass_met_u_r[i],
                                                        np.log10(ssfr_u_r_met[i]), 
                                                        weights = SDSS_V_max,
                                        bins=40, range=[[8,12],[-12, -9.3]]) 
    plt.contourf(logmass_edges[0:-1], ssfr_edges[0:-1], hist_ssfr.T,
                 cmap='Greys_r', levels=20)    
    
    u_r_median = np.interp(tau_p50_u_r[i], tau[::-1], u_r_model[i][::-1])
    ssfr_p10 = np.log10(SFH.ssfr(tau_p10_u_r[i]*1e9))
    ssfr_p90 = np.log10(SFH.ssfr(tau_p90_u_r[i]*1e9))
    ssfr_p50 = np.log10(SFH.ssfr(tau_p50_u_r[i]*1e9))
    
    plt.plot(np.sort(logmass_met_u_r[i])[:90277], ssfr_p10[:90277], '-r')
    plt.plot(np.sort(logmass_met_u_r[i])[:90277], ssfr_p90[:90277], '-b')
    
    plt.scatter(np.sort(logmass_met_u_r[i])[:90277], ssfr_p50[:90277], 
                c=u_r_median[:90277],
                cmap='rainbow', norm=mcolors.Normalize(vmin=1.0, vmax=3.), 
                s=0.5)
    
    plt.annotate(r'{:.2} $Z_\odot$'.format(models.metallicities[Z]/0.02), 
                 xy=(9.8, -9.5), color='white', weight='bold')
    
    plt.grid(b=True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='white')    
    if i==0:
        plt.ylabel(r'$\log(sSFR/[M_\odot/yr])$', fontsize=14)
    else:        
        plt.setp(ax1.get_yticklabels(), visible=False)
    if i==3:
        cbar = plt.colorbar()
        cbar.set_label(r'$u-r$')
    plt.ylim(-11.9,-9.3)
    
    ax2 = plt.subplot(2,4, i+5)
    
    hist_ssfr, logmass_edges, ssfr_edges = np.histogram2d(logmass_met_g_r[i],
                                                        np.log10(ssfr_g_r_met[i]), 
                                                        weights = SDSS_V_max,
                                        bins=40, range=[[8,12],[-12, -9.3]]) 
    plt.contourf(logmass_edges[0:-1], ssfr_edges[0:-1], hist_ssfr.T,
                 cmap='Greys_r', levels=20)    
    
    g_r_median = np.interp(tau_p50_g_r[i], tau[::-1], g_r_model[i][::-1])
    ssfr_p10 = np.log10(SFH.ssfr(tau_p10_g_r[i]*1e9))
    ssfr_p90 = np.log10(SFH.ssfr(tau_p90_g_r[i]*1e9))
    ssfr_p50 = np.log10(SFH.ssfr(tau_p50_g_r[i]*1e9))
    
    plt.plot(np.sort(logmass_met_g_r[i])[:90277], ssfr_p10[:90277], '-r')
    plt.plot(np.sort(logmass_met_g_r[i])[:90277], ssfr_p90[:90277], '-b')
    
    plt.scatter(np.sort(logmass_met_g_r[i])[:90277], ssfr_p50[:90277], 
                c=g_r_median[:90277],
                cmap='rainbow', norm=mcolors.Normalize(vmin=.2, vmax=1.), 
                s=0.5)
        
    plt.grid(b=True)
    plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='white')    
    plt.xlabel(r'$\log(M/M_\odot)$ ', fontsize=14)    
    plt.locator_params(axis='x', nbins=4)
    if i==0:
        plt.ylabel(r'$\log(sSFR/[M_\odot/yr])$', fontsize=14)
    else:        
        plt.setp(ax2.get_yticklabels(), visible=False)    
    if i==3:
        cbar = plt.colorbar()
        cbar.set_label(r'$g-r$')
    plt.ylim(-11.9,-9.3)
    
    
    
plt.subplots_adjust(hspace=0.0, wspace=0.0)    
plt.savefig('ssfr_mass_distrib_met_bins.png')
#plt.close()

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.facecolor']='white'

#%%


plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.edgecolor']='black'


mass_lim1 = 10**10
mass_lim2 = 10**(10.3)
mass_lim3 = 10**(10.5)
mass_lims = [0, mass_lim1, mass_lim2, mass_lim3, 10**13]

N_mass_bins = 4


tau_bins = np.logspace(-0, 1.7, 50)
ssfr_bins = np.logspace(-13,-9.8, 50)
u_r_bins = np.linspace(u_r_lim[0], u_r_lim[1], 30)                        
g_r_bins = np.linspace(g_r_lim[0], g_r_lim[1], 30)                        

mass_labels = [r'$\log(M/M_\odot) < 10$', r'$10 <\log(M/M_\odot) < 10.3$', 
               r'$10.3 <\log(M/M_\odot) < 10.5$', r'$\log(M/M_\odot) > 10.5$']
color_met = ['blue', 'limegreen', 'orange', 'red']
"""ANSATZ"""

#popt = [[2.1, 8.0, 2.0, 4.0],
#        [2.2, 4.0, 3.2, 4.0],
#        [2.4, 3.0, 5.0, 4.0],
#        [2.5, 2.0, 7.7, 4.0],
#        ]

popt = [[2.1, 4.5, 2.0, 4.0],
        [2.2, 3.5, 5.5, 4.0],
        [2.4, 2.5, 7.5, 4.0],
        [2.5, 1.7, 11, 4.0],
        ]

#---------------------------------------------------------------------------#
plt.figure(figsize=(12,12))
for i_mass in range(N_mass_bins):
    for j_met in range(4):    
        
        """Mass bins"""
        
        mass_bin_u_r = np.where(
                (mass_met_bins_u_r[j_met]>mass_lims[i_mass]
                )&(
                 mass_met_bins_u_r[j_met]<mass_lims[i_mass+1])
                )[0]
                
        mass_bin_g_r = np.where(
                (mass_met_bins_g_r[j_met]>mass_lims[i_mass]
                )&(
                 mass_met_bins_g_r[j_met]<mass_lims[i_mass+1])
                )[0]
                
        """dp/dtau ----------------------------------------------------------------------"""
                
        tau_u_r_SDSS = tau_u_r_met_bins[j_met][mass_bin_u_r]
        tau_g_r_SDSS = tau_g_r_met_bins[j_met][mass_bin_g_r]
        
        hist_dNdtau_u_r, tau_edges_u_r = np.histogram(tau_u_r_SDSS,
                                                      weights=SDSS_V_max[mass_bin_u_r],
                                                      bins=tau_bins,
                                                      density=True)        
        tau_edges_u_r = (tau_edges_u_r[1:]+tau_edges_u_r[:-1])/2
        
        hist_dNdtau_g_r, tau_edges_g_r = np.histogram(tau_g_r_SDSS,
                                                      weights=SDSS_V_max[mass_bin_g_r],
                                                      bins=tau_bins,
                                                      density=True)
        tau_edges_g_r = (tau_edges_g_r[1:]+tau_edges_g_r[:-1])/2
        
        ax1 = plt.subplot(2,4, i_mass+1)
        plt.annotate(mass_labels[i_mass], (1, 1))
        plt.plot(tau_edges_u_r, hist_dNdtau_u_r, '-', color=color_met[j_met])
        plt.plot(tau_edges_g_r, hist_dNdtau_g_r, ':', color=color_met[j_met])
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(10**(-2.5), 2)
        plt.xlim(0.5, 30)
        plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='black')
#        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.xlabel(r'$\tau$ [Gyrs]', fontsize=14)
        
        if i_mass==0:
            plt.ylabel(r'$\frac{dp}{d\log(\tau)}$', fontsize=14)        
        else:
            plt.setp(ax1.get_yticklabels(), visible=False)
            
        if j_met==2:
            plt.plot(tau_bins, ut.Ansatz.alpha12beta_powlaw(tau_bins, *popt[i_mass]), '--k')
        
#        plt.xscale('log')
#        plt.yscale('log')
        
        """dN/dSSFR ----------------------------------------------------------------------"""
        
        ssfr_u_r_SDSS = ssfr_u_r_met_bins[j_met][mass_bin_u_r]
        ssfr_g_r_SDSS = ssfr_g_r_met_bins[j_met][mass_bin_g_r]
        
        hist_dNdssfr_u_r, ssfr_edges_u_r = np.histogram(ssfr_u_r_SDSS,
                                                        weights=SDSS_V_max[mass_bin_u_r],
                                                        bins=ssfr_bins,
                                                        density=True
                                                      )
        ssfr_edges_u_r = (ssfr_edges_u_r[1:]+ssfr_edges_u_r[:-1])/2
        
        hist_dNdssfr_g_r, ssfr_edges_g_r = np.histogram(ssfr_g_r_SDSS,
                                                        weights=SDSS_V_max[mass_bin_g_r],
                                                        bins=ssfr_bins,
                                                        density=True
                                                      )
        ssfr_edges_g_r = (ssfr_edges_g_r[1:]+ssfr_edges_g_r[:-1])/2
        
        ax2 = plt.subplot(2,4, i_mass+5)
        
        if i_mass==0:
            plt.ylabel(r'$\frac{dN}{d\log(sSFR)}$', fontsize=14)
        else:
            plt.setp(ax2.get_yticklabels(), visible=False)
            
        if j_met==2:
            ssfr_u_r_interp = np.interp(tau_edges_u_r, tau[::-1], SFH.ssfr(tau[::-1]*1e9)) 
    
            dtau_dssfr_u_r = np.interp(tau_edges_u_r, tau_edges_u_r[0:-1], 
                                     np.abs(np.diff(tau_edges_u_r)/np.diff(ssfr_u_r_interp))) 
            dN_dtau = ut.Ansatz.alpha12beta_powlaw(tau_edges_u_r, *popt[i_mass])
            
            ssfr_g_r_interp = np.interp(tau_edges_g_r, tau[::-1], SFH.ssfr(tau[::-1]*1e9)) 
    
            dtau_dssfr_g_r = np.interp(tau_edges_g_r, tau_edges_g_r[0:-1], 
                                     np.abs(np.diff(tau_edges_g_r)/np.diff(ssfr_g_r_interp))) 
            dN_dtau = ut.Ansatz.alpha12beta_powlaw(tau_edges_g_r, *popt[i_mass])
            
            plt.plot(ssfr_u_r_interp, dN_dtau*dtau_dssfr_u_r, color='k', ls='--')
            plt.plot(ssfr_g_r_interp, dN_dtau*dtau_dssfr_g_r, color='k', ls='--')
            
        plt.plot(ssfr_edges_u_r, hist_dNdssfr_u_r, '-', color=color_met[j_met])
        plt.plot(ssfr_edges_g_r, hist_dNdssfr_g_r, ':', color=color_met[j_met])
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e9, 10**(12.5))   
        plt.xlabel(r'sSFR [yr$^{-1}$]', fontsize=14)
        plt.tick_params(which='both', length=4, direction = 'in',  right = True, top=True, labelsize=13, color='black')
plt.subplots_adjust(hspace=0.2, wspace=0.0)

plt.savefig('tau_ssfr_ansatz_panels.png')
plt.show()
plt.close()

















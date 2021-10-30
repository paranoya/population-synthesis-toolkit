
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:34:22 2019

@author: pablo
"""

import numpy as np
from astropy import units

import wquantiles as wq


class Models():

    def sfr(t, tau, M_inf):
        return M_inf/tau * t/tau * np.exp(-t/tau)

    def stellar_mass(tau, M_inf, t=13.7*1e9):
        return M_inf * (1 - np.exp(-t/tau)*(tau+t)/tau)

    def M_inf(tau, M_star, t=13.7*1e9):
        return M_star / (1 - np.exp(-t/tau)*(tau+t)/tau)

    def ssfr(tau, t=13.7*1e9):
        return t/tau**2 * np.exp(-t/tau) / (1 - np.exp(-t/tau)*(tau+t)/tau)

    def birth_rate_param_b(tau, t=13.7*units.Gyr):
        return t**2/(tau**2*np.exp(t/tau)-(t+tau)*tau)


class Statistics():
    def RunningMedian_2d(x, map_array, num_windows):
        window_elements = len(map_array)//num_windows
        sorted_map = np.sort(map_array)
        runningmedian = []
        map_central_points= []

        for i in range(0, num_windows):
            if i != num_windows-1:
                window_positions = np.where((map_array>=sorted_map[i*window_elements])&(map_array>=sorted_map[(i+1)*window_elements]))[0]
                runningmedian.append(np.nanmedian(x[window_positions]))
                map_central_points.append(np.nanmean(sorted_map[i*window_elements:(i+1)*window_elements]))
            else:
                window_positions = np.where(map_array>=sorted_map[i*window_elements])[0]
                runningmedian.append(np.nanmedian(x[window_positions]))
                map_central_points.append(np.nanmean(sorted_map[i*window_elements:(i+1)*window_elements]))
        return runningmedian, map_central_points

    def RunningMedian_2d2(x, map_array, window_width):

        sorted_positions = np.argsort(map_array)
        x_ordered = x[sorted_positions]

        runningmedian = []
        standard_deviation = []
        print('--> Generating runningmedian with ', window_width, ' neighbours per element \n')
        for i in range(len(x)):

            if i<window_width//2:
                    runningmedian.append(np.nanmedian(x_ordered[:2*i]))
                    standard_deviation.append(np.nanstd(x_ordered[:2*i]))
            elif i>(len(x)-window_width//2):
                    runningmedian.append(np.nanmedian(x_ordered[2*i-len(x):]))
                    standard_deviation.append(np.nanstd(x_ordered[2*i-len(x):]))
            else:
                runningmedian.append(np.nanmedian(x_ordered[i-window_width//2:i+window_width//2]))
                standard_deviation.append(np.nanstd(x_ordered[i-window_width//2:i+window_width//2]))

        return np.array(runningmedian), np.array(standard_deviation)

    def Runningpercentile_2d(x, map_array, window_width, q_percent):

        sorted_positions = np.argsort(map_array)
        x_ordered = x[sorted_positions]

        runningmedian = []
        standard_deviation = []
        print('--> Generating runningPercentile with ', window_width, ' neighbours per element \n')
        for i in range(len(x)):

            if i<window_width//2:
                    runningmedian.append(np.nanpercentile(x_ordered[:2*i], q_percent))
                    standard_deviation.append(np.nanstd(x_ordered[:2*i]))
            elif i>(len(x)-window_width//2):
                    runningmedian.append(np.nanpercentile(x_ordered[2*i-len(x):], q_percent))
                    standard_deviation.append(np.nanstd(x_ordered[2*i-len(x):]))
            else:
                runningmedian.append(np.nanpercentile(x_ordered[i-window_width//2:i+window_width//2]
                , q_percent))
                standard_deviation.append(np.nanstd(x_ordered[i-window_width//2:i+window_width//2]))

        return np.array(runningmedian), np.array(standard_deviation)

    def Running_weighted_percentile_2d(x, map_array, weights, window_width, q_percent):

        sorted_positions = np.argsort(map_array)
        x_ordered = x[sorted_positions]
        sorted_weights = weights[sorted_positions]

        runningmedian = []
        standard_deviation = []
        print('--> Generating runningPercentile with ', window_width, ' neighbours per element \n')
        for i in range(len(x)):

            if i<window_width//2:
                    runningmedian.append(wq.quantile_1D(x_ordered[:2*i+1], sorted_weights[:2*i+1], q_percent))
                    standard_deviation.append(np.nanstd(x_ordered[:2*i+1]))
            elif i>(len(x)-window_width//2):
                    runningmedian.append(wq.quantile_1D(x_ordered[2*i-len(x)-1:], sorted_weights[2*i-len(x)-1:]
                    , q_percent))
                    standard_deviation.append(np.nanstd(x_ordered[2*i-len(x)-1:]))
            else:
                runningmedian.append(wq.quantile_1D(x_ordered[i-window_width//2:i+window_width//2],
                sorted_weights[i-window_width//2:i+window_width//2],  q_percent))
                standard_deviation.append(np.nanstd(x_ordered[i-window_width//2:i+window_width//2]))

        return np.array(runningmedian), np.array(standard_deviation)

    def Bowley_YuleAsymm_coeff(q_075, q_050, q_025):
        """ Symmetric distributions will present q_075+q_025=2*q_050 """
        A_BY = (q_075+q_025-2*q_050)/(q_075-q_025)
        return A_BY

class Ansatz():


    def alpha12beta_powlaw(tau, tau_0, alpha1, alpha2, beta):

        x = tau/tau_0
        dN_dtau = pow(x, alpha1)/pow(1+x**beta, (alpha1+alpha2)/beta)
        integral_dndtau_dtau = np.trapz(dN_dtau, tau)
        return  dN_dtau/(integral_dndtau_dtau)

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







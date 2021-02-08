#!/usr/bin/python
# coding: utf-8

import numpy as np
import pylab as plt

import basic_properties as SPiCE_galaxy
import chemical_evolution as SPiCE_CEM
import SSP as SPiCE_SSP
import units

#-------------------------------------------------------------------------------
def compute_SED( galaxy, evolution, SSP ):    
#-------------------------------------------------------------------------------
 
  t_i = SSP.log_ages_yr - np.ediff1d( SSP.log_ages_yr, to_begin=0 )/2   
  t_i = np.append( t_i, 12 ) # 1000 Gyr
  t_i = galaxy.today - ( units.yr * np.power(10,t_i) )
  t_i[0] = galaxy.today # TODO: Warning if today > last time in SFR_file?
  t_i.clip( 0, galaxy.today, out=t_i )
  M_i = evolution.integral_SFR( t_i )
#  print(t_i/units.Gyr, M_i/units.Msun)
  M_i = -np.ediff1d( M_i )
  Z_i = evolution.integral_Z_SFR( t_i )
  Z_i = -np.ediff1d( Z_i ) / (M_i+units.kg)
  Z_i.clip( SSP.metallicities[0], SSP.metallicities[-1], out=Z_i ) # to prevent extrapolation
#  print(Z_i)
  SED=[]
  
  for k in [0,1,2]:
      Sed=np.zeros( SSP.wavelength.size ) 
      for i, m in enumerate(M_i):
        #print(t_i[i]/units.Gyr, SSP.log_ages_yr[i],'\t', m/units.Msun, Z_i[i])
        if m>0:
          index_Z_hi = SSP.metallicities.searchsorted( Z_i[i] ).clip( 1, len(SSP.metallicities)-1 )
          weight_Z_hi = np.log( Z_i[i]/SSP.metallicities[index_Z_hi-1] ) / np.log( SSP.metallicities[index_Z_hi]/SSP.metallicities[index_Z_hi-1] ) # log interpolation in Z
          Sed += m * ( SSP.SED[index_Z_hi][i][k]*weight_Z_hi + SSP.SED[index_Z_hi-1][i][k]*(1-weight_Z_hi) ) # Sed(t)  is integrated over the SSP ages   
      SED.append(Sed)
  return SED

  # L_nu(t_obs) = int { SFR(t_obs -t_ssp) * L_nu_ssp(Z(t_obs -t_ssp),t_ssp) dt_ssp  }
  
#-------------------------------------------------------------------------------

plt.switch_backend('AGG') # Default (interactive) backend crashes the server the *second* time (!!!???)

galaxy = SPiCE_galaxy.Point_source(today=13.7, R0=100)
tau = np.logspace(-1, 2, 100)


# =============================================================================
M_gas=1e11      # Initial amount of gas.
# =============================================================================

folderSED='SED_log(M)_'+str(np.log10(M_gas))
# today: Time of observation, in Gyr
# R0: Radius of the wind-blown cavity, in pc

#for IMF_i in ["sal_0.15_100", "fer_0.15_100", "kro_0.15_100", "cha_0.15_100"]:
for IMF_i in ["sal_0.15_100"]:
  SSP = SPiCE_SSP.PopStar(IMF=IMF_i)

  for Z_i in SSP.metallicities:
      
    for tau_i in tau:  # if tau < 1 Gyr -- SSP/Elliptical %%% if 3<tau<10 Gyrs Spiral
      
      CEM = SPiCE_CEM.Exponential_SFR(M_gas=1e11, Z=Z_i, M_inf=1e11, tau=tau_i)
      
      filename = IMF_i + '_Z_'+format(Z_i,'.4f') + '_tau_'+format(tau_i,'.3f')
      print(filename)

      CEM.set_current_state( galaxy )

      t = np.linspace(0,galaxy.today,num=101)
      M = CEM.integral_SFR(t)/units.Msun
#      met_Z = CEM.integral_Z_SFR(t)
      t /= units.Gyr
      if M.max() > 0.:
              
          fig = plt.figure()
          ax = plt.subplot(111)
          ax.set_xlabel(r't [Gyr]')
          ax.set_ylabel(r'M [M$_\odot$]')
#          ax.set_xscale('log')
#          ax.set_yscale('log')
#          ax.set_ylim( [0, 2*M.max()] )
          ax.plot( t, M, figure=fig )
#          ax1=ax.twinx()
#          ax1.plot( t, met_Z, figure=fig )
#          ax1.set_ylabel(r'Z')
#          fig.tight_layout()
          fig.savefig('Results/ExponentialSFR/M_star/Mstar_vs_time_for_Z_for_tau_'+format(tau_i,'.3f')+'.png')
          plt.close(fig)
          with open('Results/ExponentialSFR/M_star/Mstar_vs_time_for_Z_for_tau_'+format(tau_i,'.3f')+'.txt', 'w' ) as f:
              f.write('# t [Gyr] M [Msun]\n')
              for tt, mm in zip(t,M):
                  f.write('{:.4} {:.4}\n'.format(tt,mm))
                
      SED = compute_SED( galaxy, CEM, SSP  )
      
      nu_Lnu_erg_s_stel = SSP.wavelength*SED[0] / (units.erg/units.second)
      nu_Lnu_erg_s_neb = SSP.wavelength*SED[1] / (units.erg/units.second)
      nu_Lnu_erg_s_Tot = SSP.wavelength*SED[2] / (units.erg/units.second)
      
      l_A = SSP.wavelength/units.Angstrom
      nu = units.c/SSP.wavelength

      L_lambda0=SED[0]/(3.82e33*units.erg/units.second/units.Angstrom)
      L_lambda1=SED[1]/(3.82e33*units.erg/units.second/units.Angstrom)
      L_lambda2=SED[2]/(3.82e33*units.erg/units.second/units.Angstrom)
      
      sed = zip( l_A, nu, nu_Lnu_erg_s_stel, nu_Lnu_erg_s_neb , nu_Lnu_erg_s_Tot, L_lambda1, L_lambda1, L_lambda2 )
      
     

      if nu_Lnu_erg_s_stel.max() > 0.:
	       fig = plt.figure()
	       ax = plt.subplot(111)
	       ax.set_xlabel(r'$\lambda$ [$\AA$]')
	       ax.set_ylabel(r'$\nu$ L$_\nu$ [erg/s]')
	       ax.set_xlim( [100, 1e5] )
	       ax.set_ylim( [1e-6*nu_Lnu_erg_s_stel.max(), 2*nu_Lnu_erg_s_stel.max()] )
	       ax.set_title(filename)
	       plt.loglog(l_A, nu_Lnu_erg_s_stel, figure=fig)
           
          
	#plt.show()        
	       fig.savefig('Results/ExponentialSFR/'+ folderSED +'/SED_'+filename+'.png')
	       plt.close(fig)
	       with open( 'Results/ExponentialSFR/'+ folderSED +'/SED_'+filename+'.txt', 'w' ) as f:
	           f.write('# wavelength [AA] ---- Frec [HZ] ---- STELLAR nu*f_nu [erg/s] ---- NEBULAR nu*f_nu [erg/s] ---- Total nu*f_nu [erg/s] ---- f_lambda_stellar[erg/s/AA] ---- f_nu_neb ----f_nu_Tot \n ')
	           for ll, nn, ss, dd, tt, zz, kk, jj in sed:
	               f.write('{:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}\n'.format(ll, nn, ss, dd, tt, zz, kk, jj))
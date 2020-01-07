import numpy as np
import units

#-------------------------------------------------------------------------------
class Galaxy:
#-------------------------------------------------------------------------------
  
  def guess_central_density(self):
    #print self.M_gas/units.Msun, self.M_stars/units.Msun
    best_error = 2
    best_n = 0
    for n_cm in np.logspace(-1,4,num=100):
      n = n_cm*units.cm**-3
      g = 0
      R_out, g = self.compute_Mgas( n )
      error = abs(g-1)/(g+1)
      if error < best_error:
	      best_error = error
	      best_n = n
      #print n/units.cm**-3, g, error
    n = best_n
    R_out, g = self.compute_Mgas( n )
    #print 'n_auto =', n/units.cm**-3, 'g =', g, 'R_out =', R_out/units.kpc
    return n, R_out
  
  def get_log_incident_flux(self, nu_Lnu_erg_s):
    print ('---------------')
    print ('INCIDENT:', nu_Lnu_erg_s, self.A_cm2)
    return np.log10( nu_Lnu_erg_s/self.A_cm2 )
  
  def summary(self):
    blah = '> Summary of Basic properties:\n'
    blah += '  Current time: t = {:.3e} Gyr\n'.format(self.today/units.Gyr)
    blah += '  M_gas = {:.3e} Msun\n'.format(self.M_gas/units.Msun)
    blah += '  M_stars = {:.3e} Msun\n'.format(self.M_stars/units.Msun)
    blah += '  Gas metallicity Z = {:.3e}\n'.format(self.Z)
    return blah

#-------------------------------------------------------------------------------
class Point_source(Galaxy):
#-------------------------------------------------------------------------------
  
  def __init__(self, today, R0):
    self.today = today*units.Gyr
    self.R0 = R0*units.pc
    self.Area = 4*np.pi*self.R0**2
    self.A_cm2 = self.Area/units.cm**2
    self.Area_included = 1*units.Mpc**2
    
  def get_log_column_density(self, n_gas):
    L = ( (3*self.M_gas/units.m_p)/(4*np.pi*n_gas) + self.R0**3 )**(1./3.) - self.R0
    return np.log10( (n_gas*L)/units.cm**-2 )
  
  def set_hydrostatic_cloudy(self, cloudy_input):
    n, R_out = self.guess_central_density()
    cloudy_input.set_radius( r_in=np.log10(self.R0/units.cm), r_out=np.log10(R_out/units.cm) )
    cloudy_input.set_other('gravity spherical')
    cloudy_input.set_other('gravity external {} Msun'.format(self.M_stars/units.Msun))
    return n
  
  def compute_Mgas(self, n_gas, figure=None):
    mu = 0.6
    T_eff = 3e4*units.Kelvin
    r_s = mu*units.m_p*units.G*self.M_gas/(units.k*T_eff)
    rho_s = self.M_gas/(4*np.pi*r_s**3)
    M_s = self.M_stars/self.M_gas
    M_g = 0
    x = self.R0/r_s
    rho_over_rho_s = (n_gas*mu*units.m_p)/rho_s
    rho_end = 1e-3*rho_over_rho_s
    #while M_g<1 and rho_over_rho_s>rho_end:
    while rho_over_rho_s>rho_end:
      dMg_dx = x*x*rho_over_rho_s
      drhorhos_dx = rho_over_rho_s*(M_s+M_g)/(x*x)
      h = .01*rho_over_rho_s/drhorhos_dx
      x += h
      M_g += h*dMg_dx
      rho_over_rho_s -= h*drhorhos_dx
      if figure!=None and M_g<1:
	       figure.plot( np.sqrt((x*r_s-self.R0)/units.pc), (rho_over_rho_s*rho_s)/(mu*units.proton_mass/units.cm**3), 'r+' )
    return (x*r_s, M_g)

#-------------------------------------------------------------------------------
class Extended_object(Galaxy):
#-------------------------------------------------------------------------------
  
  def __init__(self, today, Area):
    self.today = today*units.Gyr
    self.Area = Area*units.kpc**2
    self.A_cm2 = self.Area/units.cm**2
    self.Area_included = self.Area

  def get_log_inicident_flux(self, nu_Lnu_erg_s):
    return np.log10( nu_Lnu_erg_s/self.A_cm2 )
  
  def get_log_column_density(self, n_gas=0):
    return np.log10( self.M_gas/units.m_p/self.A_cm2 )
  
  def set_hydrostatic_cloudy(self, cloudy_input):
    #central_pressure = (0.5*np.pi*units.G)*(self.M_gas/self.Area)*(self.M_gas+self.M_stars)/self.Area
    #central_density = central_pressure/(units.k*1e4*units.Kelvin)
    n, R_out = self.guess_central_density()
    cloudy_input.set_other('gravity plane-parallel')
    cloudy_input.set_other('gravity external {} Msun/pc^2'.format((self.M_stars/self.Area)/(units.Msun/units.pc**2)))
    cloudy_input.set_other( 'stop column density {}'.format(self.get_log_column_density()) )
    return n
    
  def compute_Mgas(self, n_gas, figure=None):
    mu = 0.6
    T_eff = 3e4*units.Kelvin
    S_tot = self.M_gas/self.Area
    z_s = (units.k*T_eff)/(mu*units.m_p*2*np.pi*units.G*S_tot)
    rho_s = 0.5*S_tot/z_s
    s = self.M_stars/self.M_gas
    g = 0
    x = 0
    rho_over_rho_s = n_gas*mu*units.m_p/rho_s
    rho_end = 1e-3*rho_over_rho_s
    while rho_over_rho_s>rho_end:
      dg_dx = rho_over_rho_s
      drhorhos_dx = rho_over_rho_s*(s+g)
      h = .01*rho_over_rho_s/drhorhos_dx
      x += h
      g += h*dg_dx
      rho_over_rho_s -= h*drhorhos_dx
      if figure!=None and g<1:
	      figure.plot( np.sqrt((x*z_s)/units.pc), (rho_over_rho_s*rho_s)/(mu*units.proton_mass/units.cm**3), 'r+' )
    return (x*z_s, g)

#-------------------------------------------------------------------------------

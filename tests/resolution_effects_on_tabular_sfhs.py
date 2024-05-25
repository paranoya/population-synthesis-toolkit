import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar, BaseGM, XSL
from pst import models
from astropy import units as u

ssp = PyPopStar(IMF='KRO')
granada_ssp = BaseGM()
#ssp = PopStar(IMF='cha')
xsl_ssp = XSL(IMF='Kroupa', ISO='P00')
ssp.cut_wavelength(3000, 11000)

print(ssp.metallicities, ssp.ages)
print(granada_ssp.metallicities, granada_ssp.ages)
print(xsl_ssp.metallicities, xsl_ssp.ages)

lbtime = np.geomspace(1e-3, 13.7, 300)
time = 13.7 - lbtime[::-1]
time[-1] = 13.7

lbtime2 = np.geomspace(1e-3, 13.7, 10)
# lbtime2 = np.linspace(0, 1, 15)**2 * 13.7
time2 = 13.7 - lbtime2[::-1]
time2[-1] = 13.7

dummy_t = np.linspace(0, 13.7, 1000) * u.Gyr

tau = 30.0
alpha = -2.0
z_0 = 0.02
t_0 = 7.0

def exponential_sfh(time, tau):
    m =  (1 - np.exp(-time / tau)) 
    m /= m[-1]
    return m* u.Msun

def z_star(time, alpha, z_0, t_0):
    z = z_0 * (1 - np.power((time + t_0)/ t_0, alpha))
    return z * u.dimensionless_unscaled

m1 = exponential_sfh(time, tau)
z1 = np.ones_like(m1.value) * 0.02 * u.dimensionless_unscaled
z1 = z_star(time, alpha, z_0, t_0)

# time2 = np.interp(np.array([0, 0.5, 0.9, 0.99, 1.0]) * u.Msun,
#                   exponential_sfh(dummy_t.to_value('Gyr'), tau), dummy_t).to_value('Gyr')

m2 = exponential_sfh(time2, tau)
z2 = np.ones_like(m2) * 0.02 * u.dimensionless_unscaled
z2 = z_star(time2, alpha, z_0, t_0)

plt.figure()
plt.subplot(211)
plt.plot(time, m1, '-+')
plt.plot(time2, m2, '-o')
plt.subplot(212)
plt.plot(time, z1, '-+')
plt.plot(time2, z2, '-o')
plt.show()

model1 = models.Tabular_MFH(times=time * u.Gyr, masses=m1, Z=z1)

model2 = models.Tabular_MFH(times=time2 * u.Gyr, masses=m2, Z=z2)

plt.figure()
plt.subplot(311)
plt.plot(dummy_t, model1.integral_SFR(dummy_t))
plt.plot(dummy_t, model2.integral_SFR(dummy_t))
plt.subplot(312)
plt.plot(dummy_t, model1.integral_Z_SFR(dummy_t))
plt.plot(dummy_t, model2.integral_Z_SFR(dummy_t))
plt.subplot(313)
plt.plot(dummy_t, model1.integral_Z_SFR(dummy_t) / model1.integral_SFR(dummy_t))
plt.plot(dummy_t, model2.integral_Z_SFR(dummy_t)/ model2.integral_SFR(dummy_t))
plt.show()

sed1 = model1.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
granada_sed1 = model1.compute_SED(granada_ssp, t_obs=13.7 * u.Gyr, allow_negative=False)
xsl_sed1 = model1.compute_SED(xsl_ssp, t_obs=13.7 * u.Gyr, allow_negative=False)

#ssp_sed1 = ssp.compute_SED(time * u.Gyr, m1, z1, t_obs=14.7 * u.Gyr)

sed2 = model2.compute_SED(ssp, t_obs=13.7 * u.Gyr, allow_negative=False)

plt.figure()
plt.subplot(211)
plt.plot(ssp.wavelength, sed1, alpha=0.5)
plt.plot(ssp.wavelength, sed2, alpha=0.5)
plt.plot(granada_ssp.wavelength, granada_sed1, alpha=0.5)
plt.plot(xsl_ssp.wavelength, xsl_sed1, alpha=0.5)
plt.yscale('log')
#plt.plot(ssp.wavelength, ssp_sed1)

plt.subplot(212)
print("MEDIAN OFFSET: ", np.nanmedian(sed1 / sed2))
plt.plot(ssp.wavelength, sed1 / sed2)
plt.show()
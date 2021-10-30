# =============================================================================
#  Length
# =============================================================================
meter = 1.
m = meter
km = 1e3 * meter
cm = 0.01 * meter
micron = 1e-6 * meter
nm=1e-9 * meter
Angstrom = 1e-10 * meter
pc = 3.08568025e16 * meter
kpc = 1e3 * pc
Mpc = 1e6 * pc
ly=0.306*pc

# =============================================================================
# Mass
# =============================================================================
kg = 1.
g = 1e-3 * kg
m_p = 1.67262158e-27 * kg
proton_mass = m_p
Msun = 1.98892e30 * kg
Msun_cgs = 1.98892e30 * kg/g

# =============================================================================
# Time
# =============================================================================
second = 1.
s = second
yr = 31556926. * second
Myr = 1e6 * yr
Gyr = 1e9 * yr

Hz = 1./second
MHz = 1e6 * Hz
GHz = 1e9 * Hz
# =============================================================================
# Energy
# =============================================================================
Joule = kg *(m/s)**2
J = Joule
erg = 1e-7 * Joule
eV = 1.60217646e-19 * Joule
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
Rydberg = 13.60569 * eV
Ry = Rydberg

Jansky = 1e-26 * kg /second/second
Jy = Jansky
mJy = 1e-3 * Jansky
L_sun= 3.82e33   #Solar luminosity [erg/s]


Kelvin = 1
N_A =  6.022140  * 1e23  # part/mol
G = 6.67300e-11 * m**3/(kg*s**2)
h_Planck = 6.626068e-34 * m**2 *kg/s
h_Planck_cgs = 6.626070150*1e-27 #[erg/s]

k_Boltzmann = 1.3806503e-23 * Joule/Kelvin
k_cgs = 1.3806488*1e-16
speed_of_light = 299792458 * m/s
h = h_Planck
h_cgs = h_Planck_cgs
k = k_Boltzmann
c = speed_of_light
c_cgs = c*100

import unittest
import numpy as np
from astropy import units as u

from pst.galaxy import GalaxySED
from pst.sed import SedComponent


class _DummySSP:
    def __init__(self):
        self.wavelength = np.geomspace(1000, 10000, 64) * u.AA

    def interpolate_sed(self, target_wavelength):
        self.wavelength = target_wavelength


class _DummySFH:
    def __init__(self):
        self.today = None


class _DummyStellar(SedComponent):
    default_unit = u.Lsun / u.AA
    name = "dummy_stellar"

    def __init__(self):
        self.ssp = _DummySSP()
        self.sfh = _DummySFH()

    def emission_spectrum(self, wavelength: u.Quantity, **params):
        return np.ones(wavelength.size) * self.default_unit


class _DummyCosmology:
    def age(self, z):
        zq = z if isinstance(z, u.Quantity) else z * u.dimensionless_unscaled
        return (13.7 - 0.5 * np.asarray(zq.value)) * u.Gyr

    def luminosity_distance(self, z):
        zq = z if isinstance(z, u.Quantity) else z * u.dimensionless_unscaled
        return (1000.0 + 100.0 * np.asarray(zq.value)) * u.Mpc


class TestGalaxyValidation(unittest.TestCase):
    def setUp(self):
        self.stars = _DummyStellar()
        self.cosmo = _DummyCosmology()

    def test_negative_redshift_rejected(self):
        with self.assertRaises(ValueError):
            GalaxySED(stellar_model=self.stars, cosmology=self.cosmo, redshift=-0.1)

    def test_non_finite_redshift_rejected(self):
        with self.assertRaises(ValueError):
            GalaxySED(stellar_model=self.stars, cosmology=self.cosmo, redshift=np.nan)

    def test_redshift_updates_derived_state(self):
        g = GalaxySED(stellar_model=self.stars, cosmology=self.cosmo, redshift=0.2)
        self.assertTrue(np.isfinite(g.distance_factor.to_value(u.cm**2)))
        self.assertTrue(u.isclose(self.stars.sfh.today, g.t_obs))

        g.redshift = 0.5
        self.assertTrue(np.isfinite(g.distance_factor.to_value(u.cm**2)))
        self.assertTrue(u.isclose(self.stars.sfh.today, g.t_obs))


if __name__ == "__main__":
    unittest.main()

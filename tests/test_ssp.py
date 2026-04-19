import unittest
import numpy as np
from astropy import units as u

from pst import SSP
from pst.observables import TopHatFilter


def make_dummy_ssp():
    ssp = SSP.SSPBase()
    ssp.name = "toy_ssp"
    ssp.isochrone = "toy_isochrone"
    ssp.stellar_library = "toy_library"
    ssp.imf = "toy_imf"
    ssp.ages = np.array([1e6, 1e7]) * u.yr
    ssp.metallicities = np.array([0.01, 0.02]) << u.dimensionless_unscaled
    ssp.wavelength = np.array([100, 200, 400, 800, 1600]) << u.AA
    ssp.L_lambda = np.ones((2, 2, 5)) << (u.Lsun / u.AA / u.Msun)
    return ssp

class TestSSP(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting SSP model")
        self.ssp_model = SSP.PopStar(IMF='cha')
    
    def test_get_ssp_logedges(self):
        self.assertIsNotNone(self.ssp_model.get_ssp_logedges())

    def test_cut_wavelength(self):
        self.ssp_model.cut_wavelength(5000, 9000)
    
    def test_interpolate_sed(self):
        new_wl = np.arange(5000, 8000, 5.5)
        self.ssp_model.interpolate_sed(new_wl)
        self.assertEqual(self.ssp_model.wavelength.size, new_wl.size)
        self.assertEqual(self.ssp_model.L_lambda.shape[-1], new_wl.size)

    def test_regrid(self):
        new_ages = np.array([0.5, 1.0, 5.0])
        new_metallicities = np.array([0.01, 0.02])
        copy_ssp = self.ssp_model.copy()
        copy_ssp.regrid(new_ages, new_metallicities)

    def test_get_mass_lum_ratio(self):
        mass_to_lum = self.ssp_model.get_mass_lum_ratio([5000, 5500])
        self.assertTrue(np.isfinite(mass_to_lum).all(),
                        "Mass-to-light ratio is NaN")

    def test_compute_photometry(self):
        filters = [TopHatFilter(central_wave=cw, width=w,
                                wavelength=self.ssp_model.wavelength
                                ) for cw, w in zip(
            [2000, 4000, 6000], [10, 100, 1000])]
        photometry = self.ssp_model.compute_photometry(filters, z_obs=0.0)
        self.assertTrue(np.isfinite(photometry).all(), "SSP photometry is NaN")

    def test_metadata_and_size_properties(self):
        ssp = make_dummy_ssp()
        self.assertEqual(ssp.name, "toy_ssp")
        self.assertEqual(ssp.isochrone, "toy_isochrone")
        self.assertEqual(ssp.stellar_library, "toy_library")
        self.assertEqual(ssp.imf, "toy_imf")
        self.assertEqual(ssp.n_ages, 2)
        self.assertEqual(ssp.n_metallicities, 2)
        self.assertEqual(ssp.n_ssps, 4)
        self.assertEqual(ssp.n_wavelengths, 5)

    def test_current_mass_and_copy_are_independent(self):
        ssp = make_dummy_ssp()
        ssp.returned_mass_frac = np.array([[0.1, 0.2], [0.3, 0.4]])

        self.assertTrue(np.allclose(ssp.current_mass, np.array([[0.9, 0.8], [0.7, 0.6]])))

        ssp_copy = ssp.copy()
        ssp.returned_mass_frac[0, 0] = 0.0

        self.assertTrue(np.allclose(ssp_copy.returned_mass_frac,
                                    np.array([[0.1, 0.2], [0.3, 0.4]])))

    def test_get_specific_mass_lum_ratio(self):
        ssp = make_dummy_ssp()
        ratio = ssp.get_specific_mass_lum_ratio([100, 800] * u.AA)
        self.assertTrue(np.isfinite(ratio).all())

    def test_get_ionising_photon_rate_populates_log_tables(self):
        ssp = make_dummy_ssp()

        hi_rate = ssp.get_ionising_photon_rate(species='HI')
        hei_rate = ssp.get_ionising_photon_rate(species='HeI')
        heii_rate = ssp.get_ionising_photon_rate(species='HeII')

        self.assertEqual(hi_rate.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(hi_rate.value)))
        self.assertTrue(np.all(np.isfinite(hei_rate.value)))
        self.assertTrue(np.all(np.isfinite(heii_rate.value)))
        self.assertIsNotNone(ssp.log_ionising_HI_photons)
        self.assertIsNotNone(ssp.log_ionising_HeI_photons)
        self.assertIsNotNone(ssp.log_ionising_HeII_photons)

        with self.assertRaises(ValueError):
            ssp.get_ionising_photon_rate(species='CIV')

if __name__ == '__main__':
    unittest.main()
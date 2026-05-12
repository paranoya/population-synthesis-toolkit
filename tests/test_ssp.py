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


def make_dummy_cloudy_ssp(include_gas=False):
    ssp = SSP.SSPwithCloudyGasModel()
    ssp.name = "toy_cloudy_ssp"
    ssp.ages = np.array([1e6, 1e7]) * u.yr
    ssp.metallicities = np.array([0.01, 0.02]) << u.dimensionless_unscaled
    ssp.wavelength = np.array([100, 200, 400, 800, 1600]) << u.AA
    ssp.stellar_L_lambda = np.ones((2, 2, 5)) << (u.Lsun / u.AA / u.Msun)

    if include_gas:
        ssp.log_u_array = np.array([-3.0, -1.0]) << u.dex(u.dimensionless_unscaled)
        ssp.gas_L_lambda_grid = np.array([
            np.ones((2, 2, 5)) * 2.0,
            np.ones((2, 2, 5)) * 4.0,
        ]) << (u.Lsun / u.AA / u.Msun)
        ssp.gas_transmission_grid = np.array([
            np.ones((2, 2, 5)) * 0.8,
            np.ones((2, 2, 5)) * 0.6,
        ]) << u.dimensionless_unscaled
        ssp.log_u = -2.0 << u.dex(u.dimensionless_unscaled)

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

    def test_cloudy_mixin_cut_wavelength_without_gas_keeps_sed_consistent(self):
        ssp = make_dummy_cloudy_ssp(include_gas=False)

        ssp.cut_wavelength(150, 900, verbose=False)

        self.assertEqual(ssp.wavelength.size, 3)
        self.assertEqual(ssp.stellar_L_lambda.shape[-1], 3)
        self.assertEqual(ssp.L_lambda.shape[-1], 3)

    def test_cloudy_mixin_regrid_without_gas_keeps_sed_consistent(self):
        ssp = make_dummy_cloudy_ssp(include_gas=False)

        ssp.regrid(np.array([0.5, 5.0]), np.array([0.01]), verbose=False)

        self.assertEqual(ssp.ages.size, 2)
        self.assertEqual(ssp.metallicities.size, 1)
        self.assertEqual(ssp.stellar_L_lambda.shape[:2], (1, 2))
        self.assertEqual(ssp.L_lambda.shape[:2], (1, 2))

    def test_cloudy_mixin_log_u_clamps_to_grid_edges(self):
        ssp = make_dummy_cloudy_ssp(include_gas=True)

        ssp.log_u = -4.0 << u.dex(u.dimensionless_unscaled)
        self.assertTrue(np.allclose(
            ssp.gas_L_lambda.value,
            ssp.gas_L_lambda_grid[0].value))
        self.assertTrue(np.allclose(
            ssp.transmission.value,
            ssp.gas_transmission_grid[0].value))

        ssp.log_u = 0.0 << u.dex(u.dimensionless_unscaled)
        self.assertTrue(np.allclose(
            ssp.gas_L_lambda.value,
            ssp.gas_L_lambda_grid[-1].value))
        self.assertTrue(np.allclose(
            ssp.transmission.value,
            ssp.gas_transmission_grid[-1].value))

    def test_cloudy_mixin_interpolate_with_gas_keeps_components_aligned(self):
        ssp = make_dummy_cloudy_ssp(include_gas=True)

        new_wl = np.array([150, 300, 600]) << u.AA
        ssp.interpolate_sed(new_wl, verbose=False)

        self.assertEqual(ssp.wavelength.size, new_wl.size)
        self.assertEqual(ssp.stellar_L_lambda.shape[-1], new_wl.size)
        self.assertEqual(ssp.gas_L_lambda_grid.shape[-1], new_wl.size)
        self.assertEqual(ssp.gas_transmission_grid.shape[-1], new_wl.size)
        self.assertEqual(ssp.gas_L_lambda.shape[-1], new_wl.size)
        self.assertEqual(ssp.transmission.shape[-1], new_wl.size)
        self.assertEqual(ssp.L_lambda.shape[-1], new_wl.size)

    def test_cloudy_mixin_rejects_unsorted_log_u_array(self):
        ssp = make_dummy_cloudy_ssp(include_gas=False)

        with self.assertRaises(ValueError):
            ssp.log_u_array = np.array([-1.0, -3.0]) << u.dex(u.dimensionless_unscaled)

    def test_cloudy_mixin_rejects_malformed_gas_cube_shape(self):
        ssp = make_dummy_cloudy_ssp(include_gas=False)
        ssp.log_u_array = np.array([-3.0, -1.0]) << u.dex(u.dimensionless_unscaled)

        with self.assertRaises(ValueError):
            ssp.gas_L_lambda_grid = np.ones((2, 3, 2, 5)) << (u.Lsun / u.AA / u.Msun)

        with self.assertRaises(ValueError):
            ssp.gas_transmission_grid = np.ones((2, 2, 3, 5)) << u.dimensionless_unscaled

    def test_cloudy_mixin_transmission_uses_pointwise_interpolation(self):
        ssp = make_dummy_cloudy_ssp(include_gas=False)
        ssp.log_u_array = np.array([-2.0]) << u.dex(u.dimensionless_unscaled)
        ssp.gas_L_lambda_grid = np.ones((1, 2, 2, 5)) << (u.Lsun / u.AA / u.Msun)
        transmission = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        ssp.gas_transmission_grid = np.broadcast_to(transmission, (1, 2, 2, 5)) << u.dimensionless_unscaled
        ssp.log_u = -2.0 << u.dex(u.dimensionless_unscaled)

        new_wl = np.array([150, 300, 600]) << u.AA
        ssp.interpolate_sed(new_wl, verbose=False)

        expected = np.interp(new_wl.value, np.array([100, 200, 400, 800, 1600]), transmission)
        self.assertTrue(np.allclose(ssp.transmission[0, 0].value, expected))

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
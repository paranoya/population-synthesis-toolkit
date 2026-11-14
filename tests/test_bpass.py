"""Tests for the BPASS native-to-FITS converter and the pst.SSP.BPASS class.

A small mock data set written in the native BPASS format is converted to the
PST FITS format and then loaded with :class:`pst.SSP.BPASS`.
"""
import importlib.util
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from astropy import units as u

from pst.SSP import BPASS, SSPBase
from pst.utils import convert_wavelength_medium

CONVERTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    "ssp_installation", "bpass", "convert_bpass_to_fits.py")


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_bpass_to_fits",
                                                  CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


N_AGES = 51
LOG_AGES = 6.0 + 0.1 * np.arange(N_AGES)
WAVELENGTH = np.arange(1.0, 3001.0)
# Mock models: (population, library, Z tag, alpha tag) -> scale factor
MODELS = {
    ("bin", "C3K", "z020", "a+00"): 3.0,
    ("bin", "C3K", "z002", "a+00"): 2.0,
    ("bin", "C3K", "zem4", "a+00"): 1.0,
    ("bin", "C3K", "z020", "a+04"): 4.0,
    ("sin", "C3K", "z020", "a+00"): 5.0,
}
LSUN_RATIO = 3.848 / 3.828


def native_sed(scale):
    """Mock L_lambda (Lsun / A per 1e6 Msun), shape (n_wave, n_ages)."""
    return scale * 1e3 * np.outer(WAVELENGTH / 1000.0, 10**(-(LOG_AGES - 6)))


def native_starmass(scale):
    return 1e6 * (1 - 0.01 * scale * (LOG_AGES - 6) / 5)


def native_sn_counts(scale, population):
    counts = np.zeros((N_AGES, 8))
    counts[10, 0] = 100 * scale   # IIP
    if population == "bin":
        counts[30, 7] = 10 * scale  # Ia
    return counts


def write_table(path, table):
    np.savetxt(path, table, fmt="%16.7E", delimiter="")


def write_mock_bpass(directory):
    h, c, lsun = 6.62607015e-27, 2.99792458e18, 3.848e33
    for (pop, lib, z_tag, a_tag), scale in MODELS.items():
        suffix = f"-{pop}-imf135_300.{lib}.{z_tag}.{a_tag}.dat"
        sed = native_sed(scale)
        write_table(os.path.join(directory, "spectra" + suffix),
                    np.column_stack([WAVELENGTH, sed]))
        write_table(os.path.join(directory, "starmass" + suffix),
                    np.column_stack([LOG_AGES, native_starmass(scale)]))
        write_table(os.path.join(directory, "supernovae" + suffix),
                    np.column_stack([LOG_AGES, native_sn_counts(scale, pop)]))
        mask = WAVELENGTH <= 912
        log_q = np.log10((sed[mask] * lsun * WAVELENGTH[mask, None]
                          / (h * c)).sum(axis=0))
        ionizing = np.column_stack([LOG_AGES, log_q, log_q - 11.87,
                                    np.zeros(N_AGES), np.zeros(N_AGES)])
        write_table(os.path.join(directory, "ionizing" + suffix), ionizing)


class TestBPASS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.native_dir = os.path.join(cls.tmpdir, "native")
        cls.fits_dir = os.path.join(cls.tmpdir, "fits")
        os.makedirs(cls.native_dir)
        write_mock_bpass(cls.native_dir)
        cls.converter = load_converter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            status = cls.converter.main(
                [cls.native_dir, "--output-dir", cls.fits_dir, "--quiet"])
        assert status == 0
        cls.converter_warnings = [str(w.message) for w in caught]

    def test_converter_consistency_checks(self):
        # The mock ionising tables and SN columns are consistent with the
        # SEDs, so the converter sanity checks must not complain.
        for message in self.converter_warnings:
            self.assertNotIn("log Q(HI)", message)
            self.assertNotIn("column order", message)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_output_files(self):
        self.assertEqual(sorted(os.listdir(self.fits_dir)),
                         ["bpass_v2.3_bin_imf135_300_C3K.fits",
                          "bpass_v2.3_sin_imf135_300_C3K.fits"])

    def test_grid_and_metadata(self):
        ssp = BPASS("c3k", alpha_fe=0.0, path=self.fits_dir,
                    load_properties=True, verbose=False)
        np.testing.assert_allclose(ssp.metallicities.value, [1e-4, 2e-3, 2e-2])
        np.testing.assert_allclose(np.log10(ssp.ages.to_value(u.yr)), LOG_AGES)
        self.assertEqual(ssp.L_lambda.shape, (1, 3, N_AGES, WAVELENGTH.size))
        self.assertEqual(ssp.stellar_library, "C3K")
        self.assertEqual(ssp.imf, "imf135_300")
        np.testing.assert_array_equal(ssp.alpha_fe.value, [0.0])
        self.assertEqual(ssp.n_alpha_fe, 1)
        self.assertIsNone(ssp.available_models)
        self.assertEqual(ssp.name, "BPASS_v2.3_bin_imf135_300_C3K_afe+0.0")

    def test_all_alpha_fe_loaded_by_default(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False)
        np.testing.assert_array_equal(ssp.alpha_fe.value, [0.0, 0.4])
        # Union of the metallicities of both [alpha/Fe] values
        np.testing.assert_allclose(ssp.metallicities.value, [1e-4, 2e-3, 2e-2])
        self.assertEqual(ssp.L_lambda.shape, (2, 3, N_AGES, WAVELENGTH.size))
        np.testing.assert_array_equal(
            ssp.available_models, [[True, True, True], [False, False, True]])
        # Missing models have zero SED
        self.assertTrue(np.all(ssp.L_lambda[1, :2].value == 0))
        self.assertEqual(ssp.name, "BPASS_v2.3_bin_imf135_300_C3K_afe+0.0_+0.4")
        # Selecting both explicitly gives the same grid
        both = BPASS("C3K", alpha_fe=[0.4, 0.0], path=self.fits_dir,
                     verbose=False)
        np.testing.assert_array_equal(both.L_lambda.value, ssp.L_lambda.value)

    def test_weights_skip_missing_models(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False)
        age = 1.0 * u.Gyr
        # [alpha/Fe] = +0.4 only has Z = 0.02: all the mass goes there
        w = ssp.get_weights(age, 2e-3, alpha_fe=0.4).to_value(u.Msun)
        self.assertAlmostEqual(w[1, 2].sum(), 1.0)
        self.assertAlmostEqual(w.sum(), 1.0)
        # Midway in [alpha/Fe]: the alpha/Fe = 0 half follows the full
        # metallicity grid, the +0.4 half is clipped to its only node
        w = ssp.get_weights(age, 2e-3, alpha_fe=0.2).to_value(u.Msun)
        self.assertAlmostEqual(w[0, 1].sum(), 0.5)
        self.assertAlmostEqual(w[1, 2].sum(), 0.5)
        # No mass ever lands on a missing model
        rng = np.random.default_rng(1)
        n = 2000
        w = ssp.get_weights(10**rng.uniform(-3, 1, n) * u.Gyr,
                            10**rng.uniform(-4.5, -1.2, n),
                            rng.uniform(0, 2, n),
                            alpha_fe=rng.uniform(-0.2, 0.6, n)).value
        missing = ~ssp.available_models
        self.assertEqual(np.abs(w[missing]).sum(), 0.0)

    def test_regrid_completes_the_grid(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False)
        new_z = np.array([1e-3, 2e-2])
        ssp.regrid(ssp.ages[::5], new_z, verbose=False)
        self.assertEqual(ssp.L_lambda.shape,
                         (2, 2, ssp.ages.size, WAVELENGTH.size))
        self.assertIsNone(ssp.available_models)
        # At +0.4 both new metallicities map to the only model (Z = 0.02)
        np.testing.assert_allclose(ssp.L_lambda[1, 0].value,
                                   ssp.L_lambda[1, 1].value)

    def test_sed_normalised_per_msun_formed(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False,
                    load_properties=True)
        # [alpha/Fe] = 0, metallicity index 2 is Z=0.020 (scale 3)
        expected = native_sed(3.0).T / 1e6 * LSUN_RATIO
        sed = ssp.L_lambda[0, 2].to_value(u.Lsun / u.AA / u.Msun)
        np.testing.assert_allclose(sed, expected, rtol=1e-6)
        # [alpha/Fe] = +0.4, Z=0.020 (scale 4)
        np.testing.assert_allclose(
            ssp.L_lambda[1, 2].to_value(u.Lsun / u.AA / u.Msun),
            native_sed(4.0).T / 1e6 * LSUN_RATIO, rtol=1e-6)

    def test_alpha_selection(self):
        ssp = BPASS("C3K", alpha_fe=0.4, path=self.fits_dir, verbose=False,
                    load_properties=True)
        np.testing.assert_allclose(ssp.metallicities.value, [0.02])
        np.testing.assert_allclose(
            ssp.L_lambda[0, 0].to_value(u.Lsun / u.AA / u.Msun),
            native_sed(4.0).T / 1e6 * LSUN_RATIO, rtol=1e-6)
        self.assertEqual(ssp.name, "BPASS_v2.3_bin_imf135_300_C3K_afe+0.4")
        with self.assertRaises(ValueError):
            BPASS("C3K", alpha_fe=0.2, path=self.fits_dir, verbose=False,
                  load_properties=True)

    def test_single_star_population(self):
        ssp = BPASS("C3K", binary=False, path=self.fits_dir, verbose=False,
                    load_properties=True)
        self.assertEqual(ssp.metallicities.size, 1)
        self.assertEqual(ssp.n_alpha_fe, 1)
        self.assertTrue(np.all(ssp.supernova_rate_by_type["IA"].value == 0))

    def test_mass_fractions(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False, load_properties=True)
        expected = native_starmass(3.0) / 1e6
        np.testing.assert_allclose(ssp.current_mass[0, 2], expected)
        np.testing.assert_allclose(ssp.returned_mass_frac[0, 2], 1 - expected)
        np.testing.assert_allclose(ssp.current_mass[1, 2],
                                   native_starmass(4.0) / 1e6)
        # Missing models: no mass returned
        self.assertTrue(np.all(ssp.returned_mass_frac[1, :2] == 0))
        self.assertTrue(np.all(ssp.remnant_mass_frac == 0))

    def test_supernova_rates(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False, load_properties=True)
        dt = 10**(LOG_AGES + 0.05) - 10**(LOG_AGES - 0.05)
        dt[0] = 10**6.05
        rate = ssp.supernova_rate.to_value(1 / (u.yr * u.Msun))
        np.testing.assert_allclose(rate[0, 2, 10], 300 / dt[10] / 1e6)
        np.testing.assert_allclose(rate[0, 2, 30], 30 / dt[30] / 1e6)
        np.testing.assert_allclose(
            rate, sum(r.to_value(1 / (u.yr * u.Msun))
                      for r in ssp.supernova_rate_by_type.values()))
        # Integrated number of SNe per Msun formed
        np.testing.assert_allclose(np.sum(rate[0, 2] * dt), 330 / 1e6)
        self.assertTrue(np.all(rate[1, :2] == 0))

    def test_ionising_photons(self):
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False, load_properties=True)
        available = ssp.available_models
        log_q = ssp.log_ionising_HI_photons.to_value(u.dex(u.s**-1 / u.Msun))
        # Same definition as SSPBase.get_ionising_photon_rate, up to the
        # integration scheme (BPASS sums over 1 A bins; PST uses trapezoids)
        computed = ssp.copy().get_ionising_photon_rate("HI")
        with np.errstate(divide="ignore"):
            computed = np.log10(computed.to_value(u.s**-1 / u.Msun))
        np.testing.assert_allclose(log_q[available], computed[available],
                                   atol=0.01)
        # Missing models emit no ionising photons
        self.assertTrue(np.all(np.isneginf(log_q[~available])))
        self.assertTrue(np.all(np.isneginf(computed[~available])))
        self.assertTrue(np.all(
            ssp.log_ionising_HeII_photons.value[available]
            < ssp.log_ionising_HeI_photons.value[available]))
        self.assertTrue(np.all(
            ssp.log_ionising_HeI_photons.value[available]
            < ssp.log_ionising_HI_photons.value[available]))

    def test_wavelength_cut(self):
        out_dir = os.path.join(self.tmpdir, "fits_cut")
        self.converter.main([self.native_dir, "--output-dir", out_dir,
                             "--populations", "bin", "--wl-min", "1000",
                             "--wl-max", "2000", "--quiet"])
        ssp = BPASS("C3K", path=out_dir, verbose=False, load_properties=True)
        self.assertEqual(ssp.wavelength.size, 1001)
        # Ionising photons are computed before the wavelength cut
        full = BPASS("C3K", path=self.fits_dir, verbose=False, load_properties=True)
        np.testing.assert_allclose(ssp.log_ionising_HI_photons.value,
                                   full.log_ionising_HI_photons.value)

    def test_wavelength_medium(self):
        # C3K is tabulated in vacuum and converted to air by default
        native = WAVELENGTH * u.AA
        ssp = BPASS("C3K", path=self.fits_dir, verbose=False)
        np.testing.assert_allclose(
            ssp.wavelength.to_value(u.AA),
            convert_wavelength_medium(native, "vacuum", "air").value)
        SSPBase.wavelength_medium = "vacuum"
        try:
            ssp = BPASS("C3K", path=self.fits_dir, verbose=False)
        finally:
            SSPBase.wavelength_medium = "air"
        np.testing.assert_allclose(ssp.wavelength.to_value(u.AA), WAVELENGTH)

    def test_unknown_library(self):
        with self.assertRaises(NameError):
            BPASS("MILES", path=self.fits_dir, verbose=False)


if __name__ == "__main__":
    unittest.main()

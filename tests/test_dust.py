"""
Unit tests for pst.dust

This test module covers the main public API of the dust module:
- Utility functions (modified_blackbody)
- Attenuation curves (PowerLawAttenuationCurve, ExtinctionLibCurve)
- Attenuation models (DustScreenAttenuation, CharlotFall00Attenuation)
- Dust emission (Casey2012DustComponent)
- Calorimetric coupling (CalorimetricDustComponent)

Notes
-----
These tests aim to be robust and fast. They avoid downloading external data.
They rely on the extinction package being installed.

Run
---
python -m unittest -v test_dust.py
"""

import unittest
import numpy as np
from astropy import units as u
from astropy import constants as const

from pst import dust
from pst import SSP

class TestUtils(unittest.TestCase):
    def test_modified_blackbody_per_wavelength_units_and_finite(self):
        wavelength = np.geomspace(10, 1e4, 500) * u.um
        temperature = 30 * u.K
        beta = 1.5

        mbb = dust.modified_blackbody(
            wavelength, T=temperature, beta=beta, per_freq=False
        )

        self.assertEqual(mbb.unit, u.Unit("erg / (Angstrom s sr cm2)"))
        self.assertTrue(np.isfinite(mbb.value).all())
        self.assertEqual(mbb.shape, wavelength.shape)

    def test_modified_blackbody_per_frequency_finite(self):
        wavelength = np.geomspace(10, 1e4, 200) * u.um
        temperature = 35 * u.K
        beta = 2.0

        mbb = dust.modified_blackbody(
            wavelength, T=temperature, beta=beta, per_freq=True
        )

        self.assertTrue(hasattr(mbb, "unit"))
        self.assertTrue(np.isfinite(mbb.value).all())
        self.assertEqual(mbb.shape, wavelength.shape)

    def test_modified_blackbody_optically_thin_vs_thick_different(self):
        wavelength = np.geomspace(10, 1e4, 200) * u.um
        temperature = 30 * u.K
        beta = 1.5

        thin = dust.modified_blackbody(
            wavelength, T=temperature, beta=beta, lam_0=None, per_freq=False
        )
        thick = dust.modified_blackbody(
            wavelength, T=temperature, beta=beta, lam_0=200 * u.um, per_freq=False
        )

        # Shapes should differ for typical parameters
        self.assertFalse(np.allclose(thin.value, thick.value))

    def test_modified_blackbody_extreme_grid_stays_finite(self):
        wavelength = np.geomspace(1e-6, 1e8, 800) * u.um
        mbb = dust.modified_blackbody(
            wavelength, T=40 * u.K, beta=2.8, lam_0=200 * u.um, per_freq=False
        )
        self.assertTrue(np.isfinite(mbb.value).all())


class TestAttenuationCurves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl = np.geomspace(1000, 30000, 2000) * u.AA

    def test_powerlaw_k_lambda_dimensionless_and_normalized(self):
        curve = dust.PowerLawAttenuationCurve()
        k = curve.k_lambda(self.wl)

        self.assertEqual(k.unit, u.dimensionless_unscaled)
        self.assertEqual(k.shape, self.wl.shape)

        # By definition k(pivot) should be 1 (up to interpolation of grid)
        pivot = curve.pivot.q.to_value(u.AA)
        idx = int(np.argmin(np.abs(self.wl.to_value(u.AA) - pivot)))
        self.assertAlmostEqual(float(k[idx].to_value(u.dimensionless_unscaled)), 1.0, places=3)

    def test_powerlaw_positive_alpha_raises(self):
        with self.assertRaises(ValueError):
            _ = dust.PowerLawAttenuationCurve(alpha=dust.Parameter(0.5, unit=u.dimensionless_unscaled))

    def test_extinction_lib_curve_exists_and_dimensionless(self):
        curve = dust.ExtinctionLibCurve(law="ccm89")
        k = curve.k_lambda(self.wl)

        self.assertEqual(k.unit, u.dimensionless_unscaled)
        self.assertEqual(k.shape, self.wl.shape)
        self.assertTrue(np.isfinite(k.value).all())

    def test_extinction_lib_unknown_law_raises(self):
        with self.assertRaises(ValueError):
            _ = dust.ExtinctionLibCurve(law="definitely_not_a_real_law")

    def test_attenuation_factor_extreme_a_lambda_is_finite(self):
        curve = dust.PowerLawAttenuationCurve()
        f = curve.attenuation_factor(self.wl, a_v=-1e4 * u.mag)

        self.assertEqual(f.unit, u.dimensionless_unscaled)
        self.assertTrue(np.isfinite(f.value).all())


class TestAttenuationModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl = np.geomspace(1000, 30000, 1000) * u.AA
        cls.spec = np.ones(cls.wl.size) * (3631 * u.Jy) * const.c / cls.wl**2

    def test_dust_screen_attenuation_factor_range(self):
        model = dust.DustScreenAttenuation()
        model.a_v.set(1.0 * u.mag)

        f = model.attenuation_factor(self.wl)

        self.assertEqual(f.unit, u.dimensionless_unscaled)
        self.assertEqual(f.shape, self.wl.shape)
        self.assertTrue(np.isfinite(f.value).all())

        # attenuation factor should be between 0 and 1 for A_lambda >= 0
        self.assertTrue(np.all(f.value > 0.0))
        self.assertTrue(np.all(f.value <= 1.0))

    def test_dust_screen_apply_preserves_shape(self):
        model = dust.DustScreenAttenuation()
        model.a_v.set(0.5 * u.mag)

        out = model.apply(self.wl, self.spec, axis=-1)

        self.assertEqual(out.shape, self.spec.shape)
        self.assertTrue(np.isfinite(out.value).all())

    def test_cf00_factor_shape(self):
        model = dust.CharlotFall00Attenuation()
        model.a_v_young.set(1.0 * u.mag)
        model.a_v_old.set(0.2 * u.mag)

        f = model.attenuation_factor(self.wl)

        self.assertEqual(f.unit, u.dimensionless_unscaled)
        self.assertEqual(f.shape, (2, self.wl.size))
        self.assertTrue(np.isfinite(f.value).all())

        # Young should be more attenuated than old at most wavelengths
        self.assertTrue(np.median(f[0].value) <= np.median(f[1].value))


class TestDustEmissionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl = np.geomspace(0.1, 1000, 300) * u.um

    def test_casey2012_lambda_pivot_unit(self):
        model = dust.Casey2012DustComponent()
        lam_piv = model._lambda_pivot(t_dust=35 * u.K, alpha=2.0)
        self.assertEqual(lam_piv.unit, u.um)
        self.assertTrue(np.isfinite(lam_piv.to_value(u.um)))

    def test_casey2012_emission_spectrum_units_and_finite(self):
        model = dust.Casey2012DustComponent()
        spec = model.emission_spectrum(self.wl, lum_ir=1e11 * u.Lsun)

        self.assertEqual(spec.unit, u.Lsun / u.AA)
        self.assertEqual(spec.shape, self.wl.shape)
        self.assertTrue(np.isfinite(spec.value).all())

    def test_casey2012_normalization_scales_with_lum_ir(self):
        model = dust.Casey2012DustComponent()
        s1 = model.emission_spectrum(self.wl, lum_ir=1e10 * u.Lsun)
        s2 = model.emission_spectrum(self.wl, lum_ir=2e10 * u.Lsun)

        # Should scale approximately linearly in normalization
        ratio = np.nanmedian((s2 / s1).to_value(u.dimensionless_unscaled))
        self.assertTrue(np.isfinite(ratio))
        self.assertAlmostEqual(float(ratio), 2.0, places=2)


class _DummyStellarSource(dust.StellarComponent):
    """
    Minimal StellarComponent-like object for calorimetric tests.

    The real StellarComponent expects an SSP and SFH. For unit testing of the
    calorimetric machinery, we only need an object with emission_spectrum.
    """
    def __init__(self, wavelength: u.Quantity, spectrum: u.Quantity):
        self._wl = wavelength
        self._spec = spectrum

    def emission_spectrum(self, wavelength: u.Quantity, **params) -> u.Quantity:
        # return in requested sampling, assume same grid
        return self._spec


class _DummyBinnedStellarSource(_DummyStellarSource):
    """
    Returns a binned spectrum with shape (2, n_wave) for CF00 tests.
    """
    def emission_spectrum(self, wavelength: u.Quantity, **params) -> u.Quantity:
        s = super().emission_spectrum(wavelength, **params)
        return np.stack([s.value, s.value], axis=0) << s.unit


class TestCalorimetricDustComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl = np.geomspace(1000, 300000, 1500) * u.AA
        cls.Lsrc = (np.ones(cls.wl.size) * 1e10) * (u.Lsun / u.AA)

    def test_calorimetric_returns_three_spectra(self):
        attenuation = dust.DustScreenAttenuation()
        attenuation.a_v.set(1.0 * u.mag)

        dust_em = dust.Casey2012DustComponent()
        calor = dust.CalorimetricDustComponent(
            attenuation=attenuation,
            dust_sed_component=dust_em,
        )

        source = _DummyStellarSource(self.wl, self.Lsrc)
        Lsrc, Latt, Ldust = calor.emission_spectrum(self.wl, source=source)

        self.assertEqual(Lsrc.unit, u.Lsun / u.AA)
        self.assertEqual(Latt.unit, u.Lsun / u.AA)
        self.assertEqual(Ldust.unit, u.Lsun / u.AA)

        self.assertEqual(Lsrc.shape, self.wl.shape)
        self.assertEqual(Latt.shape, self.wl.shape)
        self.assertEqual(Ldust.shape, self.wl.shape)

        self.assertTrue(np.isfinite(Lsrc.value).all())
        self.assertTrue(np.isfinite(Latt.value).all())
        self.assertTrue(np.isfinite(Ldust.value).all())

        # Attenuated should be <= intrinsic for a screen
        self.assertTrue(np.all(Latt.value <= Lsrc.value))

    def test_calorimetric_cf00_requires_binned_output(self):
        attenuation = dust.CharlotFall00Attenuation()
        dust_em = dust.Casey2012DustComponent()
        calor = dust.CalorimetricDustComponent(
            attenuation=attenuation,
            dust_sed_component=dust_em,
        )

        source = _DummyStellarSource(self.wl, self.Lsrc)

        with self.assertRaises(ValueError):
            _ = calor.emission_spectrum(self.wl, source=source)

    def test_calorimetric_cf00_with_binned_output(self):
        attenuation = dust.CharlotFall00Attenuation()
        attenuation.a_v_young.set(1.0 * u.mag)
        attenuation.a_v_old.set(0.2 * u.mag)

        dust_em = dust.Casey2012DustComponent()
        calor = dust.CalorimetricDustComponent(
            attenuation=attenuation,
            dust_sed_component=dust_em,
        )

        source = _DummyBinnedStellarSource(self.wl, self.Lsrc)

        Lsrc, Latt, Ldust = calor.emission_spectrum(self.wl, source=source)

        self.assertTrue(np.isfinite(Lsrc.value).all())
        self.assertTrue(np.isfinite(Latt.value).all())
        self.assertTrue(np.isfinite(Ldust.value).all())

# Legacy code

class TestDust(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting SSP model for testing dust models")
        self.dummy_wavelength = np.geomspace(100, 1e5, 3000) * u.angstrom
        # Monocromatic SED
        self.dummy_spectra = np.ones(
            self.dummy_wavelength.size) * const.c / self.dummy_wavelength**2 * 3631 * u.Jy

        self.ssp_model = SSP.PopStar(IMF="cha")

    def test_dust_screen(self):
        dust_screen = dust.DustScreen("ccm89")
        
        ext = dust_screen.get_extinction(self.dummy_wavelength)
        self.assertTrue(np.isfinite(ext).all())
        emission = dust_screen.get_emission(self.dummy_wavelength)
        self.assertTrue(np.allclose(emission, 0))

        ext_spectra = dust_screen.apply_extinction(
            self.dummy_wavelength, self.dummy_spectra)
        self.assertTrue(np.isfinite(ext_spectra).all())
        ext_em_spectra = dust_screen.apply_emission(
            self.dummy_wavelength, ext_spectra)
        self.assertTrue(np.allclose(ext_spectra, ext_em_spectra))

        reddened_ssp = dust_screen.redden_ssp_model(self.ssp_model, a_v=1.0)


if __name__ == '__main__':
    unittest.main()

import unittest
import os
import numpy as np
from astropy import units as u
from astropy import constants
from pst import observables

class TestObservables(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting SSP model")
        self.dummy_wavelength = np.geomspace(100, 1e5, 3000) * u.angstrom
        # Monocromatic SED
        self.dummy_flam = np.ones(self.dummy_wavelength.size
                                     ) * constants.c / self.dummy_wavelength**2 * 3631 * u.Jy
        self.dummy_fnu = np.ones(self.dummy_wavelength.size
                                ) * 3631 * u.Jy

    def test_default_dir(self):
        self.assertTrue(
            os.path.isdir(observables.PST_DATA_DIR),
            f"Default PST data dir {observables.PST_DATA_DIR} does not exists")

    def test_list_filters_available(self):
        paths = observables.list_of_available_filters()
        self.assertIsNotNone(paths)

    def test_load_photometric_filters(self):
        filters = observables.load_photometric_filters(
            ["SLOAN_SDSS.u", "SLOAN_SDSS.g", "SLOAN_SDSS.r"])
        self.assertIsNotNone(filters)
        for f in filters:
            self.assertTrue(isinstance(f, observables.Filter))
    
    def test_filter(self):
        filter = observables.Filter.from_svo("SLOAN_SDSS.i")
        self.assertIsNotNone(filter)
        # Reload filter from file
        filter = observables.Filter.from_text_file(
            os.path.join(observables.PST_DATA_DIR, "filters", "SLOAN_SDSS.i.dat")
        )
        self.assertIsNotNone(filter)

        self.assertTrue(np.isclose(filter.effective_wavelength(),
                                   7499.70417446 * u.angstrom),
                        "Unexpected effective wavelength value")
        
        self.assertTrue(np.isclose(filter.effective_bandwidth(),
                                   902.02184282 * u.angstrom),
                        "Unexpected effective bandwidth value")
        
        self.assertTrue(np.isclose(filter.effective_transmission(),
                                   0.32484839450189695),
                        "Unexpected effective transmission value")

        # Interpolate filter to input wavelength array
        filter.interpolate(self.dummy_wavelength)
        # Use flam
        flux, _ = filter.get_fnu(self.dummy_flam)
        self.assertTrue(np.isclose(flux, 3631.0 * u.Jy),
                        f"Unexpected integrated flux value: {flux}")
        
        mag, _ = filter.get_ab(self.dummy_flam)
        self.assertTrue(np.isclose(mag, 0.0, atol=1e-4),
                        f"Unexpected magnitude value: {mag}")
        # Use fnu
        flux, _ = filter.get_fnu(self.dummy_fnu)
        self.assertTrue(np.isclose(flux, 3631.0 * u.Jy),
                        f"Unexpected integrated flux value: {flux}")

        mag, _ = filter.get_ab(self.dummy_fnu)
        self.assertTrue(np.isclose(mag, 0.0, atol=1e-4),
                        f"Unexpected magnitude value: {mag}")

        fig, ax = filter.plot(show=False)

        self.assertIsNotNone(ax)
        self.assertIsNotNone(fig)

        f_lambda, f_lambda_err = filter.get_flambda_vegamag(
            self.dummy_flam, spectra_err=0.05 * self.dummy_flam
        )
        self.assertTrue(np.isfinite(f_lambda))
        self.assertTrue(np.isfinite(f_lambda_err))

    def test_equivalent_width(self):
        eqwidth = observables.EquivalentWidth.from_name("lick_ha")
        ew, ew_err = eqwidth.compute_ew(self.dummy_wavelength, self.dummy_flam)
        self.assertTrue(np.isfinite(ew), "Unexpected EW value")
        ew2, ew2_err = eqwidth.compute_ew(
            self.dummy_wavelength, self.dummy_flam, spectra_err=0.1 * self.dummy_flam
        )
        self.assertTrue(np.isfinite(ew2))
        self.assertTrue(np.isfinite(ew2_err))

    def test_equivalent_width_error_shape_consistency(self):
        eqwidth = observables.EquivalentWidth.from_name("lick_ha")
        ew_1d, ew_err_1d = eqwidth.compute_ew(
            self.dummy_wavelength,
            self.dummy_flam,
            spectra_err=0.1 * self.dummy_flam,
        )

        ew_2d, ew_err_2d = eqwidth.compute_ew(
            self.dummy_wavelength,
            self.dummy_flam[:, np.newaxis],
            spectra_err=(0.1 * self.dummy_flam)[:, np.newaxis],
        )

        self.assertTrue(np.isclose(ew_1d, ew_2d[0]))
        self.assertTrue(np.isclose(ew_err_1d, ew_err_2d[0]))

    def test_equivalent_width_error_matches_analytic_constant_spectrum(self):
        wavelength = np.linspace(3500.0, 4500.0, 1001) * u.angstrom
        flux_level = 10.0 * u.erg / (u.s * u.cm**2 * u.angstrom)
        sigma_level = 2.0 * u.erg / (u.s * u.cm**2 * u.angstrom)
        spectra = np.ones(wavelength.size) * flux_level
        spectra_err = np.ones(wavelength.size) * sigma_level

        eqwidth = observables.EquivalentWidth(
            left_wl_range=(3600.0, 3700.0),
            central_wl_range=(3900.0, 4100.0),
            right_wl_range=(4300.0, 4400.0),
        )

        ew, ew_err = eqwidth.compute_ew(wavelength, spectra, spectra_err=spectra_err)

        left_mask = ((wavelength >= eqwidth.left_wl_range[0])
                     & (wavelength <= eqwidth.left_wl_range[1]))
        right_mask = ((wavelength >= eqwidth.right_wl_range[0])
                      & (wavelength <= eqwidth.right_wl_range[1]))
        central_mask = ((wavelength >= eqwidth.central_wl_range[0])
                        & (wavelength <= eqwidth.central_wl_range[1]))

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        left_mean_wl = eqwidth.left_wl_range.mean()
        right_mean_wl = eqwidth.right_wl_range.mean()
        central_wl = wavelength[central_mask]
        t = (central_wl - left_mean_wl) / (right_mean_wl - left_mean_wl)

        if central_wl.size < 2:
            trapz_w = np.zeros_like(central_wl)
        else:
            trapz_w = np.empty_like(central_wl)
            trapz_w[0] = 0.5 * (central_wl[1] - central_wl[0])
            trapz_w[-1] = 0.5 * (central_wl[-1] - central_wl[-2])
            if central_wl.size > 2:
                trapz_w[1:-1] = 0.5 * (central_wl[2:] - central_wl[:-2])

        var_cont = sigma_level**2 * ((1 - t)**2 / n_left + t**2 / n_right)
        expected_ew_var = np.sum((trapz_w / flux_level)**2 * (sigma_level**2 + var_cont))
        expected_ew_err = np.sqrt(expected_ew_var)

        self.assertTrue(np.isclose(ew, 0.0 * u.angstrom, atol=1e-10 * u.angstrom))
        self.assertTrue(np.isclose(ew_err, expected_ew_err, rtol=1e-12))

    def test_equivalent_width_uses_wavelength_resolved_integrand(self):
        wavelength = np.linspace(3500.0, 4500.0, 2001) * u.angstrom
        eqwidth = observables.EquivalentWidth(
            left_wl_range=(3600.0, 3700.0),
            central_wl_range=(3900.0, 4100.0),
            right_wl_range=(4300.0, 4400.0),
        )

        left_mean_wl = eqwidth.left_wl_range.mean()
        right_mean_wl = eqwidth.right_wl_range.mean()

        true_left = 12.0 * u.erg / (u.s * u.cm**2 * u.angstrom)
        true_right = 8.0 * u.erg / (u.s * u.cm**2 * u.angstrom)
        slope = (true_right - true_left) / (right_mean_wl - left_mean_wl)
        continuum = true_left + slope * (wavelength - left_mean_wl)

        line_profile = 1.0 - 0.2 * np.exp(-0.5 * ((wavelength - 4000.0 * u.angstrom)
                                                  / (10.0 * u.angstrom))**2)
        spectra = continuum * line_profile

        ew, _ = eqwidth.compute_ew(wavelength, spectra)

        central_mask = ((wavelength >= eqwidth.central_wl_range[0])
                        & (wavelength <= eqwidth.central_wl_range[1]))
        central_wl = wavelength[central_mask]
        t = (central_wl - left_mean_wl) / (right_mean_wl - left_mean_wl)
        fcont = (1 - t) * true_left + t * true_right
        expected_ew = np.trapezoid(1 - spectra[central_mask] / fcont, x=central_wl)

        self.assertTrue(np.isclose(ew, expected_ew, rtol=1e-12))

    def test_equivalent_width_from_atlas(self):
        hb = observables.EquivalentWidth.from_atlas("Hb")
        self.assertTrue(np.allclose(hb.left_wl_range.value, [4828.875, 4848.875]))
        self.assertTrue(np.allclose(hb.central_wl_range.value, [4848.875, 4877.625]))
        self.assertTrue(np.allclose(hb.right_wl_range.value, [4877.625, 4892.625]))


class TestEquivalentWidthList(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.wavelength = np.linspace(3500.0, 7000.0, 5000) * u.angstrom
        base = np.ones(self.wavelength.size) * 10.0
        absorption = 1.0 - 0.15 * np.exp(-0.5 * ((self.wavelength.value - 6563.0) / 3.0) ** 2)
        emission = 1.0 + 0.10 * np.exp(-0.5 * ((self.wavelength.value - 4861.0) / 4.0) ** 2)
        self.spectrum = (base * absorption * emission) * u.erg / (u.s * u.cm**2 * u.angstrom)
        self.spectrum_err = 0.05 * self.spectrum

        self.ew_ha = observables.EquivalentWidth.from_name("lick_ha")
        self.ew_hb = observables.EquivalentWidth.from_name("lick_hb")

    def test_requires_wavelength(self):
        ew_list = observables.EquivalentWidthList([self.ew_ha, self.ew_hb])
        with self.assertRaises(TypeError):
            ew_list.compute_ew(self.spectrum)

    def test_matches_individual_indices(self):
        ew_list = observables.EquivalentWidthList([self.ew_ha, self.ew_hb])
        ew_many, ew_many_err = ew_list.compute_ew(
            self.wavelength,
            self.spectrum,
            spectra_err=self.spectrum_err,
        )

        ew_ha, ew_ha_err = self.ew_ha.compute_ew(self.wavelength, self.spectrum, spectra_err=self.spectrum_err)
        ew_hb, ew_hb_err = self.ew_hb.compute_ew(self.wavelength, self.spectrum, spectra_err=self.spectrum_err)

        self.assertTrue(np.isclose(ew_many[0], ew_ha, rtol=1e-12))
        self.assertTrue(np.isclose(ew_many[1], ew_hb, rtol=1e-12))
        self.assertTrue(np.isclose(ew_many_err[0], ew_ha_err, rtol=1e-12))
        self.assertTrue(np.isclose(ew_many_err[1], ew_hb_err, rtol=1e-12))

    def test_shape_for_batched_spectra(self):
        ew_list = observables.EquivalentWidthList([self.ew_ha, self.ew_hb])
        spectra = np.vstack([self.spectrum.value, 1.2 * self.spectrum.value]) * self.spectrum.unit
        spectra = spectra.T
        ew_many, ew_many_err = ew_list.compute_ew(
            self.wavelength,
            spectra,
            spectra_err=0.1 * spectra,
        )

        self.assertEqual(ew_many.shape, (2, 2))
        self.assertEqual(ew_many_err.shape, (2, 2))

# Add these tests to your existing TestObservables class.
# They assume FilterList is available as observables.FilterList (or adjust import accordingly).

class TestFilterList(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dummy_wavelength = np.geomspace(100, 1e5, 3000) * u.angstrom
        # Flat f_nu = 3631 Jy (AB=0 everywhere) expressed in f_lambda
        self.dummy_flam = (constants.c / self.dummy_wavelength**2) * (3631 * u.Jy)
        self.dummy_fnu = np.ones(self.dummy_wavelength.size) * 3631 * u.Jy

        # Load a few real filters
        self.filters = observables.load_photometric_filters(
            ["SLOAN_SDSS.u", "SLOAN_SDSS.g", "SLOAN_SDSS.r"]
        )
        self.flist = observables.FilterList(self.filters).interpolate(self.dummy_wavelength)

    def test_filterlist_postinit_names(self):
        fl = observables.FilterList(self.filters)
        self.assertEqual(fl.n_bands, len(self.filters))
        self.assertEqual(len(fl.names), len(self.filters))
        self.assertTrue(all(isinstance(n, str) for n in fl.names))

    def test_filterlist_requires_interpolated(self):
        fl = observables.FilterList(self.filters)
        with self.assertRaises(RuntimeError):
            fl.get_fnu(self.dummy_flam)
        with self.assertRaises(RuntimeError):
            fl.get_photons(self.dummy_flam)
        with self.assertRaises(RuntimeError):
            fl.abmag(self.dummy_flam)

    def test_filterlist_interpolate_sets_arrays(self):
        fl = observables.FilterList(self.filters).interpolate(self.dummy_wavelength)
        self.assertIsNotNone(fl.wavelength)
        self.assertIsNotNone(fl.response)
        self.assertIsNotNone(fl.dlambda)
        self.assertIsNotNone(fl.kernel_phot)
        self.assertIsNotNone(fl.norm_phot)

        self.assertEqual(fl.response.shape, (fl.n_bands, self.dummy_wavelength.size))
        self.assertEqual(fl.dlambda.shape, self.dummy_wavelength.shape)
        self.assertEqual(fl.kernel_phot.shape, (fl.n_bands, self.dummy_wavelength.size))
        self.assertEqual(fl.norm_phot.shape, (fl.n_bands,))

        # dlambda should be positive for strictly increasing wavelength
        self.assertTrue(np.all(fl.dlambda > 0 * fl.dlambda.unit))

    def test_filterlist_matches_single_filter_get_fnu(self):
        # Compare FilterList.get_fnu against per-filter Filter.get_fnu
        fnu_list, _ = self.flist.get_fnu(self.dummy_flam)

        for i, f in enumerate(self.filters):
            f.interpolate(self.dummy_wavelength)
            fnu_single, _ = f.get_fnu(self.dummy_flam)
            self.assertTrue(
                np.isclose(fnu_list[i].to_value(u.Jy), fnu_single.to_value(u.Jy), rtol=1e-6, atol=0.0),
                f"Band {f.name}: FilterList fnu={fnu_list[i]}, Filter fnu={fnu_single}",
            )

    def test_filterlist_abmag_zero_for_flat_fnu(self):
        mag, _ = self.flist.abmag(self.dummy_flam)
        # AB system: flat 3631 Jy should give 0 mag in every band
        self.assertTrue(
            np.allclose(mag.value, 0.0, atol=1e-4),
            f"Unexpected AB mags: {mag}",
        )

        # Same check using flat fnu input directly
        mag2, _ = self.flist.abmag(self.dummy_fnu)
        self.assertTrue(
            np.allclose(mag2.value, 0.0, atol=1e-4),
            f"Unexpected AB mags from fnu input: {mag2}",
        )

    def test_filterlist_shapes_broadcast(self):
        # Provide a batch of spectra: (n_spec, n_wave)
        F = np.vstack([self.dummy_flam.value, 2.0 * self.dummy_flam.value]) * self.dummy_flam.unit
        fnu, _ = self.flist.get_fnu(F)
        self.assertEqual(fnu.shape, (2, self.flist.n_bands))
        # Second spectrum has 2x flux -> 2x fnu
        ratio = (fnu[1] / fnu[0]).to_value(u.dimensionless_unscaled)
        self.assertTrue(np.allclose(ratio, 2.0, rtol=1e-6))

    def test_filterlist_error_propagation_linear(self):
        # If Ferr is constant fractional error, photons error should scale linearly too
        frac = 0.1
        Ferr = self.dummy_flam * frac
        fnu, fnu_err = self.flist.get_fnu(self.dummy_flam, spectra_err=Ferr)
        self.assertIsNotNone(fnu_err)
        rel = (fnu_err / fnu).to_value(u.dimensionless_unscaled)
        # Expect ~10% relative error in all bands (ignoring any NaN handling)
        self.assertTrue(np.allclose(rel, frac, rtol=1e-6, atol=0.0))

    def test_filterlist_nan_masking(self):
        F = self.dummy_flam.copy()
        F = u.Quantity(F.value.copy(), unit=F.unit)
        F.value[100:200] = np.nan

        # With masking, should produce finite outputs
        fnu, _ = self.flist.get_fnu(F, mask_nan=True)
        self.assertTrue(np.all(np.isfinite(fnu.to_value(u.Jy))))

        # Without masking, einsum should propagate NaNs -> at least one band NaN likely
        fnu2, _ = self.flist.get_fnu(F, mask_nan=False)
        self.assertTrue(np.any(~np.isfinite(fnu2.to_value(u.Jy))))

    def test_filterlist_wavelength_range(self):
        lo, hi = self.flist.wavelength_range()
        self.assertTrue(lo < hi)
        self.assertTrue(lo.unit.is_equivalent(u.angstrom))
        self.assertTrue(hi.unit.is_equivalent(u.angstrom))

    def test_filterlist_interpolate_rejects_non_1d(self):
        wl2d = np.vstack([self.dummy_wavelength.value, self.dummy_wavelength.value]) * u.angstrom
        fl = observables.FilterList(self.filters)
        with self.assertRaises(ValueError):
            fl.interpolate(wl2d)

    def test_filterlist_interpolate_rejects_non_monotonic(self):
        wl = self.dummy_wavelength.copy()
        wl = u.Quantity(wl.value.copy(), wl.unit)
        wl.value[100] = wl.value[99]
        fl = observables.FilterList(self.filters)
        with self.assertRaises(ValueError):
            fl.interpolate(wl)

    def test_filterlist_interpolate_rejects_too_short_grid(self):
        fl = observables.FilterList(self.filters)
        with self.assertRaises(ValueError):
            fl.interpolate(np.array([5500.0]) * u.angstrom)
        
if __name__ == '__main__':
    unittest.main()

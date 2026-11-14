"""Batched flux-conserving resampling of SSP spectra."""
import unittest

import numpy as np
from astropy import units as u

from pst import SSP
from pst.utils import flux_conserving_interpolation, resample_via_bin_frac

LSUN = u.Lsun / u.AA / u.Msun


class TestBatchedResampling(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.wl = np.cumsum(rng.uniform(0.5, 1.5, 800)) + 3000.0
        self.new_wl = np.geomspace(3050.0, 3750.0, 333)
        self.spectra = rng.uniform(0.5, 2.0, (800, 6))
        self.spectra[[10, 400], [1, 4]] = np.nan
        self.var = rng.uniform(0.01, 0.1, (800, 6))

    def test_2d_matches_column_by_column(self):
        for fill_value in (0.0, 1.5):
            f2d, v2d = resample_via_bin_frac(
                self.wl, self.spectra, w_out=self.new_wl, var_in=self.var,
                fill_value=fill_value)
            self.assertEqual(f2d.shape, (self.new_wl.size, 6))
            for k in range(6):
                f1d, v1d = resample_via_bin_frac(
                    self.wl, self.spectra[:, k], w_out=self.new_wl,
                    var_in=self.var[:, k], fill_value=fill_value)
                np.testing.assert_array_equal(f2d[:, k], f1d)
                np.testing.assert_array_equal(v2d[:, k], v1d)

    def _make_ssp(self, shape):
        ssp = SSP.SSPBase()
        ssp.ages = np.geomspace(0.01, 13.0, shape[-1]) << u.Gyr
        ssp.metallicities = np.geomspace(1e-3, 0.03, shape[-2]) << u.dimensionless_unscaled
        if len(shape) == 3:
            ssp.alpha_fe = np.linspace(0.0, 0.4, shape[0])
        ssp.wavelength = self.wl << u.AA
        rng = np.random.default_rng(3)
        ssp.L_lambda = rng.uniform(0.1, 1.0, (*shape, self.wl.size)
                                   ).astype(np.float32) << LSUN
        return ssp

    def test_interpolate_sed_matches_loop(self):
        for shape in ((3, 4), (2, 3, 4)):
            ssp = self._make_ssp(shape)
            expected = np.empty((*shape, self.new_wl.size), dtype=np.float32)
            for index in np.ndindex(shape):
                expected[index] = flux_conserving_interpolation(
                    self.new_wl << u.AA, ssp.wavelength, ssp.L_lambda[index]).value
            # Small chunks force several batches
            new = SSP._resample_spectra(self.new_wl << u.AA, ssp.wavelength,
                                        ssp.L_lambda, chunk_size=3 * self.wl.size)
            np.testing.assert_array_equal(new.value, expected)
            ssp.interpolate_sed(self.new_wl << u.AA, verbose=False)
            np.testing.assert_array_equal(ssp.L_lambda.value, expected)
            self.assertEqual(ssp.L_lambda.dtype, np.float32)
            self.assertEqual(ssp.L_lambda.unit, LSUN)

    def test_other_methods_still_loop(self):
        ssp = self._make_ssp((2, 3))
        new = SSP._resample_spectra(self.new_wl << u.AA, ssp.wavelength,
                                    ssp.L_lambda, method="cumulative")
        expected = flux_conserving_interpolation(
            self.new_wl << u.AA, ssp.wavelength, ssp.L_lambda[1, 2],
            method="cumulative").value
        np.testing.assert_allclose(new.value[1, 2], expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

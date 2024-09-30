import unittest
import numpy as np

from pst import SSP
from pst.observables import TopHatFilter

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
        self.assertEqual(self.ssp_model.wavelength.size, new_wl.size - 1)
        self.assertEqual(self.ssp_model.L_lambda.shape[-1], new_wl.size - 1)

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

if __name__ == '__main__':
    unittest.main()
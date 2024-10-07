import unittest
import numpy as np
from astropy import units as u
from astropy import constants
from pst import dust, SSP

class TestDust(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting SSP model for testing dust models")
        self.dummy_wavelength = np.geomspace(100, 1e5, 3000) * u.angstrom
        # Monocromatic SED
        self.dummy_spectra = np.ones(
            self.dummy_wavelength.size) * constants.c / self.dummy_wavelength**2 * 3631 * u.Jy

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
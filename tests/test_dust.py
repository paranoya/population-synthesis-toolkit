import unittest
import numpy as np
from astropy import units as u
from astropy import constants
from pst import dust, SSP

class TestUtils(unittest.TestCase):
    def test_modified_blackbody(self):
        wavelength = np.geomspace(10, 1e4, 1000) * u.um
        temperature = 30 * u.K
        beta = 1.5

        mbb_spectrum = dust.modified_blackbody(
            wavelength, T=temperature, beta=beta, per_freq=False)

        self.assertEqual(mbb_spectrum.unit, u.Unit("erg / (Angstrom s sr cm2)"))
        self.assertTrue(np.isfinite(mbb_spectrum).all())

class TestAttenuationModels(unittest.TestCase):

    def test_dust_screen(self):
        dust_screen = dust.DustScreenAttenuation()

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

class TestDustEmissionModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Testing Casey 2012 IR emission model")
        self.dummy_wl = np.geomspace(0.1, 1000, 100) << u.um

    def test_casey2012(self):
        model = dust.Casey2012DustComponent()
        # Check lambda pivot
        lam_c = model._lambda_pivot(t_dust=35 << u.K, alpha=2.0)
        self.assertEqual(lam_c.unit, u.um)
        # Check components
        mbb, pl = model._shape_l_lambda(self.dummy_wl, t_dust=55 << u.K, beta=1.6, alpha=2,
                                        lam_pivot=lam_c)
        # Check composite
        spec = model.emission_spectrum(self.dummy_wl,
                                       t_dust=55 << u.K, beta=1.6, alpha=2,
                                       lum_ir=1e11 << u.Lsun)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.dummy_wl, spec)
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel("Flux (" + str(spec.unit) + ")")
        plt.show()


if __name__ == '__main__':
    unittest.main()
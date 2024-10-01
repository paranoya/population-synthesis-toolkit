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
        self.dummy_spectra = np.ones(self.dummy_wavelength.size
                                     ) * constants.c / self.dummy_wavelength**2 * 3631 * u.Jy

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

        filter.interpolate(self.dummy_wavelength)
        flux, _ = filter.get_fnu(self.dummy_spectra)
        self.assertTrue(np.isclose(flux, 3631.0 * u.Jy),
                        f"Unexpected integrated flux value: {flux}")
        
        mag, _ = filter.get_ab(self.dummy_spectra)
        self.assertTrue(np.isclose(mag, 0.0, atol=1e-4),
                        f"Unexpected magnitude value: {mag}")

        fig = filter.plot(show=False)

    def test_equivalent_width(self):
        eqwidth = observables.EquivalentWidth.from_name("lick_ha")
        ew, ew_err = eqwidth.compute_ew(self.dummy_wavelength, self.dummy_spectra)
        self.assertTrue(np.isfinite(ew), "Unexpected EW value")


if __name__ == '__main__':
    unittest.main()
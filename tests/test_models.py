import unittest
import numpy as np
from astropy import units as u
from astropy import constants
from pst import models, SSP

np.random.seed(50)

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting SSP model for testing dust models")
        self.dummy_times = (13.7 - np.geomspace(1e-3, 13.7, 50)[::-1]
                            ) * u.Gyr
        self.ssp_model = SSP.PopStar(IMF="cha")

    def test_single_burst(self):
        model = models.SingleBurstCEM(time_burst=5 * u.Gyr,
                                      mass_burst=1 * u.Msun,
                                      burst_metallicity=0.02)
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(mass[0] == 0 * u.Msun)
        self.assertTrue(mass[-1] == 1 * u.Msun)

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.allclose(z, 0.02))

    def test_exponential(self):
        model = models.ExponentialCEM(tau= 0.1 * u.Gyr,
                                      stellar_mass_inf=1 * u.Msun,
                                      metallicity=0.02)
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(np.isclose(mass[0], 0 * u.Msun))
        self.assertTrue(np.isclose(mass[-1], 1 * u.Msun))

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.allclose(z, 0.02))

    def test_exponential_quenched(self):
        model = models.ExponentialQuenchedCEM(tau= 10 * u.Gyr,
                                      stellar_mass_inf=1 * u.Msun,
                                      metallicity=0.02,
                                      quenching_time=13.0 * u.Gyr)
        quenched_times = self.dummy_times >= 13.0 * u.Gyr
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(np.isclose(mass[0], 0 * u.Msun))
        self.assertTrue((mass[quenched_times] == mass[-1]).all())

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.allclose(z, 0.02))

    def test_delayed_tau(self):
        model = models.ExponentialDelayedCEM(tau= 10 * u.Gyr,
                                    today=13.7 * u.Gyr,
                                    mass_today = 1.0 * u.Msun,
                                    ism_metallicity_today=0.02)
        mass = model.stellar_mass_formed(self.dummy_times)
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(np.isclose(mass[0], 0 * u.Msun, rtol=1e-4))
        self.assertTrue(np.isclose(mass[-1], 1 * u.Msun, rtol=1e-4))

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.allclose(z, 0.02))

    def test_delayed_tau_powerlaw(self):
        model = models.ExponentialDelayedZPowerLawCEM(
            tau= 10 * u.Gyr,
            today=13.7 * u.Gyr,
            mass_today = 1 * u.Msun,
            ism_metallicity_today=0.02,
            alpha_powerlaw=1)
        mass = model.stellar_mass_formed(self.dummy_times)
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(np.isclose(mass[0], 0 * u.Msun, rtol=1e-4))
        self.assertTrue(np.isclose(mass[-1], 1 * u.Msun, rtol=1e-4))

        z = model.ism_metallicity(self.dummy_times)
        print(z[-1])
        self.assertTrue(np.isclose(z[-1], 0.02, rtol=1e-4))

    def test_delayed_tau_quenched(self):
        model = models.ExponentialDelayedQuenchedCEM(
            tau= 10 * u.Gyr,
            today=13.7 * u.Gyr,
            mass_today = 1 * u.Msun,
            ism_metallicity_today=0.02,
            alpha_powerlaw=1,
            quenching_time=13.0 * u.Gyr)
        
        quenched_times = self.dummy_times >= 13.0 * u.Gyr
        mass = model.stellar_mass_formed(self.dummy_times)

        self.assertTrue(np.isclose(mass[0], 0 * u.Msun, rtol=1e-4))
        self.assertTrue((mass[quenched_times] == mass[-1]).all())

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.isclose(z[-1].value, 0.02, rtol=1e-4))

    def test_lognormal_zpowerlaw(self):
        model = models.LogNormalZPowerLawCEM(
            t0=3.0, scale=1.0, mass_today=1.0,
            today=13.7,
            ism_metallicity_today=0.02, alpha_powerlaw=2.0
        )
        mass = model.stellar_mass_formed(self.dummy_times)
        metals = model.ism_metallicity(self.dummy_times)

        self.assertEqual(mass[0], 0.0)
        self.assertTrue(np.isclose(mass[-1], 1.0 * u.Msun, rtol=1e-4))
        self.assertEqual(metals[0], 0.0)
        self.assertTrue(np.isclose(metals[-1], 0.02, rtol=1e-4))
    
    def test_tabular(self):
        low_res_time = np.linspace(0, 13.7, 10) * u.Gyr
        masses = 1 - np.exp(-low_res_time / 3.0 / u.Gyr)
        model = models.TabularCEM(
            times=low_res_time, masses=masses * u.Msun,
            metallicities=np.full(masses.size, fill_value=0.02))

        mass = model.stellar_mass_formed(self.dummy_times)
        real_mass = 1 - np.exp(- self.dummy_times / 3 / u.Gyr)
        self.assertTrue(np.allclose(mass, real_mass * u.Msun, rtol=1e-2))

    def test_particle_grid(self):
        n_particles = 10000
        particles_z = 10**(np.random.uniform(-4, 0.3, n_particles))
        particles_t_form = np.random.exponential(3, n_particles)
        particles_mass = 10**(np.random.uniform(5, 6, n_particles))
        model = models.ParticleListCEM(
            time_form=particles_t_form * u.Gyr,
            metallicities=particles_z * u.dimensionless_unscaled,
            masses=particles_mass * u.Msun)
        
        _ = model.stellar_mass_formed(self.dummy_times)
    
        spectra = model.compute_SED(self.ssp_model, t_obs=13.7 * u.Gyr)
        self.assertTrue(np.isfinite(spectra).all())


if __name__ == '__main__':
    unittest.main()
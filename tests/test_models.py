import unittest
from unittest import mock
import numpy as np
from astropy import units as u
from astropy import constants
from pst import cem, models, SSP

np.random.seed(50)


def make_toy_ssp_for_cem():
    ssp = SSP.SSPBase()
    ssp.name = "toy_cem_ssp"
    ssp.ages = np.array([1.0, 10.0]) * u.Gyr
    ssp.metallicities = np.array([0.01, 0.02]) << u.dimensionless_unscaled
    ssp.wavelength = np.array([1000, 2000, 3000]) << u.AA
    ssp.L_lambda = np.ones((2, 2, 3)) << (u.Lsun / u.AA / u.Msun)
    ssp.returned_mass_frac = np.array([[0.5, 0.75], [0.8, 0.9]])
    ssp.supernova_rate = np.array([[1.0, 2.0], [3.0, 4.0]]) << (u.yr**-1 / u.Msun)
    ssp.log_ionising_HI_photons = np.array([[40.0, 41.0], [39.0, 38.0]]) << u.dex(u.s**-1 / u.Msun)
    ssp.log_ionising_HeI_photons = np.array([[39.5, 40.5], [38.5, 37.5]]) << u.dex(u.s**-1 / u.Msun)
    ssp.log_ionising_HeII_photons = np.array([[39.0, 40.0], [38.0, 37.0]]) << u.dex(u.s**-1 / u.Msun)
    return ssp


class LinearCEM(cem.ChemicalEvolutionModel):
    name = "linear_cem"

    def stellar_mass_formed(self, time: u.Quantity) -> u.Quantity:
        time = u.Quantity(time).to(u.Gyr)
        return np.atleast_1d(time.value) << u.Msun

    def ism_metallicity(self, time: u.Quantity) -> u.Quantity:
        time = u.Quantity(time).to(u.Gyr)
        return np.full(np.atleast_1d(time.value).shape, 0.01) << u.dimensionless_unscaled

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
                                      today=13.7 * u.Gyr,
                                      burst_metallicity=0.02)
        mass = model.stellar_mass_formed(self.dummy_times)
        self.assertTrue(model.name == "single_burst_cem")
        self.assertTrue(mass[0] == 0 * u.Msun)
        self.assertTrue(mass[-1] == 1 * u.Msun)

        z = model.ism_metallicity(self.dummy_times)
        self.assertTrue(np.allclose(z, 0.02))

    def test_exponential(self):
        model = models.ExponentialCEM(tau= 0.1 * u.Gyr,
                                      stellar_mass_inf=1 * u.Msun,
                                      metallicity=0.02)
        self.assertTrue(model.name == "exponential_cem")
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
        self.assertTrue(model.name == "exponential_quenched_cem")
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

    def test_cc25tabular(self):
        tau = np.array([1.0, 0.1]) << u.Gyr
        ssfr = np.array([1.0, 0.1]) << (1 / u.Gyr)
        model = models.CC25TabularCEM(
            tau_ssfr=tau, ssfr=ssfr, mass_today=1.0 * u.Msun,
            today=13.7 << u.Gyr,
            ism_metallicity_today=0.02, alpha_powerlaw=1.0)

        parameters = model.parameters_recursive(include_fixed=False)
        # This parameters should be fixed
        self.assertFalse("times" in parameters)
        self.assertFalse("masses" in parameters)
        
        self.assertTrue(
            np.all(model.table_mass.to("Msun") == np.array([0., 0., 0.99, 1.0]) << u.Msun))
        self.assertTrue(
            np.all(model.table_t.to("Gyr") == np.array([0., 12.7, 13.6, 13.7]) << u.Gyr))

    def test_fixedmassfrac(self):
        m_frac = np.array([0, 0.5, 1.0])
        times = np.array([0.1, 5, 10]) << u.Gyr        
        model = models.TabularMassFracCEM(mass_frac=m_frac, times=times, today=13.7,
                                          mass_today=1, ism_metallicity_today=0.02,
                                          alpha_powerlaw=1.0)
        parameters = model.parameters_recursive(include_fixed=True)
        self.assertIn("times", parameters.keys())
        self.assertIn("masses", parameters.keys())
        parameters = model.parameters_recursive(include_fixed=False)
        self.assertNotIn("masses", parameters.keys())

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

        spectra = model.compute_SED(self.ssp_model, t_obs=13.7 * u.Gyr,
                                    age_bin_edges=[0, 1e9, 1e10])
        self.assertEqual(spectra.ndim, 2)
        # from matplotlib import pyplot as plt
        # plt.figure()
        # for s in spectra:
        #     plt.plot(self.ssp_model.wavelength, s)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.show()

    def test_time_at_stellar_mass_frac_requires_today(self):
        model = models.ExponentialCEM(
            tau=1.0 * u.Gyr, stellar_mass_inf=1.0 * u.Msun, metallicity=0.02
        )
        with self.assertRaises(ValueError):
            _ = model.time_at_stellar_mass_frac(0.5)

    def test_time_at_stellar_mass_frac_monotonic(self):
        model = models.ExponentialDelayedCEM(
            tau=2.0 * u.Gyr,
            today=13.7 * u.Gyr,
            mass_today=1.0 * u.Msun,
            ism_metallicity_today=0.02,
        )
        tf = model.time_at_stellar_mass_frac([0.1, 0.5, 0.9], time_res=0.05 * u.Gyr)
        self.assertTrue(np.all(np.diff(tf.to_value(u.Gyr)) > 0))

    def test_compute_photometry_ndarray_and_age_bins(self):
        model = models.ExponentialDelayedCEM(
            tau=3.0 * u.Gyr,
            today=13.7 * u.Gyr,
            mass_today=1.0 * u.Msun,
            ism_metallicity_today=0.02,
        )
        n_band = 4
        n_z = self.ssp_model.metallicities.size
        n_age = self.ssp_model.ages.size
        phot_grid = np.ones((n_band, n_z, n_age), dtype=float)

        p = model.compute_photometry(self.ssp_model, 13.7 * u.Gyr, photometry=phot_grid)
        self.assertEqual(p.shape, (n_band,))
        self.assertTrue(np.isfinite(p.value).all())

        p_bin = model.compute_photometry(
            self.ssp_model,
            13.7 * u.Gyr,
            photometry=phot_grid,
            age_bin_edges=[0, 1e9, 1e10] * u.yr,
        )
        self.assertEqual(p_bin.shape[0], 2)
        self.assertEqual(p_bin.shape[1], n_band)
        self.assertTrue(np.isfinite(p_bin.value).all())

    def test_cc25tabular_rejects_invalid_tau_order(self):
        tau_bad = np.array([0.1, 1.0]) << u.Gyr
        ssfr = np.array([1.0, 0.1]) << (1 / u.Gyr)
        with self.assertRaises(ValueError):
            _ = models.CC25TabularCEM(
                tau_ssfr=tau_bad,
                ssfr=ssfr,
                mass_today=1.0 * u.Msun,
                today=13.7 << u.Gyr,
                ism_metallicity_today=0.02,
                alpha_powerlaw=1.0,
            )

    def test_cc25tabular_rejects_nonpositive_tau(self):
        tau_bad = np.array([1.0, 0.0]) << u.Gyr
        ssfr = np.array([1.0, 0.1]) << (1 / u.Gyr)
        with self.assertRaises(ValueError):
            _ = models.CC25TabularCEM(
                tau_ssfr=tau_bad,
                ssfr=ssfr,
                mass_today=1.0 * u.Msun,
                today=13.7 << u.Gyr,
                ism_metallicity_today=0.02,
                alpha_powerlaw=1.0,
            )

    def test_tabular_massfrac_rejects_out_of_range(self):
        times = np.array([0.1, 5, 10]) << u.Gyr
        with self.assertRaises(ValueError):
            _ = models.TabularMassFracCEM(
                mass_frac=np.array([0.0, 1.2, 1.0]),
                times=times,
                today=13.7,
                mass_today=1,
                ism_metallicity_today=0.02,
                alpha_powerlaw=1.0,
            )

    def test_tabular_massfrac_rejects_nonmonotonic(self):
        times = np.array([0.1, 5, 10]) << u.Gyr
        with self.assertRaises(ValueError):
            _ = models.TabularMassFracCEM(
                mass_frac=np.array([0.0, 0.8, 0.7]),
                times=times,
                today=13.7,
                mass_today=1,
                ism_metallicity_today=0.02,
                alpha_powerlaw=1.0,
            )

    def test_interpolate_ssp_masses_cache_disabled_returns_fresh_arrays(self):
        model = LinearCEM(today=13.7 * u.Gyr, cache_interp_ssp_mass=False)
        ssp = make_toy_ssp_for_cem()

        weights_1 = model.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)
        weights_2 = model.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)

        self.assertIsNot(weights_1, weights_2)
        self.assertEqual(model._ssp_weights_cache, {})

    def test_interpolate_ssp_masses_cache_enabled_is_instance_local(self):
        ssp = make_toy_ssp_for_cem()
        model_1 = LinearCEM(today=13.7 * u.Gyr, cache_interp_ssp_mass=True)
        model_2 = LinearCEM(today=13.7 * u.Gyr, cache_interp_ssp_mass=True)

        weights_1a = model_1.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)
        weights_1b = model_1.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)
        weights_2 = model_2.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)

        self.assertIs(weights_1a, weights_1b)
        self.assertIsNot(model_1._ssp_weights_cache, model_2._ssp_weights_cache)
        self.assertEqual(len(model_1._ssp_weights_cache), 1)
        self.assertEqual(len(model_2._ssp_weights_cache), 1)
        self.assertIsNot(weights_1a, weights_2)

    def test_surviving_stellar_mass_uses_ssp_current_mass(self):
        model = LinearCEM(today=13.7 * u.Gyr)
        ssp = make_toy_ssp_for_cem()
        weights = np.array([[1.0, 2.0], [3.0, 4.0]]) << u.Msun

        with mock.patch.object(model, 'interpolate_ssp_masses', return_value=weights):
            surviving_mass = model.surviving_stellar_mass(ssp, 13.0 * u.Gyr)

        expected = np.sum(ssp.current_mass * weights)
        self.assertTrue(u.isclose(surviving_mass, expected))

    def test_supernova_rate_uses_ssp_supernova_grid(self):
        model = LinearCEM(today=13.7 * u.Gyr)
        ssp = make_toy_ssp_for_cem()
        weights = np.array([[1.0, 0.5], [0.25, 0.125]]) << u.Msun

        with mock.patch.object(model, 'interpolate_ssp_masses', return_value=weights):
            sn_rate = model.supernova_rate(ssp, 13.0 * u.Gyr)

        expected = np.sum(ssp.supernova_rate * weights)
        self.assertTrue(u.isclose(sn_rate, expected))

    def test_mean_stellar_age_linear_log_and_surviving_mass(self):
        model = LinearCEM(today=13.7 * u.Gyr)
        ssp = make_toy_ssp_for_cem()
        weights = np.array([[1.0, 3.0], [0.0, 0.0]]) << u.Msun
        surviving_weights = weights.copy() * ssp.current_mass

        with mock.patch.object(model, 'interpolate_ssp_masses', return_value=weights):
            mean_age = model.mean_stellar_age(ssp, 13.0 * u.Gyr)
            mean_log_age = model.mean_stellar_age(ssp, 13.0 * u.Gyr, log=True)
            surviving_mean_age = model.mean_stellar_age(
                ssp, 13.0 * u.Gyr, surviving_mass=True)

        expected_mean = ((1.0 * 1.0) + (3.0 * 10.0)) / 4.0 * u.Gyr
        expected_log = 10 ** ((1.0 * np.log10(1.0) + 3.0 * np.log10(10.0)) / 4.0) * u.Gyr
        expected_surviving = (
            np.sum(surviving_weights.value * ssp.ages.to_value(u.Gyr)) /
            np.sum(surviving_weights.value)
        ) * u.Gyr

        self.assertTrue(u.isclose(mean_age, expected_mean))
        self.assertTrue(u.isclose(mean_log_age, expected_log))
        self.assertTrue(u.isclose(surviving_mean_age, expected_surviving))

    def test_ionising_photon_rate_supports_species_selection(self):
        model = LinearCEM(today=13.7 * u.Gyr)
        ssp = make_toy_ssp_for_cem()
        weights = np.array([[2.0, 0.0], [0.0, 0.0]]) << u.Msun

        with mock.patch.object(model, 'interpolate_ssp_masses', return_value=weights):
            q_hi = model.ionising_photon_rate_hi(ssp, 13.0 * u.Gyr, species='HI')
            q_hei = model.ionising_photon_rate_hi(ssp, 13.0 * u.Gyr, species='HeI')
            q_heii = model.ionising_photon_rate_hi(ssp, 13.0 * u.Gyr, species='HeII')

        expected_hi = np.sum(
            ssp.log_ionising_HI_photons.to(1e40 * u.s**-1 / u.Msun) * weights
        ).to(u.dex(u.s**-1))
        expected_hei = np.sum(
            ssp.log_ionising_HeI_photons.to(1e40 * u.s**-1 / u.Msun) * weights
        ).to(u.dex(u.s**-1))
        expected_heii = np.sum(
            ssp.log_ionising_HeII_photons.to(1e40 * u.s**-1 / u.Msun) * weights
        ).to(u.dex(u.s**-1))

        self.assertTrue(u.isclose(q_hi, expected_hi))
        self.assertTrue(u.isclose(q_hei, expected_hei))
        self.assertTrue(u.isclose(q_heii, expected_heii))

        with self.assertRaises(ValueError):
            model.ionising_photon_rate_hi(ssp, 13.0 * u.Gyr, species='CIV')


if __name__ == '__main__':
    unittest.main()

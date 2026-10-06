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
        self.assertEqual(metals[0], 1e-6)
        self.assertTrue(np.isclose(metals[-1], 0.02, rtol=1e-4))

    def test_beta_cem_direct_shape_parameters(self):
        model = models.BetaCEM(
            mass_today=1.0 * u.Msun,
            alpha=2.0,
            beta=3.0,
            t_start=1.0 * u.Gyr,
            t_end=11.0 * u.Gyr,
            today=13.7 * u.Gyr,
            ism_metallicity_today=0.02,
        )

        times = np.array([0.0, 1.0, 6.0, 11.0, 13.7]) * u.Gyr
        mass = model.stellar_mass_formed(times)
        sfr = model.sfr(times)
        metals = model.ism_metallicity(times)

        self.assertEqual(model.name, "beta_cem")
        self.assertTrue(np.isclose(mass[0], 0.0 * u.Msun))
        self.assertTrue(np.isclose(mass[1], 0.0 * u.Msun))
        self.assertTrue(np.isclose(mass[-2], 1.0 * u.Msun))
        self.assertTrue(np.isclose(mass[-1], 1.0 * u.Msun))
        self.assertTrue(np.all(np.diff(mass.to_value(u.Msun)) >= 0.0))
        self.assertTrue(u.isclose(model.t_peak(), (1.0 + 10.0 / 3.0) * u.Gyr))
        self.assertTrue(np.all(sfr >= 0.0 * u.Msun / u.Gyr))
        self.assertTrue(np.isclose(sfr[0], 0.0 * u.Msun / u.Gyr))
        self.assertTrue(np.isclose(sfr[-1], 0.0 * u.Msun / u.Gyr))
        self.assertTrue(np.allclose(metals, 0.02))

    def test_beta_cem_mean_concentration_parameters(self):
        model = models.BetaCEM(
            mass_today=2.0 * u.Msun,
            t_mean=0.25,
            kappa=8.0,
            today=13.7 * u.Gyr,
            ism_metallicity_today=0.02,
        )

        mass = model.stellar_mass_formed(self.dummy_times)

        self.assertEqual(model.alpha_val, 2.0)
        self.assertEqual(model.beta_val, 6.0)
        self.assertEqual(model.kappa_val, 8.0)
        self.assertTrue(np.isclose(model.t_mean_q, 0.25 * 13.7 * u.Gyr))
        self.assertTrue(np.isclose(mass[0], 0.0 * u.Msun, rtol=1e-4))
        self.assertTrue(np.isclose(mass[-1], 2.0 * u.Msun, rtol=1e-4))

    def test_beta_zpowerlaw(self):
        model = models.BetaZPowerLawCEM(
            mass_today=1.0 * u.Msun,
            alpha=2.0,
            beta=2.0,
            today=13.7 * u.Gyr,
            ism_metallicity_today=0.02,
            alpha_powerlaw=1.0,
        )

        mass = model.stellar_mass_formed(self.dummy_times)
        metals = model.ism_metallicity(self.dummy_times)

        self.assertEqual(model.name, "beta_zpowlaw_cem")
        self.assertTrue(np.isclose(mass[0], 0.0 * u.Msun, rtol=1e-4))
        self.assertTrue(np.isclose(mass[-1], 1.0 * u.Msun, rtol=1e-4))
        self.assertTrue(np.all(np.diff(metals) >= 0.0))
        self.assertTrue(np.isclose(metals[-1], 0.02, rtol=1e-4))

    def test_beta_cem_rejects_invalid_parameterizations(self):
        with self.assertRaises(ValueError):
            _ = models.BetaCEM(mass_today=1.0 * u.Msun, alpha=0.0, beta=2.0, today=13.7)

        with self.assertRaises(ValueError):
            _ = models.BetaCEM(mass_today=1.0 * u.Msun, t_mean=0.5, today=13.7)

        with self.assertRaises(ValueError):
            _ = models.BetaCEM(
                mass_today=1.0 * u.Msun,
                t_mean=1.0,
                kappa=5.0,
                today=13.7,
            )
    
    def test_tabular(self):
        low_res_time = np.linspace(0, 13.7, 10) * u.Gyr
        masses = 1 - np.exp(-low_res_time / 3.0 / u.Gyr)
        # A smooth history on a coarse grid is reproduced to 1% only by the
        # cubic interpolation
        model = models.TabularCEM(
            times=low_res_time, masses=masses * u.Msun,
            metallicities=np.full(masses.size, fill_value=0.02),
            interpolation="pchip")

        mass = model.stellar_mass_formed(self.dummy_times)
        real_mass = 1 - np.exp(- self.dummy_times / 3 / u.Gyr)
        self.assertTrue(np.allclose(mass, real_mass * u.Msun, rtol=1e-2))

        # Default (linear): exact at the nodes, mean SFR within each interval
        model = models.TabularCEM(
            times=low_res_time, masses=masses * u.Msun,
            metallicities=np.full(masses.size, fill_value=0.02))
        self.assertEqual(model.interpolation, "linear")
        np.testing.assert_allclose(
            model.stellar_mass_formed(low_res_time).to_value(u.Msun), masses,
            rtol=1e-12, atol=1e-15)

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

    def test_average_ssfr_over_tau_uses_lookback_interval(self):
        model = LinearCEM(today=13.7 * u.Gyr)

        ssfr = model.average_ssfr_over_tau(t_obs=10.0 * u.Gyr, tau=2.0 * u.Gyr)
        expected = ((10.0 - 8.0) / 10.0 / (2.0 * u.Gyr)).to(1 / u.yr)

        self.assertTrue(u.isclose(ssfr, expected))

        with self.assertRaises(ValueError):
            _ = model.average_ssfr_over_tau(t_obs=10.0 * u.Gyr, tau=0.0 * u.Gyr)

        with self.assertRaises(ValueError):
            _ = model.average_ssfr_over_tau(t_obs=10.0 * u.Gyr, tau=11.0 * u.Gyr)

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

## Test cached interpolation

def reference_get_weights(ssp, ages, metallicities, masses=None):
    """SSPBase.get_weights before the age-grid cache (verbatim)."""
    from pst.utils import check_unit
    ages = check_unit(ages, u.Gyr)
    metallicities = check_unit(metallicities, u.dimensionless_unscaled)
    if masses is None:
        masses = np.ones(ages.size) << u.Msun
    else:
        masses = check_unit(masses, u.Msun)
    age_idx = np.clip(ssp.ages.searchsorted(ages), 1, ssp.ages.size-1)
    weights_age = np.log(ages / ssp.ages[age_idx-1])
    weights_age /= np.log(ssp.ages[age_idx] / ssp.ages[age_idx-1])
    weights_age = np.clip(weights_age, 0., 1.)
    z_idx = np.clip(ssp.metallicities.searchsorted(metallicities), 1, ssp.metallicities.size-1)
    weights_z = np.log(metallicities / ssp.metallicities[z_idx-1])
    weights_z /= np.log(ssp.metallicities[z_idx] / ssp.metallicities[z_idx-1])
    weights_z = np.clip(weights_z, 0., 1.)
    weights = np.zeros((ssp.metallicities.size, ssp.ages.size)) << masses.unit
    np.add.at(weights, (z_idx, age_idx), masses * weights_age * weights_z)
    np.add.at(weights, (z_idx-1, age_idx), masses * weights_age * (1-weights_z))
    np.add.at(weights, (z_idx-1, age_idx-1), masses * (1-weights_age) * (1-weights_z))
    np.add.at(weights, (z_idx, age_idx-1), masses * (1-weights_age) * weights_z)
    return weights


def reference_interpolate_ssp_masses(model, ssp, t_obs, oversample_factor=10):
    """ChemicalEvolutionModel.interpolate_ssp_masses before the cache (verbatim)."""
    age_bins = np.hstack(
        [0 << u.yr, np.sqrt(ssp.ages[1:] * ssp.ages[:-1]), 1e12 << u.yr])
    age_bins = age_bins[:age_bins.searchsorted(t_obs) + 1]
    age_bins[-1] = t_obs
    w1 = np.arange(oversample_factor) / oversample_factor
    age_bins = np.hstack(
        [(1-w1) * age_bins[i] + w1 * age_bins[i + 1] for i in range(age_bins.size - 1)]
        + [t_obs])
    mass = model.stellar_mass_formed(t_obs - age_bins)
    bin_mass = mass[:-1] - mass[1:]
    bin_age = (age_bins[1:] + age_bins[:-1]) / 2
    bin_metallicity = model.ism_metallicity(t_obs - bin_age)
    return reference_get_weights(ssp, bin_age, bin_metallicity, bin_mass)


def reference_tabular_stellar_mass_formed(model, times):
    """TabularCEM.stellar_mass_formed before the plain-float rewrite (verbatim)."""
    from pst.utils import check_unit
    from pst.model import Parameter
    times = times.q.to(u.Gyr) if isinstance(times, Parameter) else check_unit(times, u.Gyr)
    interpolator = cem.MassHistoryInterpolant(
        model.table_t.value, model.table_mass.value, model.interpolation)
    integral = interpolator(times.to_value(model.table_t.unit)) << model.table_mass.unit
    integral[times > model.table_t[-1]] = model.table_mass[-1]
    integral[times < model.table_t[0]] = 0
    return integral


class TestTabularStellarMassFormed(unittest.TestCase):
    """Plain-float TabularCEM.stellar_mass_formed matches the previous version."""

    def setUp(self):
        self.model = models.TabularMassFracCEM(
            mass_frac=np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99]),
            times=np.array([1.0, 3.0, 6.0, 10.0, 12.0, 13.5]) << u.Gyr,
            today=13.7 * u.Gyr, mass_today=2.0 * u.Msun,
            ism_metallicity_today=0.02, alpha_powerlaw=1.0)

    def check(self, times):
        new = self.model.stellar_mass_formed(times)
        ref = reference_tabular_stellar_mass_formed(self.model, times)
        self.assertEqual(new.unit, ref.unit)
        self.assertEqual(np.shape(new), np.shape(ref))
        np.testing.assert_allclose(new.value, ref.value, rtol=1e-14, atol=0)

    def test_arrays_including_out_of_range_times(self):
        times = np.concatenate(([-1.0, 0.0], np.linspace(0, 13.7, 500), [13.7, 20.0]))
        self.check(times << u.Gyr)
        self.check((times * 1e9) << u.yr)     # other time unit
        self.check(times)                      # bare numbers, Gyr

    def test_scalar_and_parameter_inputs(self):
        from pst.model import Parameter
        for value in (0.5, 13.7, 25.0):
            self.check(value * u.Gyr)
        self.check(Parameter(7.0, unit=u.Gyr))

    def test_gyr_fast_path_does_not_modify_the_input(self):
        times = np.linspace(0, 13.7, 50) << u.Gyr
        times.flags.writeable = False         # e.g. cached, read-only grids
        before = times.copy()
        self.model.stellar_mass_formed(times)
        self.model.ism_metallicity(times)
        np.testing.assert_array_equal(times, before)


class TestSSPAgeGridCache(unittest.TestCase):
    """The cached age grid reproduces the previous SSP weights exactly."""

    @classmethod
    def setUpClass(cls):
        cls.ssp = SSP.PopStar(IMF="cha")
        cls.today = 13.7 * u.Gyr

    def setUp(self):
        cem._SSP_AGE_GRID_CACHE.clear()

    def make_model(self, times_gyr, alpha=1.0):
        return models.TabularMassFracCEM(
            mass_frac=np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99]),
            times=np.asarray(times_gyr) << u.Gyr, today=self.today,
            mass_today=1.0 * u.Msun, ism_metallicity_today=0.02,
            alpha_powerlaw=alpha)

    def test_weights_match_previous_implementation(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            times = np.sort(rng.uniform(0.05, 13.6, 6))
            model = self.make_model(times, alpha=rng.uniform(0.1, 2.0))
            for t_obs in (self.today, 10.0 * u.Gyr):
                new = model.interpolate_ssp_masses(self.ssp, t_obs)
                ref = reference_interpolate_ssp_masses(model, self.ssp, t_obs)
                self.assertEqual(new.unit, ref.unit)
                np.testing.assert_allclose(new.to_value(u.Msun), ref.to_value(u.Msun),
                                           rtol=1e-12, atol=1e-15)

    def test_sed_matches_previous_implementation(self):
        model = self.make_model([1.0, 3.0, 6.0, 10.0, 12.0, 13.5])
        new = model.compute_SED(self.ssp, self.today)
        weights = reference_interpolate_ssp_masses(model, self.ssp, self.today)
        weights = np.where(weights > 0, weights, 0.0 << weights.unit)
        ref = np.einsum("za,zaw->w", weights.value, self.ssp.L_lambda.value)
        np.testing.assert_allclose(new.value, ref, rtol=1e-12)

    def test_sed_matches_previous_implementation_float64(self):
        # float64 SSP grid: compute_SED uses a single matrix-vector product
        import copy
        ssp = copy.deepcopy(self.ssp)
        ssp.L_lambda = self.ssp.L_lambda.astype(np.float64)
        model = self.make_model([0.5, 2.0, 5.0, 9.0, 12.5, 13.6])
        new = model.compute_SED(ssp, self.today)
        weights = reference_interpolate_ssp_masses(model, ssp, self.today)
        weights = np.where(weights > 0, weights, 0.0 << weights.unit)
        ref = np.einsum("za,zaw->w", weights.value, ssp.L_lambda.value)
        self.assertEqual(new.unit, weights.unit * ssp.L_lambda.unit)
        np.testing.assert_allclose(new.value, ref, rtol=1e-12)

    def test_mass_history_is_evaluated_once(self):
        # Z(t) = Z_today (M(t)/M_today)^alpha reuses the same mass evaluation
        model = self.make_model([1.0, 3.0, 6.0, 10.0, 12.0, 13.5])
        calls = []
        original = model.stellar_mass_formed

        def counting(times):
            calls.append(times)
            return original(times)

        model.stellar_mass_formed = counting
        weights = model.interpolate_ssp_masses(self.ssp, self.today)
        self.assertEqual(len(calls), 1)
        del model.stellar_mass_formed
        ref = reference_interpolate_ssp_masses(model, self.ssp, self.today)
        np.testing.assert_allclose(weights.value, ref.value, rtol=1e-12, atol=1e-15)

    def test_grid_is_cached_and_keyed_on_ages_and_time(self):
        model = self.make_model([1.0, 3.0, 6.0, 10.0, 12.0, 13.5])
        model.interpolate_ssp_masses(self.ssp, self.today)
        grid = cem._ssp_age_grid(self.ssp, self.today)
        self.assertEqual(len(cem._SSP_AGE_GRID_CACHE), 1)
        # Same SSP grid and time: reused
        model.interpolate_ssp_masses(self.ssp, self.today)
        self.assertIs(cem._ssp_age_grid(self.ssp, self.today), grid)
        self.assertEqual(len(cem._SSP_AGE_GRID_CACHE), 1)
        # New observing time or oversampling: new entries
        cem._ssp_age_grid(self.ssp, 10.0 * u.Gyr)
        cem._ssp_age_grid(self.ssp, self.today, oversample_factor=5)
        self.assertEqual(len(cem._SSP_AGE_GRID_CACHE), 3)
        # Cached arrays cannot be modified in place
        with self.assertRaises(ValueError):
            grid.age_bins[0] = 1 * u.yr

    def test_changed_ssp_ages_are_not_served_from_cache(self):
        ssp = make_toy_ssp_for_cem()
        model = LinearCEM(today=13.7 * u.Gyr)
        first = model.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)
        ssp.ages = np.array([2.0, 8.0]) * u.Gyr
        second = model.interpolate_ssp_masses(ssp, 12.0 * u.Gyr)
        ref = reference_interpolate_ssp_masses(model, ssp, 12.0 * u.Gyr)
        np.testing.assert_allclose(second.value, ref.value, rtol=1e-12)
        self.assertFalse(np.allclose(first.value, second.value))

    def test_get_weights_matches_previous_implementation(self):
        rng = np.random.default_rng(4)
        ages = 10 ** rng.uniform(5.5, 10.2, 300) * u.yr
        metallicities = 10 ** rng.uniform(-4.5, -1.0, 300) << u.dimensionless_unscaled
        masses = rng.uniform(0, 2, 300) * u.Msun
        for kwargs in ({"masses": masses}, {}):
            new = self.ssp.get_weights(ages, metallicities, **kwargs)
            ref = reference_get_weights(self.ssp, ages, metallicities, **kwargs)
            np.testing.assert_allclose(new.to_value(u.Msun), ref.to_value(u.Msun),
                                       rtol=1e-12, atol=1e-15)
        # Scalar inputs, as used by SSPBase.regrid
        new = self.ssp.get_weights(1.3 * u.Gyr, 0.01, 1.0 * u.Msun)
        ref = reference_get_weights(self.ssp, 1.3 * u.Gyr, 0.01, 1.0 * u.Msun)
        np.testing.assert_allclose(new.value, ref.value, rtol=1e-12, atol=1e-15)
        self.assertAlmostEqual(new.sum().to_value(u.Msun), 1.0)


def reference_compute_SED(weights, ssp, allow_negative=False, age_bin_edges=None):
    """compute_SED weight cleaning and projection before the plain-array path."""
    weights = np.where(np.isfinite(weights), weights, 0.0 << weights.unit)
    if not allow_negative:
        weights = np.where(weights > 0, weights, 0.0 << weights.unit)
    if age_bin_edges is None:
        sed = np.einsum("za,zaw->w", weights.value, ssp.L_lambda.value)
        return sed * (weights.unit * ssp.L_lambda.unit)
    idx = np.digitize(ssp.ages.value, age_bin_edges.to_value(ssp.ages.unit)
                      ).clip(0, len(age_bin_edges) - 1) - 1
    M = cem.ChemicalEvolutionModel._age_bin_matrix(None, idx, len(age_bin_edges) - 1)
    sed = np.einsum("za,zaw,an->nw", weights.value, ssp.L_lambda.value, M)
    return sed * (weights.unit * ssp.L_lambda.unit)


def reference_compute_photometry(weights, photometry, ssp, allow_negative=False,
                                 age_bin_edges=None):
    weights = np.where(np.isfinite(weights), weights, 0.0 << weights.unit)
    if not allow_negative:
        weights = np.where(weights > 0, weights, 0.0 << weights.unit)
    if age_bin_edges is None:
        out = np.einsum("bza,za->b", photometry.value, weights.value)
        return out * (photometry.unit * weights.unit)
    idx = np.digitize(ssp.ages.value, age_bin_edges.to_value(ssp.ages.unit)
                      ).clip(0, len(age_bin_edges) - 1) - 1
    M = cem.ChemicalEvolutionModel._age_bin_matrix(None, idx, len(age_bin_edges) - 1)
    out = np.einsum("bza,za,an->nb", photometry.value, weights.value, M)
    return out * (photometry.unit * weights.unit)


class TestPlainArraySynthesis(unittest.TestCase):
    """compute_SED / compute_photometry on plain arrays match the Quantity version."""

    def setUp(self):
        rng = np.random.default_rng(7)
        ssp = SSP.SSPBase()
        ssp.name = "toy_plain_array_ssp"
        ssp.ages = np.geomspace(1e6, 1.3e10, 5) << u.yr
        ssp.metallicities = np.array([0.004, 0.02, 0.05]) << u.dimensionless_unscaled
        ssp.wavelength = np.linspace(3000, 7000, 7) << u.AA
        ssp.L_lambda = rng.uniform(0.1, 2.0, (3, 5, 7)) << (u.Lsun / u.AA / u.Msun)
        self.ssp = ssp
        self.photometry = rng.uniform(0.1, 2.0, (4, 3, 5)) << (u.Jy / u.Msun)
        # Weights with every kind of value the cleaning has to handle
        weights = rng.uniform(-1.0, 3.0, (3, 5))
        weights[0, 0], weights[1, 2], weights[2, 4] = np.nan, np.inf, -np.inf
        weights[2, 1] = 0.0
        self.weights = weights << u.Msun
        self.model = LinearCEM(today=13.7 * u.Gyr)
        self.model.interpolate_ssp_masses = lambda ssp, t_obs: self.weights
        self.edges = [0.0, 1e8, 1e9, 2e10] * u.yr

    def assert_same(self, new, ref):
        self.assertEqual(new.unit, ref.unit)
        self.assertEqual(new.shape, ref.shape)
        # allow_negative can cancel terms: compare relative to the largest value
        np.testing.assert_allclose(new.value, ref.value, rtol=1e-12,
                                   atol=1e-12 * np.abs(ref.value).max())

    def test_sed_matches_quantity_implementation(self):
        for dtype in (np.float64, np.float32):
            self.ssp.L_lambda = self.ssp.L_lambda.astype(dtype)
            for allow_negative in (False, True):
                for edges in (None, self.edges):
                    with self.subTest(dtype=dtype, allow_negative=allow_negative,
                                      binned=edges is not None):
                        new = self.model.compute_SED(
                            self.ssp, 13.7 * u.Gyr, allow_negative=allow_negative,
                            age_bin_edges=edges)
                        ref = reference_compute_SED(self.weights, self.ssp,
                                                    allow_negative, edges)
                        self.assert_same(new, ref)
                        self.assertTrue(np.isfinite(new.value).all())

    def test_photometry_matches_quantity_implementation(self):
        for allow_negative in (False, True):
            for edges in (None, self.edges):
                with self.subTest(allow_negative=allow_negative,
                                  binned=edges is not None):
                    new = self.model.compute_photometry(
                        self.ssp, 13.7 * u.Gyr, photometry=self.photometry,
                        allow_negative=allow_negative, age_bin_edges=edges)
                    ref = reference_compute_photometry(
                        self.weights, self.photometry, self.ssp, allow_negative, edges)
                    self.assert_same(new, ref)

    def test_negative_weights_are_dropped_unless_allowed(self):
        self.weights = -np.ones((3, 5)) << u.Msun
        sed = self.model.compute_SED(self.ssp, 13.7 * u.Gyr)
        np.testing.assert_array_equal(sed.value, 0.0)
        sed = self.model.compute_SED(self.ssp, 13.7 * u.Gyr, allow_negative=True)
        self.assertTrue((sed.value < 0).all())

    def test_output_unit_is_cached_and_weights_not_modified(self):
        before = self.weights.copy()
        first = self.model.compute_SED(self.ssp, 13.7 * u.Gyr)
        second = self.model.compute_SED(self.ssp, 13.7 * u.Gyr)
        self.assertIs(first.unit, second.unit)
        self.assertEqual(first.unit, u.Msun * self.ssp.L_lambda.unit)
        np.testing.assert_array_equal(self.weights.value, before.value)
        # Independent output arrays
        first[0] = 0 * first.unit
        self.assertNotEqual(second.value[0], 0.0)


class TestLinearMassHistory(unittest.TestCase):
    """Default (linear) mass-history interpolation: a step SFH."""

    times = np.array([0.0, 4.0, 8.0, 11.0, 13.0, 13.69, 13.7])
    masses = np.array([0.0, 0.3, 0.8, 0.97, 0.99, 0.999, 1.0])

    def test_default_mode_is_linear(self):
        self.assertEqual(cem.MassHistoryInterpolant(self.times, self.masses).mode,
                         "linear")

    def test_masses_are_linear_between_nodes(self):
        interp = cem.MassHistoryInterpolant(self.times, self.masses)
        grid = np.linspace(0.0, 13.7, 4001)
        np.testing.assert_allclose(interp(grid), np.interp(grid, self.times, self.masses),
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(interp(self.times), self.masses, atol=1e-15)

    def test_sfr_is_the_interval_mean(self):
        interp = cem.MassHistoryInterpolant(self.times, self.masses)
        means = np.diff(self.masses) / np.diff(self.times)
        mid = 0.5 * (self.times[:-1] + self.times[1:])
        np.testing.assert_allclose(interp(mid, nu=1), means, rtol=1e-12)
        # Nodes take the younger interval, and t_obs the last interval
        np.testing.assert_allclose(interp(self.times[:-1], nu=1), means, rtol=1e-12)
        self.assertAlmostEqual(float(interp(self.times[-1], nu=1)), means[-1], places=12)
        np.testing.assert_array_equal(interp(mid, nu=2), 0.0)
        self.assertEqual(np.shape(interp(5.0, nu=1)), ())

    def test_no_structure_inside_a_long_interval(self):
        # The case that made PCHIP rise over the last Gyr: a long interval
        # followed by a very short one with a much higher SFR
        interp = cem.MassHistoryInterpolant(self.times, self.masses)
        inside = np.linspace(13.0, 13.69, 200, endpoint=False)
        np.testing.assert_allclose(interp(inside, nu=1), 0.009 / 0.69, rtol=1e-12)
        pchip = cem.MassHistoryInterpolant(self.times, self.masses, "pchip")
        sfr_pchip = pchip(inside, nu=1)
        self.assertGreater(sfr_pchip.max() / sfr_pchip.min(), 2.0)

    def test_constant_sfr_and_mass_conservation(self):
        times = np.array([0.0, 2.0, 5.0, 9.0, 12.0, 13.7])
        interp = cem.MassHistoryInterpolant(times, 0.1 * times)
        grid = np.linspace(0.0, 13.7, 1001)
        np.testing.assert_allclose(interp(grid, nu=1), 0.1, rtol=1e-12)
        # The integral of the SFR over each interval equals its mass
        rng = np.random.default_rng(3)
        masses = np.concatenate(([0.0], np.cumsum(rng.uniform(0.0, 1.0, 5))))
        interp = cem.MassHistoryInterpolant(times, masses)
        for a, b, dm in zip(times[:-1], times[1:], np.diff(masses)):
            t = np.linspace(a, b, 2001)[:-1] + 0.5 * (b - a) / 2000
            self.assertAlmostEqual(np.sum(interp(t, nu=1)) * (b - a) / 2000, dm, places=10)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            cem.MassHistoryInterpolant(self.times, self.masses, "cubic")
        with self.assertRaises(ValueError):
            cem.TabularCEM(times=self.times * u.Gyr, masses=self.masses * u.Msun,
                           metallicities=np.full(self.times.size, 0.02),
                           interpolation="cubic")

    def test_tabular_models_forward_the_mode(self):
        kwargs = dict(today=13.7 * u.Gyr, mass_today=1.0 * u.Msun,
                      ism_metallicity_today=0.02, alpha_powerlaw=1.0)
        frac = models.TabularMassFracCEM(
            mass_frac=self.masses[1:-1], times=self.times[1:-1] << u.Gyr, **kwargs)
        self.assertEqual(frac.interpolation, "linear")
        for mode in ("pchip", "linear"):
            frac = models.TabularMassFracCEM(
                mass_frac=self.masses[1:-1], times=self.times[1:-1] << u.Gyr,
                interpolation=mode, **kwargs)
            self.assertEqual(frac.interpolation, mode.lower())
            cc25 = models.CC25TabularCEM(
                tau_ssfr=np.array([1.0, 0.1]) << u.Gyr,
                ssfr=np.array([0.1, 0.1]) << 1 / u.Gyr, interpolation=mode, **kwargs)
            self.assertEqual(cc25.interpolation, mode.lower())
        # TabularCEM.sfr and stellar_mass_formed follow the mode
        sfr = frac.sfr(np.array([2.0, 12.0]) << u.Gyr).to_value(u.Msun / u.Gyr)
        np.testing.assert_allclose(sfr, [0.3 / 4.0, 0.02 / 2.0], rtol=1e-12)
        frac.interpolation = "pchip"
        self.assertNotAlmostEqual(
            frac.sfr(np.array([2.0]) << u.Gyr).to_value(u.Msun / u.Gyr)[0], 0.3 / 4.0)


class TestMassHistoryInterpolator(unittest.TestCase):
    """End condition of the PCHIP mass-history interpolation."""

    # Last interval (3% of the mass in 2.7 Gyr) is ~5x shallower than the
    # previous one: standard PCHIP clamps SFR(today) to exactly 0 here.
    times = np.array([0.0, 4.0, 8.0, 11.0, 13.7])
    masses = np.array([0.0, 0.3, 0.8, 0.97, 1.0])

    @staticmethod
    def _expected_end_slope(times, masses):
        h1, h0 = times[-2] - times[-3], times[-1] - times[-2]
        m1 = (masses[-2] - masses[-3]) / h1
        m0 = (masses[-1] - masses[-2]) / h0
        return min(m0 * (m0 / m1) ** (h0 / (h0 + h1)), 3 * m0)

    def test_declining_sfr_is_not_forced_to_zero(self):
        from scipy.interpolate import PchipInterpolator
        standard = PchipInterpolator(self.times, self.masses)(self.times[-1], nu=1)
        self.assertAlmostEqual(standard, 0.0, places=12)  # the old behaviour

        model = cem.TabularCEM(times=self.times * u.Gyr, masses=self.masses * u.Msun,
                               metallicities=np.full(self.times.size, 0.02),
                               interpolation="pchip")
        sfr_today = model.sfr(np.array([self.times[-1] - 1e-9]) * u.Gyr)[0]
        expected = self._expected_end_slope(self.times, self.masses)
        self.assertTrue(u.isclose(sfr_today, expected * u.Msun / u.Gyr, rtol=1e-6))
        # Declining trend: below the mean SFR of the last interval, but positive
        mean_last = (self.masses[-1] - self.masses[-2]) / (self.times[-1] - self.times[-2])
        self.assertGreater(sfr_today.to_value(u.Msun / u.Gyr), 0.0)
        self.assertLess(sfr_today.to_value(u.Msun / u.Gyr), mean_last)

    def test_interpolant_is_monotone_and_passes_through_nodes(self):
        rng = np.random.default_rng(1)
        grid = np.linspace(0.0, 13.7, 5001)
        for _ in range(200):
            times = np.concatenate(([0.0], np.sort(rng.uniform(0.0, 13.7, 5)), [13.7]))
            masses = np.concatenate(([0.0], np.sort(rng.uniform(0.0, 1.0, 5)), [1.0]))
            interp = cem.MassHistoryInterpolant(times, masses, "pchip")
            np.testing.assert_allclose(interp(times), masses, atol=1e-12)
            self.assertTrue(np.all(np.diff(interp(grid)) >= -1e-12))
            self.assertTrue(np.all(interp(grid, nu=1) >= -1e-12))
            self.assertGreater(interp(times[-1], nu=1), 0.0)

    def test_constant_sfr_is_reproduced(self):
        times = np.array([0.0, 2.0, 5.0, 9.0, 12.0, 13.7])
        interp = cem.MassHistoryInterpolant(times, 0.1 * times, "pchip")
        grid = np.linspace(0.0, 13.7, 1001)
        np.testing.assert_allclose(interp(grid, nu=1), 0.1, rtol=1e-10)

    def test_only_the_last_interval_changes(self):
        from scipy.interpolate import PchipInterpolator
        grid = np.linspace(0.0, self.times[-2], 2001)
        np.testing.assert_allclose(
            cem.MassHistoryInterpolant(self.times, self.masses, "pchip")(grid),
            PchipInterpolator(self.times, self.masses)(grid), rtol=1e-12, atol=1e-14)

    def test_rising_sfr_end_slope_is_limited(self):
        times = np.array([0.0, 6.0, 12.0, 13.0, 13.7])
        masses = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
        mean_last = (masses[-1] - masses[-2]) / (times[-1] - times[-2])
        slope = cem.MassHistoryInterpolant(times, masses, "pchip")(times[-1], nu=1)
        self.assertGreater(slope, mean_last)            # follows the rising trend
        self.assertLessEqual(slope, 3 * mean_last + 1e-12)


if __name__ == '__main__':
    unittest.main()

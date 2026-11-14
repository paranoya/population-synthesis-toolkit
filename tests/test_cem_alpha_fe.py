"""[alpha/Fe] support in the chemical evolution models."""
import unittest

import numpy as np
from astropy import units as u

from pst import SSP, cem, models
from pst.model import Parameter

LSUN = u.Lsun / u.AA / u.Msun


def make_alpha_ssp(available_models=None):
    """Toy SSP with two [alpha/Fe] nodes whose SEDs are easy to tell apart."""
    ssp = SSP.SSPBase()
    ssp.name = "toy_alpha_ssp"
    ssp.ages = np.array([0.1, 1.0, 5.0, 13.0]) * u.Gyr
    ssp.metallicities = np.array([0.002, 0.01, 0.02]) << u.dimensionless_unscaled
    ssp.alpha_fe = [0.0, 0.4]
    ssp.wavelength = np.array([1000., 2000., 3000.]) << u.AA
    sed = np.ones((2, 3, 4, 3))
    sed[1] *= 3.0           # alpha-enhanced SEDs are 3x brighter
    ssp.L_lambda = sed << LSUN
    if available_models is not None:
        ssp.available_models = available_models
    return ssp


def make_cem(**kwargs):
    return models.TabularMassFracCEM(
        mass_frac=np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99]),
        times=np.array([1.0, 3.0, 6.0, 10.0, 12.0, 13.5]) << u.Gyr,
        today=13.7 * u.Gyr, mass_today=1.0 * u.Msun,
        ism_metallicity_today=0.01, alpha_powerlaw=1.0, **kwargs)


class TestCEMAlphaFe(unittest.TestCase):

    def setUp(self):
        self.ssp = make_alpha_ssp()
        self.t_obs = 13.7 * u.Gyr

    def test_default_is_solar_and_not_a_parameter(self):
        model = make_cem()
        self.assertIsNone(model.alpha_fe)
        # Tabular CEMs carry SFHAlphaFeMixin, whose default history is solar
        np.testing.assert_array_equal(model.ism_alpha_fe(np.ones(3) * u.Gyr), 0.0)
        self.assertIsNone(model.alpha_fe_in_bins(np.arange(4) << u.Gyr,
                                                 np.ones(3) << u.Msun))
        self.assertIsNone(cem.ExponentialCEM(today=13.7 * u.Gyr).ism_alpha_fe(
            np.ones(3) * u.Gyr))
        self.assertNotIn("alpha_fe", model.parameters())
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        self.assertEqual(w.shape, (2, 3, 4))
        self.assertEqual(w[1].sum(), 0.0)
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.0)

    def test_constant_alpha_fe_parameter(self):
        model = make_cem(alpha_fe=0.1)
        self.assertIn("alpha_fe", model.parameters())
        ref = make_cem().interpolate_ssp_masses(self.ssp, self.t_obs).value
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        # Linear interpolation in [alpha/Fe]: 0.75 / 0.25 split per node
        np.testing.assert_allclose(w[0], 0.75 * ref[0])
        np.testing.assert_allclose(w[1], 0.25 * ref[0])
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.1)
        sed = model.compute_SED(self.ssp, self.t_obs).to_value(LSUN * u.Msun)
        np.testing.assert_allclose(sed, w.sum() * (0.75 + 0.25 * 3.0))

    def test_alpha_fe_units_and_updates(self):
        model = make_cem(alpha_fe=0.4 * u.dex)
        self.assertAlmostEqual(model.alpha_fe.q.value, 0.4)
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.4)
        model.alpha_fe.q = 0.2 << u.dimensionless_unscaled  # as a sampler would
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.2)
        model.alpha_fe = None
        self.assertNotIn("alpha_fe", model.parameters())
        with self.assertRaises(u.UnitTypeError):
            make_cem(alpha_fe=0.2 * u.Msun)

    def test_analytic_cems_accept_alpha_fe(self):
        model = cem.ExponentialCEM(today=13.7 * u.Gyr, alpha_fe=0.4)
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        self.assertEqual(w[0].sum(), 0.0)
        self.assertGreater(w[1].sum(), 0.0)

    def test_time_dependent_alpha_fe(self):
        class AlphaDecline(models.TabularMassFracCEM):
            # [alpha/Fe] = 0.4 for stars formed before 3 Gyr, 0 afterwards
            def ism_alpha_fe(self, time):
                return np.where(u.Quantity(time).to_value(u.Gyr) < 3.0, 0.4, 0.0)

        model = AlphaDecline(
            mass_frac=np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99]),
            times=np.array([1.0, 3.0, 6.0, 10.0, 12.0, 13.5]) << u.Gyr,
            today=13.7 * u.Gyr, mass_today=1.0 * u.Msun,
            ism_metallicity_today=0.01, alpha_powerlaw=1.0)
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        # Old stars (> 10.7 Gyr) are alpha-enhanced, young ones are not
        self.assertGreater(w[1, :, -1].sum(), 0.0)
        self.assertEqual(w[1, :, :2].sum(), 0.0)
        mass_early = model.stellar_mass_formed(3.0 * u.Gyr).to_value(u.Msun)
        # Up to the age bin that straddles 3 Gyr (assigned by its midpoint)
        self.assertAlmostEqual(w[1].sum(), float(np.squeeze(mass_early)),
                               delta=0.03)

    def test_missing_models_receive_no_mass(self):
        available = np.array([[True, True, True], [False, False, True]])
        ssp = make_alpha_ssp(available)
        model = make_cem(alpha_fe=0.3)
        w = model.interpolate_ssp_masses(ssp, self.t_obs).value
        self.assertEqual(np.abs(w[~available]).sum(), 0.0)
        np.testing.assert_allclose(
            w.sum(), model.interpolate_ssp_masses(self.ssp, self.t_obs).value.sum())

    def test_cache_key_includes_alpha_fe(self):
        model = make_cem(alpha_fe=0.0)
        model._cache_interp_ssp_mass = True
        w0 = model.interpolate_ssp_masses(self.ssp, self.t_obs)
        model.alpha_fe.q = 0.4 << u.dimensionless_unscaled
        w1 = model.interpolate_ssp_masses(self.ssp, self.t_obs)
        self.assertEqual(w0.value[1].sum(), 0.0)
        self.assertEqual(w1.value[0].sum(), 0.0)

    def test_2d_cem(self):
        model = cem.TabularMassFracCEM2D(
            mass_frac=np.array([0.3, 0.7]), times=np.array([2.0, 7.0]) * u.Gyr,
            today=10.0 * u.Gyr, mass_today=10.0 * u.Msun,
            ism_metallicity_today=0.02, alpha_powerlaw=0.5,
            sigma_log_metallicity=0.25, alpha_fe=0.2)
        w = model.interpolate_ssp_masses(self.ssp, 10.0 * u.Gyr).value
        np.testing.assert_allclose(w[0], w[1])
        np.testing.assert_allclose(w.sum(), 10.0)

    def test_particle_cem(self):
        model = cem.ParticleListCEM(
            time_form=np.array([1.0, 5.0, 12.0, 14.0]) * u.Gyr,
            metallicities=np.array([0.002, 0.01, 0.02, 0.02]),
            masses=np.array([1.0, 2.0, 3.0, 4.0]) * u.Msun,
            alpha_fe=np.array([0.4, 0.4, 0.0, 0.0]))
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        # The last particle has not formed yet
        np.testing.assert_allclose(w[1].sum(), 3.0)
        np.testing.assert_allclose(w[0].sum(), 3.0)
        # Scalar [alpha/Fe] for every particle
        model.alpha_fe = 0.4
        w = model.interpolate_ssp_masses(self.ssp, self.t_obs).value
        np.testing.assert_allclose(w[1].sum(), 6.0)


def reference_alpha_fe_sfh(time_edges, formed_mass, tau_alpha=1.0,
                           alpha_max=0.35, alpha_min=0.0, gamma=1.0):
    """Bin-by-bin reference implementation of ``cem.alpha_fe_from_sfh``."""
    dt = np.diff(time_edges)
    sfr = formed_mass / dt
    alpha_fe = np.full(len(dt), np.nan)
    a_acc = m_acc = 0.0
    for i, (width, psi, dm) in enumerate(zip(dt, sfr, formed_mass)):
        half = width / 2
        a_mid = (a_acc * np.exp(-half / tau_alpha)
                 + psi * tau_alpha * (1 - np.exp(-half / tau_alpha)))
        m_mid = m_acc + psi * half
        if m_mid > 0:
            f = np.clip(a_mid / m_mid, 0, 1)
            alpha_fe[i] = alpha_min + (alpha_max - alpha_min) * f**gamma
        a_acc = (a_acc * np.exp(-width / tau_alpha)
                 + psi * tau_alpha * (1 - np.exp(-width / tau_alpha)))
        m_acc += dm
    return alpha_fe


def make_alpha_cem(mass_frac, **kwargs):
    # TabularMassFracCEM includes SFHAlphaFeMixin (via TabularCEM_ZPowerLaw)
    kwargs.setdefault("alpha_fe_max", 0.35)
    return models.TabularMassFracCEM(
        mass_frac=np.asarray(mass_frac),
        times=np.array([1.0, 3.0, 6.0, 10.0, 12.0, 13.5]) << u.Gyr,
        today=13.7 * u.Gyr, mass_today=1.0 * u.Msun,
        ism_metallicity_today=0.01, alpha_powerlaw=1.0, **kwargs)


class TestAlphaFeFromSFH(unittest.TestCase):

    def test_matches_reference_loop(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            n = rng.integers(2, 300)
            edges = np.concatenate(([0.0], np.cumsum(rng.uniform(0.001, 0.5, n))))
            mass = rng.uniform(0, 1, n) * (rng.uniform(size=n) > 0.3)
            mass[0] = max(mass[0], 0.1)   # reference is undefined before M > 0
            kwargs = dict(tau_alpha=rng.uniform(0.05, 5.0),
                          alpha_max=rng.uniform(0.2, 0.5),
                          alpha_min=rng.uniform(-0.2, 0.1),
                          gamma=rng.uniform(0.3, 3.0))
            ref = reference_alpha_fe_sfh(edges, mass, **kwargs)
            new = cem.alpha_fe_from_sfh(
                edges, mass, tau_alpha_fe=kwargs["tau_alpha"],
                alpha_fe_max=kwargs["alpha_max"], alpha_fe_min=kwargs["alpha_min"],
                alpha_fe_gamma=kwargs["gamma"])
            np.testing.assert_allclose(new, ref, rtol=1e-9, atol=1e-12)

    def test_edge_cases(self):
        edges = np.linspace(0, 14, 701)
        mass = np.ones(700)
        mass[:100] = 0.0          # no star formation in the first 2 Gyr
        new = cem.alpha_fe_from_sfh(edges, mass, tau_alpha_fe=0.01)
        # No overflow for t / tau ~ 1400, and no NaN before the first stars
        self.assertTrue(np.all(np.isfinite(new)))
        np.testing.assert_allclose(new[:100], 0.35)
        ref = reference_alpha_fe_sfh(edges[100:] - edges[100], mass[100:],
                                     tau_alpha=0.01)
        np.testing.assert_allclose(new[100:], ref, rtol=1e-9)
        # Constant SFR: f = tau / t (1 - e^{-t/tau}) at late times
        new = cem.alpha_fe_from_sfh(np.linspace(0, 10, 10001), np.ones(10000),
                                    tau_alpha_fe=1.0)
        self.assertAlmostEqual(new[-1], 0.35 * 0.1 * (1 - np.exp(-10)), places=4)


class TestSFHAlphaFeMixin(unittest.TestCase):

    def setUp(self):
        self.ssp = make_alpha_ssp()
        self.t_obs = 13.7 * u.Gyr

    def test_parameters(self):
        model = make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99])
        params = model.parameters()
        for name in ("tau_alpha_fe", "alpha_fe_max", "alpha_fe_min",
                     "alpha_fe_gamma"):
            self.assertIn(name, params)
            # Bare numbers give fixed parameters
            self.assertTrue(params[name].fixed)
        self.assertNotIn("alpha_fe", params)
        self.assertEqual(model.tau_alpha_fe.q, 1.0 * u.Gyr)
        # A free Parameter stays free
        model = make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99],
                               alpha_fe_max=Parameter(0.3, fixed=False))
        self.assertIn("alpha_fe_max", model.parameter_names(include_fixed=False))
        # Constant alpha_fe and the SFH-coupled parameters are exclusive
        with self.assertRaises(ValueError):
            make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99], alpha_fe=0.2)
        with self.assertRaises(ValueError):
            make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99],
                           tau_alpha_fe=0.0).mean_stellar_alpha_fe(
                               self.ssp, self.t_obs)

    def test_tabular_default_is_constant_solar(self):
        model = make_cem()
        self.assertEqual(model.alpha_fe_max.q.value, 0.0)
        self.assertEqual(model.alpha_fe_min.q.value, 0.0)
        # No new free parameters and the solar (None) interpolation path
        self.assertEqual(
            sorted(model.parameter_names(include_fixed=False)),
            sorted(n for n in model.parameter_names(include_fixed=False)
                   if "alpha_fe" not in n))
        edges = np.linspace(0, 13.7, 11) << u.Gyr
        self.assertIsNone(model.alpha_fe_in_bins(edges, np.ones(10) << u.Msun))
        np.testing.assert_array_equal(model.ism_alpha_fe(edges), 0.0)
        # Constant enhanced history: alpha_fe_max == alpha_fe_min
        model = make_cem(alpha_fe_max=0.3, alpha_fe_min=0.3)
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.3)
        # The constant alpha_fe parameter is still available
        model = make_cem(alpha_fe=0.2)
        self.assertAlmostEqual(model.mean_stellar_alpha_fe(self.ssp, self.t_obs), 0.2)

    def test_early_formation_is_more_alpha_enhanced(self):
        early = make_alpha_cem([0.9, 0.95, 0.97, 0.98, 0.99, 0.995])
        extended = make_alpha_cem([0.05, 0.15, 0.35, 0.6, 0.75, 0.95])
        a_early = early.mean_stellar_alpha_fe(self.ssp, self.t_obs)
        a_ext = extended.mean_stellar_alpha_fe(self.ssp, self.t_obs)
        self.assertGreater(a_early, a_ext)
        self.assertTrue(0.0 <= a_ext < a_early <= 0.35)

    def test_weights_follow_the_sfh_bins(self):
        model = make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99],
                               tau_alpha_fe=0.5, alpha_fe_gamma=0.5)
        grid = cem._ssp_age_grid(self.ssp, self.t_obs)
        mass = model.stellar_mass_formed(grid.formation_times)
        bin_mass = (mass[:-1] - mass[1:])[::-1]
        alpha = model.alpha_fe_in_bins(grid.formation_times[::-1], bin_mass)
        expected = np.sum(bin_mass.value * alpha) / np.sum(bin_mass.value)
        self.assertAlmostEqual(
            model.mean_stellar_alpha_fe(self.ssp, self.t_obs), expected, places=10)
        # ism_alpha_fe (fine uniform grid) agrees with the SFH-bin values
        mid = 0.5 * (grid.formation_times[::-1][1:] + grid.formation_times[::-1][:-1])
        np.testing.assert_allclose(model.ism_alpha_fe(mid)[50:], alpha[50:],
                                   atol=0.01)

    def test_cache_key_includes_mixin_parameters(self):
        model = make_alpha_cem([0.3, 0.5, 0.75, 0.9, 0.95, 0.99])
        model._cache_interp_ssp_mass = True
        a0 = model.mean_stellar_alpha_fe(self.ssp, self.t_obs)
        model.alpha_fe_max.q = 0.2 << u.dimensionless_unscaled
        a1 = model.mean_stellar_alpha_fe(self.ssp, self.t_obs)
        self.assertLess(a1, a0)

    def test_2d_cem(self):
        model = cem.TabularMassFracCEM2D(
            mass_frac=np.array([0.3, 0.7]), times=np.array([2.0, 7.0]) * u.Gyr,
            today=10.0 * u.Gyr, mass_today=10.0 * u.Msun,
            ism_metallicity_today=0.02, alpha_powerlaw=0.5,
            sigma_log_metallicity=0.25, alpha_fe_max=0.35)
        w = model.interpolate_ssp_masses(self.ssp, 10.0 * u.Gyr).value
        np.testing.assert_allclose(w.sum(), 10.0)
        self.assertGreater(w[1].sum(), 0.0)
        self.assertGreater(w[0].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()

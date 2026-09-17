import unittest

import numpy as np
from astropy import units as u

from pst import SSP, cem


def make_toy_ssp():
    ssp = SSP.SSPBase()
    ssp.name = "toy_cem_2d_ssp"
    ssp.ages = np.array([0.1, 1.0, 5.0, 10.0]) * u.Gyr
    ssp.metallicities = (
        np.array([0.002, 0.005, 0.01, 0.02, 0.05])
        << u.dimensionless_unscaled
    )
    return ssp


def make_model(sigma_log_metallicity):
    return cem.TabularMassFracCEM2D(
        mass_frac=np.array([0.3, 0.7]),
        times=np.array([2.0, 7.0]) * u.Gyr,
        today=10.0 * u.Gyr,
        mass_today=10.0 * u.Msun,
        ism_metallicity_today=0.02,
        alpha_powerlaw=0.5,
        sigma_log_metallicity=sigma_log_metallicity,
    )


class TestChemicalEvolutionModel2D(unittest.TestCase):
    def test_model_is_2d_cem_and_exposes_sigma_parameter(self):
        model = make_model(0.25)

        self.assertIsInstance(model, cem.ChemicalEvolutionModel2D)
        self.assertIn(
            "sigma_log_metallicity",
            model.parameters_recursive(include_fixed=True),
        )
        self.assertEqual(model.name, "tabular_mass_frac_cem_2d")

    def test_lognormal_distribution_is_normalized_and_preserves_mean(self):
        model = make_model(0.25)
        edges = np.geomspace(1e-8, 1.0, 4001)

        probabilities = model.metallicity_distribution(10.0 * u.Gyr, edges)
        bin_centres = np.sqrt(edges[:-1] * edges[1:])
        recovered_mean = np.sum(probabilities * bin_centres)

        self.assertEqual(probabilities.shape, (edges.size - 1,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=12)
        self.assertTrue(np.isclose(recovered_mean, 0.02, rtol=1e-4))

    def test_joint_mass_weights_conserve_each_time_bin(self):
        model = make_model(0.25)
        time_edges = np.array([0.0, 2.0, 7.0, 10.0]) * u.Gyr
        metallicity_edges = np.geomspace(1e-4, 0.1, 17)

        joint_mass = model.joint_mass_weights(time_edges, metallicity_edges)
        expected_mass = np.diff(model.stellar_mass_formed(time_edges))

        self.assertEqual(joint_mass.shape, (3, 16))
        self.assertTrue(u.allclose(joint_mass.sum(axis=1), expected_mass))

    def test_mass_differences_tolerate_only_roundoff_sized_negatives(self):
        epsilon = np.finfo(float).eps
        cumulative = (
            np.array([0.0, 0.5, 0.5 - 0.5 * epsilon, 1.0]) * u.Msun
        )

        differences = cem.ChemicalEvolutionModel2D._validated_mass_differences(
            cumulative
        )

        self.assertTrue(np.all(differences >= 0.0 * u.Msun))
        self.assertEqual(differences[1], 0.0 * u.Msun)

        genuinely_decreasing = np.array([0.0, 0.5, 0.49, 1.0]) * u.Msun
        with self.assertRaises(ValueError):
            cem.ChemicalEvolutionModel2D._validated_mass_differences(
                genuinely_decreasing
            )

    def test_zero_scatter_exactly_recovers_deterministic_model(self):
        common = dict(
            mass_frac=np.array([0.3, 0.7]),
            times=np.array([2.0, 7.0]) * u.Gyr,
            today=10.0 * u.Gyr,
            mass_today=10.0 * u.Msun,
            ism_metallicity_today=0.02,
            alpha_powerlaw=0.5,
        )
        deterministic = cem.TabularMassFracCEM(**common)
        model_2d = cem.TabularMassFracCEM2D(
            **common, sigma_log_metallicity=0.0
        )
        ssp = make_toy_ssp()

        expected = deterministic.interpolate_ssp_masses(
            ssp, 10.0 * u.Gyr, oversample_factor=3
        )
        actual = model_2d.interpolate_ssp_masses(
            ssp, 10.0 * u.Gyr, oversample_factor=3
        )

        self.assertTrue(u.allclose(actual, expected, rtol=0.0, atol=0.0 * u.Msun))

    def test_positive_scatter_converges_to_deterministic_model(self):
        common = dict(
            mass_frac=np.array([0.3, 0.7]),
            times=np.array([2.0, 7.0]) * u.Gyr,
            today=10.0 * u.Gyr,
            mass_today=10.0 * u.Msun,
            ism_metallicity_today=0.02,
            alpha_powerlaw=0.5,
        )
        deterministic = cem.TabularMassFracCEM(**common)
        model_2d = cem.TabularMassFracCEM2D(
            **common, sigma_log_metallicity=1e-8
        )
        ssp = make_toy_ssp()

        expected = deterministic.interpolate_ssp_masses(
            ssp, 10.0 * u.Gyr, oversample_factor=3
        )
        actual = model_2d.interpolate_ssp_masses(
            ssp, 10.0 * u.Gyr, oversample_factor=3
        )

        self.assertTrue(
            u.allclose(actual, expected, rtol=1e-7, atol=1e-12 * u.Msun)
        )

    def test_interpolation_weights_match_numerical_quadrature(self):
        model = make_model(0.25)
        ssp = make_toy_ssp()
        actual = model.metallicity_interpolation_weights(
            10.0 * u.Gyr, ssp.metallicities
        )

        sigma = model.sigma_log_metallicity.to_value()
        mean = model.ism_metallicity(10.0 * u.Gyr).to_value()
        location = np.log10(mean) - 0.5 * np.log(10.0) * sigma**2
        log_nodes = np.log10(ssp.metallicities.value)
        log_metallicity = np.linspace(
            location - 10.0 * sigma,
            location + 10.0 * sigma,
            200001,
        )
        density = (
            np.exp(-0.5 * ((log_metallicity - location) / sigma) ** 2)
            / (sigma * np.sqrt(2.0 * np.pi))
        )
        basis = np.zeros((log_metallicity.size, log_nodes.size))
        upper = np.searchsorted(log_nodes, log_metallicity)
        upper = np.clip(upper, 1, log_nodes.size - 1)
        lower = upper - 1
        fraction = (
            (log_metallicity - log_nodes[lower])
            / (log_nodes[upper] - log_nodes[lower])
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        rows = np.arange(log_metallicity.size)
        basis[rows, lower] = 1.0 - fraction
        basis[rows, upper] += fraction
        expected = np.trapezoid(
            density[:, np.newaxis] * basis,
            log_metallicity,
            axis=0,
        )

        self.assertTrue(np.allclose(actual, expected, rtol=2e-6, atol=1e-9))
        self.assertTrue(np.all(actual >= 0.0))
        self.assertAlmostEqual(actual.sum(), 1.0, places=12)

    def test_finite_scatter_spreads_mass_and_conserves_total(self):
        model = make_model(0.25)
        ssp = make_toy_ssp()

        weights = model.interpolate_ssp_masses(ssp, 10.0 * u.Gyr)
        mass_per_metallicity = weights.sum(axis=1)

        self.assertTrue(u.isclose(weights.sum(), 10.0 * u.Msun))
        self.assertGreater(np.count_nonzero(mass_per_metallicity.value), 2)

        expected_mean = (
            np.sum(ssp.metallicities[:, np.newaxis] * weights)
            / np.sum(weights)
        )
        self.assertTrue(
            u.isclose(model.mean_stellar_metallicity(ssp, 10.0 * u.Gyr), expected_mean)
        )

    def test_rejects_invalid_scatter_and_bin_edges(self):
        with self.assertRaises(ValueError):
            make_model(-0.1)

        model = make_model(0.25)
        with self.assertRaises(ValueError):
            model.metallicity_distribution(
                np.array([1.0, 2.0]) * u.Gyr,
                np.array([0.01, 0.005, 0.02]),
            )


if __name__ == "__main__":
    unittest.main()

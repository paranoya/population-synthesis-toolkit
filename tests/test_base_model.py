import unittest
import numpy as np
from astropy import units as u

# Adjust this import to match your package layout, e.g.
# from pst.models.base import Parameter, ModelBase
from pst.model import Parameter, ModelBase


class TestParameter(unittest.TestCase):
    def test_init_from_number_with_unit(self):
        p = Parameter(2.5, unit=u.mag)
        self.assertIsInstance(p.q, u.Quantity)
        self.assertEqual(p.q.unit, u.mag)
        self.assertAlmostEqual(p.q.value, 2.5)

    def test_init_from_quantity_sets_unit(self):
        p = Parameter(3.0 * u.AA)
        self.assertEqual(p.unit, u.AA)
        self.assertEqual(p.q.unit, u.AA)
        self.assertAlmostEqual(p.q.value, 3.0)

    def test_to_and_to_value(self):
        p = Parameter(5000.0, unit=u.AA)
        q_um = p.to(u.um)
        self.assertTrue(u.isclose(q_um, (5000.0 * u.AA).to(u.um)))
        self.assertAlmostEqual(p.to_value(u.AA), 5000.0)

    def test_convert_to_inplace(self):
        p = Parameter(5000.0, unit=u.AA)
        p.convert_to(u.um)
        self.assertEqual(p.q.unit, u.um)
        self.assertTrue(u.isclose(p.q, (5000.0 * u.AA).to(u.um)))

    def test_convert_to_raises_if_fixed(self):
        p = Parameter(1.0, unit=u.mag, fixed=True)
        with self.assertRaises(RuntimeError):
            p.convert_to(u.dimensionless_unscaled)

    def test_set_numeric_preserves_unit(self):
        p = Parameter(1.0, unit=u.mag)
        p.set(2.0)
        self.assertEqual(p.q.unit, u.mag)
        self.assertAlmostEqual(p.q.value, 2.0)

    def test_set_quantity_updates_unit(self):
        p = Parameter(1.0, unit=u.mag)
        p.set(0.5 * u.mag)
        self.assertEqual(p.q.unit, u.mag)
        self.assertAlmostEqual(p.q.value, 0.5)

    def test_set_raises_if_fixed(self):
        p = Parameter(1.0, unit=u.mag, fixed=True)
        with self.assertRaises(RuntimeError):
            p.set(2.0)

    def test_set_validate_range_dimensionless(self):
        p = Parameter(0.5, unit=u.dimensionless_unscaled, vrange=(0.0, 1.0))
        p.set(0.9, validate=True)
        with self.assertRaises(ValueError):
            p.set(1.1, validate=True)

    def test_set_validate_range_with_units(self):
        p = Parameter(1.0, unit=u.mag, vrange=(0.0 * u.mag, 2.0 * u.mag))
        p.set(1.5 * u.mag, validate=True)
        with self.assertRaises(ValueError):
            p.set(2.5 * u.mag, validate=True)

    def test_array_protocol(self):
        p = Parameter([1, 2, 3], unit=u.dimensionless_unscaled)
        arr = np.asarray(p)
        self.assertTrue(np.all(arr == np.array([1, 2, 3])))

    def test_numpy_ufunc_returns_quantity(self):
        p = Parameter(2.0, unit=u.mag)
        out = np.add(p, 1.0 * u.mag)
        self.assertIsInstance(out, u.Quantity)
        self.assertTrue(u.isclose(out, 3.0 * u.mag))

    def test_binary_ops_return_quantity(self):
        p = Parameter(2.0, unit=u.mag)
        q = p + (1.0 * u.mag)
        self.assertIsInstance(q, u.Quantity)
        self.assertTrue(u.isclose(q, 3.0 * u.mag))


class ExampleModel(ModelBase):
    def __init__(self):
        self.name = "example"
        self.a_v = Parameter(0.2, unit=u.mag, vrange=(0.0 * u.mag, 5.0 * u.mag), doc="V band attenuation")
        self.r_v = Parameter(3.1, vrange=(2.0, 6.0), fixed=True, doc="Total to selective ratio")


class TestModelBase(unittest.TestCase):
    def test_parameters_discovery(self):
        m = ExampleModel()
        ps = m.parameters()
        self.assertIn("a_v", ps)
        self.assertIn("r_v", ps)
        self.assertIsInstance(ps["a_v"], Parameter)

    def test_parameter_names_include_fixed(self):
        m = ExampleModel()
        names = m.parameter_names(include_fixed=True)
        self.assertCountEqual(names, ["a_v", "r_v"])

    def test_parameter_names_exclude_fixed(self):
        m = ExampleModel()
        names = m.parameter_names(include_fixed=False)
        self.assertEqual(names, ["a_v"])

    def test_get_unknown_raises(self):
        m = ExampleModel()
        with self.assertRaises(KeyError):
            m.get("does_not_exist")

    def test_set_values_updates_free_param(self):
        m = ExampleModel()
        m.set_values({"a_v": 1.0}, validate=True)
        self.assertTrue(u.isclose(m.a_v.q, 1.0 * u.mag))

    def test_set_values_raises_on_fixed_param(self):
        m = ExampleModel()
        with self.assertRaises(RuntimeError):
            m.set_values({"r_v": 4.0}, validate=True)

    def test_set_values_strict_unknown_key_raises(self):
        m = ExampleModel()
        with self.assertRaises(KeyError):
            m.set_values({"unknown": 1.0}, strict=True)

    def test_set_values_non_strict_ignores_unknown(self):
        m = ExampleModel()
        m.set_values({"unknown": 1.0, "a_v": 0.9}, strict=False)
        self.assertTrue(u.isclose(m.a_v.q, 0.9 * u.mag))

    def test_freeze_and_unfreeze(self):
        m = ExampleModel()
        self.assertFalse(m.a_v.fixed)
        m.freeze(["a_v"])
        self.assertTrue(m.a_v.fixed)
        m.unfreeze(["a_v"])
        self.assertFalse(m.a_v.fixed)

    def test_search_by_name(self):
        m = ExampleModel()
        hits = m.search("a_v", in_names=True, in_docs=False)
        self.assertEqual(hits, ["a_v"])

    def test_search_by_doc(self):
        m = ExampleModel()
        hits = m.search("atten", in_names=False, in_docs=True)
        self.assertEqual(hits, ["a_v"])

    def test_to_dict_structure(self):
        m = ExampleModel()
        d = m.to_dict(include_fixed=True)
        self.assertIn("name", d)
        self.assertIn("parameters", d)
        self.assertIn("a_v", d["parameters"])
        self.assertIn("r_v", d["parameters"])
        self.assertEqual(d["name"], "example")

    def test_to_dict_excludes_fixed(self):
        m = ExampleModel()
        d = m.to_dict(include_fixed=False)
        self.assertIn("a_v", d["parameters"])
        self.assertNotIn("r_v", d["parameters"])


if __name__ == "__main__":
    unittest.main()


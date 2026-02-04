from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Set
import numpy as np
from astropy import units as u

Number = Union[int, float, np.number]


@dataclass
class Parameter:
    """
    A model parameter that behaves like an astropy Quantity.

    """
    value: Union[Number, u.Quantity]
    unit: Optional[u.Unit] = None
    vrange: Optional[Tuple[Union[Number, u.Quantity], Union[Number, u.Quantity]]] = None
    fixed: bool = False
    doc: str = ""

    def __post_init__(self):
        # Normalize to an internal Quantity
        if isinstance(self.value, u.Quantity):
            self._q = self.value
            if self.unit is None:
                self.unit = self._q.unit
        else:
            self._q = (self.value << (self.unit or u.dimensionless_unscaled))
            if self.unit is None:
                self.unit = self._q.unit

    # --- Quantity-like accessors ------------------------------------------------

    @property
    def q(self) -> u.Quantity:
        """Return the underlying Quantity."""
        return self._q

    @q.setter
    def q(self, value):
        if not isinstance(self.value, u.Quantity):
            raise ValueError("Input value must be a quantity")
        elif not value.unit.is_equivalent(self._q.unit):
            raise ValueError(f"Input quantity ({value.unit}) must be equivalent to current units ({self._q.unit})")
        self._q = value

    @property
    def value_raw(self):
        """Raw numeric value of the underlying Quantity (unit-stripped)."""
        return self._q.value

    @property
    def unit_raw(self) -> u.Unit:
        """Unit of the underlying Quantity."""
        return self._q.unit

    def to(self, unit: u.Unit, equivalencies=None) -> u.Quantity:
        """Return a converted Quantity (does not modify the Parameter)."""
        return self._q.to(unit, equivalencies=equivalencies)

    def to_value(self, unit: Optional[u.Unit] = None, equivalencies=None):
        """Return numeric value in the requested unit."""
        if unit is None:
            return self._q.value
        return self._q.to_value(unit, equivalencies=equivalencies)

    # Optional: in-place conversion
    def convert_to(self, unit: u.Unit, equivalencies=None) -> "Parameter":
        """Convert the parameter in place and return self."""
        if self.fixed:
            raise RuntimeError("Parameter is fixed and cannot be modified.")
        self._q = self._q.to(unit, equivalencies=equivalencies)
        self.unit = self._q.unit
        return self

    # --- Set and validate -------------------------------------------------------

    def set(self, new_value: Union[Number, u.Quantity], *, validate: bool = True) -> None:
        if self.fixed:
            raise RuntimeError("Parameter is fixed and cannot be modified.")

        if isinstance(new_value, u.Quantity):
            q = new_value
        else:
            q = new_value << (self.unit or u.dimensionless_unscaled)

        if validate and self.vrange is not None:
            vmin, vmax = self.vrange
            # Compare with units safely
            if isinstance(vmin, u.Quantity) or isinstance(vmax, u.Quantity) or isinstance(q, u.Quantity):
                vmin_q = vmin if isinstance(vmin, u.Quantity) else (vmin << q.unit)
                vmax_q = vmax if isinstance(vmax, u.Quantity) else (vmax << q.unit)
                if np.any(q < vmin_q) or np.any(q > vmax_q):
                    raise ValueError(f"Value {q} outside allowed range [{vmin_q}, {vmax_q}].")
            else:
                if float(q) < float(vmin) or float(q) > float(vmax):
                    raise ValueError(f"Value {q} outside allowed range [{vmin}, {vmax}].")

        self._q = q

    # --- Make it act like a Quantity / ndarray ---------------------------------

    def __repr__(self) -> str:
        meta = []
        if self.fixed:
            meta.append("fixed")
        if self.vrange is not None:
            meta.append(f"range={self.vrange}")
        meta_str = (", " + ", ".join(meta)) if meta else ""
        return f"Parameter({self._q!r}{meta_str})"

    def __float__(self) -> float:
        # Only valid if dimensionless or unit is compatible with float conversion expectation
        return float(self._q.to_value(self.unit_raw))

    def __array__(self, dtype=None):
        # Allows np.asarray(Parameter) to work
        arr = np.asarray(self._q.value)
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __array_priority__(self):
        # Encourage numpy to use our __array_ufunc__
        return 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept numpy ufuncs and delegate to Quantity."""
        # Replace Parameters with underlying Quantities
        q_inputs = [x._q if isinstance(x, Parameter) else x for x in inputs]

        # Let astropy handle unit-aware ufuncs when possible
        # For many ufuncs, Quantity implements __array_ufunc__ and will return Quantity.
        result = getattr(ufunc, method)(*q_inputs, **kwargs)

        return result

    # --- Arithmetic delegation (binary ops) ------------------------------------

    def _binop(self, other: Any, op):
        other_q = other._q if isinstance(other, Parameter) else other
        return op(self._q, other_q)

    def __add__(self, other): return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binop(other, lambda a, b: b + a)
    def __sub__(self, other): return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binop(other, lambda a, b: b - a)
    def __mul__(self, other): return self._binop(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binop(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._binop(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binop(other, lambda a, b: b / a)
    def __pow__(self, other): return self._binop(other, lambda a, b: a ** b)


@dataclass
class ParameterPack:
    """
    A stable view of a set of Parameters for sampling.

    Attributes
    ----------
    names : list of str
        Dotted parameter paths in fixed order.
    params : list of Parameter
        References to the live Parameter objects.
    units : list of astropy.units.Unit or None
        Preferred units for each parameter when mapping to vectors.
    """
    names: List[str]
    params: List["Parameter"]
    units: List[Optional[u.Unit]]

    def as_dict(self, *, as_quantity: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, p in zip(self.names, self.params):
            out[name] = p.q if as_quantity else p.to_value(p.unit_raw)
        return out

    def to_vector(self) -> np.ndarray:
        """
        Return a numeric vector in the pack's units.
        """
        theta = np.empty(len(self.params), dtype=float)
        for i, (p, unit) in enumerate(zip(self.params, self.units)):
            if unit is None:
                theta[i] = float(p.to_value())
            else:
                theta[i] = float(p.to_value(unit))
        return theta

    def update_from_vector(self, theta: Sequence[float], *, validate: bool = True) -> None:
        """
        Update the live Parameters from a numeric vector.

        Parameters
        ----------
        theta : sequence of float
            New values in the pack's units.
        validate : bool, optional
            If True, run Parameter.set validation.
        """
        if len(theta) != len(self.params):
            raise ValueError(f"theta has length {len(theta)} but expected {len(self.params)}")

        for x, p, unit in zip(theta, self.params, self.units):
            if unit is None:
                p.set(float(x), validate=validate)
            else:
                p.set(float(x) << unit, validate=validate)

    def update_from_dict(self, values: Mapping[str, Any], *, validate: bool = True, strict: bool = True) -> None:
        """
        Update the live Parameters from a mapping of path to value.
        """
        name_to_idx = {n: i for i, n in enumerate(self.names)}
        for k, v in values.items():
            if k not in name_to_idx:
                if strict:
                    raise KeyError(f"Unknown parameter path '{k}'")
                continue
            self.params[name_to_idx[k]].set(v, validate=validate)


class ModelBase:
    """
    Base class for models with named parameters.

    Subclasses typically declare Parameter attributes directly, for example:

    class MyModel(ModelBase):
        name: str = "my_model"
        a_v: Parameter = Parameter(0.2, vrange=(0.0, 5.0))
        r_v: Parameter = Parameter(3.1, vrange=(2.0, 6.0), fixed=True)

    Notes
    -----
    This base class provides:
    - discovery of Parameter attributes
    - getting and setting values by name
    - filtering fixed or free parameters
    - conversion to plain dictionaries for IO
    """

    name: str = "model"

    def models_recursive(
        self,
        *,
        prefix: str = "",
        max_depth: Optional[int] = None,
    ) -> Dict[str, "ModelBase"]:
        """
        Return nested models including self.

        Returns
        -------
        models : dict
            Mapping from dotted path to model instance.
        """
        out: Dict[str, ModelBase] = {}
        visited: Set[int] = set()

        def _walk(obj: ModelBase, base: str, depth: int) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            if max_depth is not None and depth > max_depth:
                return

            out[base or obj.name or "model"] = obj

            for attr_name, attr_val in obj.__dict__.items():
                if attr_name.startswith("_"):
                    continue
                if isinstance(attr_val, ModelBase):
                    child_base = f"{base}.{attr_name}" if base else attr_name
                    _walk(attr_val, child_base, depth + 1)

        _walk(self, prefix.strip("."), 0)
        return out

    def parameters_recursive(
        self,
        *,
        prefix: str = "",
        max_depth: Optional[int] = None,
        include_fixed: bool = True,
    ) -> Dict[str, "Parameter"]:
        """
        Return parameters from this model and nested models.

        Returns
        -------
        params : dict
            Mapping from dotted parameter path to Parameter.
        """
        params: Dict[str, Parameter] = {}
        visited: Set[int] = set()

        def _walk(obj: ModelBase, base: str, depth: int) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            if max_depth is not None and depth > max_depth:
                return

            for pname, p in obj.parameters().items():
                if (not include_fixed) and p.fixed:
                    continue
                key = f"{base}.{pname}" if base else pname
                params[key] = p

            for attr_name, attr_val in obj.__dict__.items():
                if attr_name.startswith("_"):
                    continue
                if isinstance(attr_val, ModelBase):
                    child_base = f"{base}.{attr_name}" if base else attr_name
                    _walk(attr_val, child_base, depth + 1)

        _walk(self, prefix.strip("."), 0)
        return params

    def parameters(self) -> Dict[str, Parameter]:
        out: Dict[str, Parameter] = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            try:
                v = getattr(self, name)
            except Exception:
                continue
            if isinstance(v, Parameter):
                out[name] = v
        return out

    def parameter_names(self, *, include_fixed: bool = True) -> List[str]:
        """
        Return parameter names.

        Parameters
        ----------
        include_fixed : bool, optional
            If False, only return free parameters.

        Returns
        -------
        names : list of str
        """
        ps = self.parameters()
        if include_fixed:
            return list(ps.keys())
        return [k for k, p in ps.items() if not p.fixed]

    def get(self, name: str) -> Parameter:
        """
        Get a Parameter object by name.

        Raises
        ------
        KeyError
            If the parameter does not exist.
        """
        p = getattr(self, name, None)
        if not isinstance(p, Parameter):
            raise KeyError(f"Unknown parameter '{name}'.")
        return p

    def get_values(self, *, include_fixed: bool = True, as_quantity: bool = False) -> Dict[str, Any]:
        """
        Get parameter values as a dict.

        Parameters
        ----------
        include_fixed : bool, optional
            If False, only returns free parameters.
        as_quantity : bool, optional
            If True, returns values as Quantities when possible.

        Returns
        -------
        values : dict
            Mapping from parameter name to value.
        """
        ps = self.parameters()
        out: Dict[str, Any] = {}
        for k, p in ps.items():
            if (not include_fixed) and p.fixed:
                continue
            out[k] = p.as_quantity() if as_quantity else p.value
        return out

    def set_values(self, values: Mapping[str, Any], *, validate: bool = True, strict: bool = True) -> None:
        """
        Set parameter values from a mapping.

        Parameters
        ----------
        values : mapping
            Mapping from parameter name to new value.
        validate : bool, optional
            If True, checks fixed and vrange constraints. Default is True.
        strict : bool, optional
            If True, unknown keys raise KeyError. If False, unknown keys are ignored.
        """
        for k, v in values.items():
            if not hasattr(self, k) or not isinstance(getattr(self, k), Parameter):
                if strict:
                    raise KeyError(f"Unknown parameter '{k}'.")
                continue
            self.get(k).set(v, validate=validate)

    def freeze(self, names: Optional[Sequence[str]] = None) -> None:
        """
        Freeze parameters.

        Parameters
        ----------
        names : sequence of str or None
            If None, freeze all parameters. Otherwise freeze selected ones.
        """
        ps = self.parameters()
        if names is None:
            for p in ps.values():
                p.fixed = True
            return
        for n in names:
            self.get(n).fixed = True

    def unfreeze(self, names: Optional[Sequence[str]] = None) -> None:
        """
        Unfreeze parameters.

        Parameters
        ----------
        names : sequence of str or None
            If None, unfreeze all parameters. Otherwise unfreeze selected ones.
        """
        ps = self.parameters()
        if names is None:
            for p in ps.values():
                p.fixed = False
            return
        for n in names:
            self.get(n).fixed = False

    def search(self, text: str, *, in_docs: bool = True, in_names: bool = True) -> List[str]:
        """
        Search parameters by substring.

        Parameters
        ----------
        text : str
            Substring to search for, case-insensitive.
        in_docs : bool, optional
            If True, search in Parameter.doc strings.
        in_names : bool, optional
            If True, search in parameter names.

        Returns
        -------
        matches : list of str
            Matching parameter names.
        """
        q = text.lower().strip()
        matches: List[str] = []
        for name, p in self.parameters().items():
            hit = False
            if in_names and q in name.lower():
                hit = True
            if in_docs and p.doc and q in p.doc.lower():
                hit = True
            if hit:
                matches.append(name)
        return matches

    def to_dict(self, *, include_fixed: bool = True) -> Dict[str, Any]:
        """
        Serialize the model configuration to a plain dict.

        Notes
        -----
        This returns a simple structure suitable for JSON/YAML. Quantities are
        represented as dicts with value and unit.
        """
        out: Dict[str, Any] = {"name": self.name, "parameters": {}}
        for k, p in self.parameters().items():
            if (not include_fixed) and p.fixed:
                continue
            val = p.value
            if isinstance(val, u.Quantity):
                val_repr: Any = {"value": float(val.value), "unit": str(val.unit)}
            else:
                val_repr = val
            vr = p.vrange
            if vr is not None:
                vmin, vmax = vr
                if isinstance(vmin, u.Quantity) or isinstance(vmax, u.Quantity):
                    vmin_r = {"value": float(vmin.to_value(vmin.unit)), "unit": str(vmin.unit)} if isinstance(vmin, u.Quantity) else vmin
                    vmax_r = {"value": float(vmax.to_value(vmax.unit)), "unit": str(vmax.unit)} if isinstance(vmax, u.Quantity) else vmax
                    vr_repr = [vmin_r, vmax_r]
                else:
                    vr_repr = [vmin, vmax]
            else:
                vr_repr = None

            out["parameters"][k] = {
                "value": val_repr,
                "vrange": vr_repr,
                "fixed": bool(p.fixed),
                "unit": (str(p.unit) if p.unit is not None else None),
                "doc": p.doc,
            }
        return out


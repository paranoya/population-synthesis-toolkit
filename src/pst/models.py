import warnings

warnings.warn(
    "pst.models module is deprecated; use pst.cem instead",
    DeprecationWarning,
    stacklevel=2,
)

from pst.cem import *

from . import SSP
from . import observables
from . import models
from . import fitting_module
from ._version import get_versions

__version__ = get_versions()["version"]

__all__ = ["SSP", "observables", "models", "fitting_module"]


from . import _version
__version__ = _version.get_versions()['version']

.. _api:

API Reference
=============

The PST public API is organized by functionality. The sections below follow the
current package layout under ``pst`` and replace the older documentation layout
that grouped most classes under ``pst.models``.

.. _ssp_api:

Simple Stellar Population Libraries
-----------------------------------

.. automodule:: pst.SSP
   :members:
   :undoc-members:
   :show-inheritance:

.. _model_api:

Parameterized Model Infrastructure
----------------------------------

.. automodule:: pst.model
   :members:
   :undoc-members:
   :show-inheritance:

.. _cem_api:

Chemical Evolution Models (CEM)
-------------------------------

.. automodule:: pst.cem
   :members:
   :undoc-members:
   :show-inheritance:

.. _sed_api:

Spectral Energy Distribution (SED) Components
---------------------------------------------

.. automodule:: pst.sed
   :members:
   :undoc-members:
   :show-inheritance:

.. _galaxy_api:

Galaxy Assembly and Composite SEDs
----------------------------------

.. automodule:: pst.galaxy
   :members:
   :undoc-members:
   :show-inheritance:

.. _dust_api:

Dust Attenuation and Emission
-----------------------------

.. automodule:: pst.dust
   :members:
   :undoc-members:
   :show-inheritance:

.. _observables_api:

Observables, Filters, and Spectral Indices
------------------------------------------

.. automodule:: pst.observables
   :members:
   :undoc-members:
   :show-inheritance:

.. _utils_api:

Utilities
---------

.. automodule:: pst.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. _legacy_models_api:

Legacy Compatibility
--------------------

.. warning::

   ``pst.models`` is retained as a compatibility layer and is deprecated.
   New code should import CEM classes from :mod:`pst.cem` instead.

.. automodule:: pst.models
   :members:
   :undoc-members:
   :show-inheritance:

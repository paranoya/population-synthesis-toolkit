.. _user_guide:

==========
User Guide
==========

This guide describes the main PST workflows beyond the quickstart examples.
It focuses on the current package structure and on the parts of the library
that are most useful when building composite stellar population models,
synthetic observables, and parameterized galaxy SEDs.

For runnable notebooks, see :ref:`tutorials`. For the full class and method
reference, see :ref:`api`.

Getting Oriented
================

PST is organized around a few core building blocks:

- :mod:`pst.SSP` provides simple stellar population libraries and interpolation
    tools on age, metallicity, and wavelength grids.
- :mod:`pst.cem` provides chemical evolution models that map a star-formation
    and enrichment history onto an SSP grid.
- :mod:`pst.sed` provides reusable SED components, including a stellar
    component built from an SSP plus a CEM.
- :mod:`pst.galaxy` combines stellar emission, attenuation, and dust emission
    into a single galaxy SED model.
- :mod:`pst.observables` provides filters and equivalent-width measurements.
- :mod:`pst.model` provides the :class:`pst.model.Parameter` and
    :class:`pst.model.ModelBase` infrastructure used by the parameterized models.

Throughout PST, time arguments in chemical evolution models refer to
**cosmic time since the Big Bang**, not lookback time. This is important when
mixing analytic histories, tabulated histories, and redshift-dependent galaxy
models.

Working with SSP Libraries
==========================

The backbone of the library is :class:`pst.SSP.SSPBase`, which stores a grid of
stellar population spectra sampled in metallicity, age, and wavelength.

Loading an SSP Library
----------------------

PST includes interfaces to several commonly used libraries, such as PopStar,
BC03, XSL, and EMILES.

.. code-block:: python

     from astropy import units as u
     from pst.SSP import PopStar

     ssp = PopStar(IMF="cha")

Where to store SSP data
^^^^^^^^^^^^^^^^^^^^^^^

PST resolves SSP data paths in this order:

1. Explicit path at instantiation (highest priority).
2. The environment variable ``PST_SSP_DIR``.
3. The package fallback directory ``src/pst/data/ssp`` (via ``SSPBase.default_path``).

You can inspect the effective base directory with:

.. code-block:: python

     from pst.SSP import SSPBase
     print(SSPBase.default_path)

Use an explicit path when you want full control for a specific model instance:

.. code-block:: python

     from pst.SSP import PopStar

     ssp = PopStar(IMF="cha", path="/data/ssp/PopStar")

To configure a global SSP base directory, set ``PST_SSP_DIR`` before importing
PST modules:

.. code-block:: bash

     export PST_SSP_DIR=/data/ssp

.. code-block:: python

     from pst.SSP import PopStar
     ssp = PopStar(IMF="cha")

For models that build their own subdirectory from the base path (for example,
PopStar uses ``<base>/PopStar`` when ``path`` is not provided), make sure your
directory structure matches each model's expected layout.

Once loaded, an SSP object exposes its main grids and metadata:

- ``ssp.ages``: age grid.
- ``ssp.metallicities``: metallicity grid.
- ``ssp.wavelength``: wavelength grid.
- ``ssp.L_lambda``: tabulated SSP spectra.
- ``ssp.name``, ``ssp.imf``, ``ssp.isochrone``, ``ssp.stellar_library``.


Interpolating SSP Weights and Spectra
-------------------------------------

PST uses interpolation in age and metallicity space to map arbitrary stellar
populations onto a discrete SSP library.

.. code-block:: python

     weights = ssp.get_weights(
             ages=[1e8, 5e8, 2e9] * u.yr,
             metallicities=[0.004, 0.008, 0.02],
             masses=[1e7, 2e7, 5e7] * u.Msun,
     )

This is the basic operation used internally by the chemical evolution models,
but it is also useful for custom population mixtures or particle-based inputs.

For wavelength-domain operations, you can trim or reinterpolate the spectral
grid before further analysis:

.. code-block:: python

     ssp.cut_wavelength(wl_min=3000 * u.angstrom, wl_max=9000 * u.angstrom)
     ssp.interpolate_sed(new_wavelength_grid)

Regridding in Age and Metallicity
---------------------------------

If you need to align an SSP library with another model grid, you can regrid the
library in age and metallicity.

.. code-block:: python

     import numpy as np

     new_ages = np.geomspace(1e6, 1.3e10, 60) * u.yr
     new_metallicities = np.array([0.0004, 0.004, 0.008, 0.02, 0.05])
     ssp.regrid(new_ages, new_metallicities)

Derived SSP Quantities
----------------------

PST SSP objects can store more than spectra. Depending on the library,
you may also have access to derived quantities such as:

- ``ssp.current_mass`` for surviving stellar mass fractions.
- ``ssp.remnant_mass_frac`` and ``ssp.returned_mass_frac``.
- ``ssp.supernova_rate``.
- ``ssp.log_ionising_HI_photons``, ``ssp.log_ionising_HeI_photons``, ``ssp.log_ionising_HeII_photons``.

These tabulated quantities are used by CEM methods such as
``surviving_stellar_mass``, ``supernova_rate``, and ionising-photon rate
estimators.

Precomputing Synthetic Photometry
---------------------------------

Broadband photometry can be tabulated directly on the SSP grid and reused later
by a CEM.

.. code-block:: python

     from pst.observables import Filter

     g_band = Filter.from_svo("SLOAN_SDSS.g")
     r_band = Filter.from_svo("SLOAN_SDSS.r")
     ssp.compute_photometry([g_band, r_band], z_obs=0.0)

The resulting ``ssp.photometry`` grid can then be combined with any compatible
chemical evolution model through :meth:`pst.cem.ChemicalEvolutionModel.compute_photometry`.

Building Chemical Evolution Models
==================================

The base interface is :class:`pst.cem.ChemicalEvolutionModel`. Subclasses define
two histories as functions of cosmic time:

- cumulative formed stellar mass, ``stellar_mass_formed(t)``
- gas-phase metallicity, ``ism_metallicity(t)``

Analytic CEMs
-------------

Analytic histories are useful for quick experiments, forward modeling, and
parameter studies.

.. code-block:: python

     from astropy import units as u
     from pst.cem import ExponentialDelayedCEM

     cem = ExponentialDelayedCEM(
             mass_today=1e10 * u.Msun,
             today=13.7 * u.Gyr,
             tau=3.0 * u.Gyr,
             ism_metallicity_today=0.02,
     )

Once defined, a CEM can be queried directly:

.. code-block:: python

     cosmic_time = [1.0, 3.0, 10.0] * u.Gyr
     formed_mass = cem.stellar_mass_formed(cosmic_time)
     metallicity = cem.ism_metallicity(cosmic_time)

Synthesizing Composite Stellar Populations
------------------------------------------

The main bridge between a CEM and an SSP library is
:meth:`pst.cem.ChemicalEvolutionModel.interpolate_ssp_masses`, which computes
the mass weights assigned to each SSP age-metallicity bin at a given observing
time.

.. code-block:: python

     weights = cem.interpolate_ssp_masses(ssp, t_obs=13.7 * u.Gyr)
     sed = cem.compute_SED(ssp, t_obs=13.7 * u.Gyr)

From the same combination, PST can derive additional integrated quantities:

- :meth:`pst.cem.ChemicalEvolutionModel.surviving_stellar_mass`
- :meth:`pst.cem.ChemicalEvolutionModel.mean_stellar_age`
- :meth:`pst.cem.ChemicalEvolutionModel.supernova_rate`
- :meth:`pst.cem.ChemicalEvolutionModel.ionising_photon_rate_hi`
- :meth:`pst.cem.ChemicalEvolutionModel.compute_photometry`

.. code-block:: python

     surviving_mass = cem.surviving_stellar_mass(ssp, 13.7 * u.Gyr)
     mean_age = cem.mean_stellar_age(ssp, 13.7 * u.Gyr, log=True)
     photometry = cem.compute_photometry(ssp, 13.7 * u.Gyr)

Caching and Repeated Evaluations
--------------------------------

The CEM layer can cache the output of ``interpolate_ssp_masses`` to accelerate
repeated evaluations at the same observing time.

.. code-block:: python

     cem = ExponentialDelayedCEM(
             mass_today=1e10 * u.Msun,
             today=13.7 * u.Gyr,
             tau=3.0 * u.Gyr,
             ism_metallicity_today=0.02,
             cache_interp_ssp_mass=True,
     )

.. warning::

     The cache key currently depends only on ``ssp.name`` and ``t_obs``. It does
     not track other arguments that affect the interpolation, such as
     ``oversample_factor``. If caching is enabled, keep those inputs fixed across
     repeated calls.

Tabular and Particle-Based Histories
------------------------------------

PST also supports more complex inputs than closed-form SFHs.

- :class:`pst.cem.TabularCEM` interpolates a cumulative mass history and a
    metallicity history from a tabulated control grid.
- :class:`pst.cem.TabularMassFracCEM` reconstructs a history from cumulative
    mass fractions and associated times.
- :class:`pst.cem.CC25TabularCEM` parameterizes the SFH in terms of average
    specific-SFR constraints in lookback-time bins.
- :class:`pst.cem.ParticleListCEM` builds a composite population directly from
    particle ages, metallicities, and masses.

These models are useful when connecting PST to simulations, non-parametric SFH
inference, or external reconstruction pipelines.

.. code-block:: python

     import numpy as np
     from pst.cem import TabularCEM

     tabular = TabularCEM(
             times=np.array([0.1, 1.0, 3.0, 6.0, 13.7]) * u.Gyr,
             masses=np.array([0.0, 1e8, 5e9, 8e9, 1e10]) * u.Msun,
             metallicities=np.array([1e-4, 5e-4, 3e-3, 1e-2, 2e-2]),
             today=13.7 * u.Gyr,
     )

SED Components and Galaxy Models
================================

The :mod:`pst.sed` and :mod:`pst.galaxy` modules make it possible to build more
structured spectral models than a bare SSP+CEM combination.

Stellar SED Components
----------------------

:class:`pst.sed.StellarComponent` wraps an SSP and a CEM into a reusable SED
component.

.. code-block:: python

     from pst.sed import StellarComponent

     stars = StellarComponent(ssp=ssp, sfh=cem)
     stellar_sed = stars.emission_spectrum(ssp.wavelength, t_obs=13.7 * u.Gyr)

This is useful when a stellar population model is one component in a larger SED
assembly that also includes attenuation and dust emission.

Dust Attenuation and Dust Emission
----------------------------------

The dust module provides both attenuation laws and emission components.

- :class:`pst.dust.DustScreenAttenuation` applies a single foreground screen.
- :class:`pst.dust.CharlotFall00Attenuation` returns separate attenuation
    factors for young and old stellar populations.
- :class:`pst.dust.Casey2012DustComponent` generates IR dust-emission spectra.
- :class:`pst.dust.CalorimetricDustComponent` enforces energy balance by using
    the absorbed stellar luminosity to normalize the dust emission.

When building galaxy models, attenuation and dust emission can be combined with
the stellar component instead of being applied manually to a single spectrum.

Composite Galaxy SEDs
---------------------

:class:`pst.galaxy.GalaxySED` combines stellar emission, attenuation, and dust
emission on a common wavelength grid.

.. code-block:: python

     from pst.dust import DustScreenAttenuation
     from pst.galaxy import GalaxySED

     galaxy = GalaxySED(
             stellar_model=stars,
             dust_attenuation_model=DustScreenAttenuation(),
             target_wavelength=ssp.wavelength,
             redshift=0.05,
     )

     rest_sed = galaxy.emission_spectrum()
     obs_sed = galaxy.emission_spectrum(to_obs_frame=True)

This interface is the preferred entry point when you need a higher-level model
that can later be connected to samplers, fitters, or observational pipelines.

Parameterized Workflows
=======================

PST model classes use :class:`pst.model.Parameter` objects rather than
bare scalars. This provides:

- explicit units,
- optional valid ranges,
- support for fixed versus free parameters,
- recursive parameter discovery across nested models.

You can inspect a model directly:

.. code-block:: python

     params = galaxy.parameters_recursive(include_fixed=False)
     for name, par in params.items():
             print(name, par.q, par.fixed)

For nested galaxy models, :class:`pst.galaxy.GalaxySED` also provides a cached
parameter index and dotted-path updates.

.. code-block:: python

     galaxy.build_param_index(include_fixed=False)
     galaxy.update_parameters(
             {
                     "redshift": 0.1,
                     "dust_attenuation.a_v": 0.4 * u.mag,
             },
             validate=True,
     )

This pattern is particularly useful when PST is embedded in fitting frameworks
or MCMC pipelines.

Observables and Measurements
============================

Filters and Synthetic Photometry
--------------------------------

Photometric filters are represented by :class:`pst.observables.Filter`.
Filters can be loaded from local files or fetched from the SVO service.

.. code-block:: python

     import numpy as np
     from pst.observables import Filter

     jwst_filter = Filter.from_svo("JWST_MIRI.F2550W")
     jwst_filter.interpolate(ssp.wavelength)

     sed_10pc = sed / (4 * np.pi * (10 * u.pc) ** 2)
     ab_mag, ab_mag_err = jwst_filter.get_ab(sed_10pc)

Equivalent Width Measurements
-----------------------------

:class:`pst.observables.EquivalentWidth` measures line indices from a spectrum
using a pseudo-continuum defined by two spectral windows, and a central feature window.
The equivalent width is defined as:

.. math::

     EW = \int_{\lambda_1}^{\lambda_2} \left(1 - \frac{F_\lambda(\lambda)}{F_{\rm cont}(\lambda)}\right) d\lambda~~[\mathring{\text{A}}]

where :math:`F_{\rm cont}` is the pseudo-continuum interpolated between the
blue and red side bands.

.. code-block:: python

     from pst.observables import EquivalentWidth

     hbeta = EquivalentWidth(
             left_wl_range=[4827.875, 4847.875],
             central_wl_range=[4847.875, 4876.625],
             right_wl_range=[4876.625, 4891.625],
     )

     ew, ew_err = hbeta.compute_ew(
             wavelength=ssp.wavelength,
             spectra=sed,
             spectra_err=0.01 * sed,
     )

For repeated measurements, you can group several indices with
:class:`pst.observables.EquivalentWidthList` and evaluate them in one call:

.. code-block:: python

     from pst.observables import EquivalentWidthList

     indices = EquivalentWidthList.from_atlas(["Mg1", "TiO2"])
     ew_values, ew_errors = indices.compute_ew(
             wavelength=ssp.wavelength,
             spectra=sed,
             spectra_err=0.01 * sed,
     )

Flux Ratios and the D4000 Break
-------------------------------

:class:`pst.observables.FluxRatio` measures the ratio between the mean flux in a
red window and the mean flux in a blue window,

.. math::

     R = \frac{\langle F_\lambda \rangle_{\rm red}}{\langle F_\lambda \rangle_{\rm blue}}

The :class:`pst.observables.D4000Index` class implements the Balmer 4000-Angstrom break using:

- blue window: 3750-3950 Angstrom
- red window: 4050-4250 Angstrom

.. code-block:: python

     from pst.observables import D4000Index

     d4000 = D4000Index()
     d4000_value, d4000_err = d4000.compute_flux_ratio(
             wavelength=ssp.wavelength,
             spectra=sed,
             spectra_err=0.01 * sed,
     )

Like equivalent widths, flux-ratio diagnostics can be measured on individual
SSP spectra or on composite stellar-population and galaxy spectra.

Choosing a Workflow
===================

The modules are designed to compose cleanly:

1. Use :mod:`pst.SSP` when you only need stellar population libraries and SSP-level observables.
2. Use :mod:`pst.cem` when you need to turn an SFH plus metallicity history into a composite population.
3. Use :mod:`pst.sed` and :mod:`pst.galaxy` when you need structured SED models with attenuation, dust emission, redshift, and parameter management.
4. Use :mod:`pst.observables` to derive photometric or spectroscopic measurements from any spectrum produced upstream.

For individual class signatures and full method documentation, refer to
:ref:`ssp_api`, :ref:`model_api`, :ref:`cem_api`, :ref:`sed_api`,
:ref:`galaxy_api`, :ref:`dust_api`, and :ref:`observables_api`.

Download SSP models
===================

Link to the official sites of the SSP models currently used in PST:

- `(Py)PopStar <https://www.fractal-es.com/PopStar>`_

    See `Mollá et al 2009 <https://ui.adsabs.harvard.edu/abs/2009MNRAS.398..451M/abstract>`_ and `Millán-Irigoyen et al 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4781M/abstract>`_ for details. To increase the read speed, these models are converted into FITS files. See `instructions <https://github.com/paranoya/population-synthesis-toolkit/tree/main/ssp_installation/popstar>`_ for ingesting the models in the PST format.

- `Granada-MILES <https://home.iaa.csic.es/~rosa/AYA2010/AYA2010/>`_

- `XSL <http://xsl.u-strasbg.fr/page_ssp_all.html>`_

- `Bruzual & Charlot 2003 <http://www.bruzual.org/bc03/>`_

   See `instructions <https://github.com/paranoya/population-synthesis-toolkit/tree/main/ssp_installation/bc03>`_ for ingesting the models in the PST format.

Ingesting new SSP models is a straitforward exercise. Users only need to create a new SSP model class by inheriting from the base class :class:`SSPBase`:

.. code-block:: python

    from pst.SSP import SSPBase

    class MyNewSSPModel(SSPBase):
        def __init__(self, *args, **kwargs):
            # Do some stuff to initialise the main SSP properies
            self.initialise_model()
        
        def initialise_model(self):
            self.metallicities = ...
            self.ages = ...
            self.L_lambda = ...

The SSP model must always include these three properties for it work.
# PST tutorials

In this directory you can find some pratical use cases to get you up to speed!

- A first glimpse about creating and manipulating an `SSP` object is provided in [simple_stellar_population.ipynb](./simple_stellar_population.ipynb).

- Several examples of how to build a `ChemicalEvolutionModel` can be found in the [models](./models/) directory. 
    - Many of them are based on [analytical](./models/analytical.ipynb) expressions for the star formation and chemical enrichemt histories.
    - A `TabularCEM` can be used to build models from [tabulated](./models/tabular.ipynb) data.
    - A set of individual simple stellar populations (referred to as "[particles](./models/particles.ipynb)") can be combined using `ParticleListCEM`.

- There is also a tutorial showing how to generate a [grid of photometric fluxes](./observables/create_a_photometric_grid.ipynb) from a given model, including the effects of redshift, and dust attenuation.

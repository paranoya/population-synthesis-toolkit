# Instructions for downloading SSP models

This directory contains basic instructions for downloading the SSP models supported by PST.

PST assumes that the data of each SSP model is located within the installation directory. Users can check the default base path by doing:
```
import os
from pst.SSP import SSPBase
print("Default base directory", SSPBase.default_path)
print("SSP models installed at the default path: ", os.listdir(SSPBase.default_path))
```

This directory includes by default the PopStar model using the Kroupa 2001 IMF.If some other models are required, users are recommended to follow the instructios below.

- For installing the Bruzual and Charlote 2003 models, please follow this [instructions](./bc03/README.md)
- For installing PopStar models from Mollá et al 2003, please follow this [instructions](./popstar/README.md)
- For installing the E-MILES models from Vazdekis et al. 2016, please follow this [instructions](./emiles/README.md)
- For installing the XSL models from Verro et al. 2022b, please follow this [instructions](./xsl/README.md)

If users would like to store the SSP models in a different location, the path to the SSP model data needs to be specified every time the models are initialised. For example:
```
from pst.SSP import PopStar
ssp_model = PopStar("cha", path="path/to/dir/")
```

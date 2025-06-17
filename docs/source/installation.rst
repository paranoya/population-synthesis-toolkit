.. _installation:

==================
Installation Guide
==================

System Requirements
===================
- Python 3.8 or later
- Required packages: NumPy, SciPy, Matplotlib, Astropy, extinction,

Installing PST
================

You can install PST directly from PyPI using `pip`:

.. code-block:: bash

    pip install population-synthesis-toolkit

Installing from Source
======================
To install the package from the source repository:

.. code-block:: bash

    git clone https://github.com/paranoya/population-synthesis-toolkit.git
    cd population-synthesis-toolkit
    python3 -m pip install .

Verifying the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To confirm that the installation was successful, run the provided test script:

.. code-block:: bash

    bash ./run_tests.sh

This script will execute all the core unit tests located in the `/tests` directory. All tests should pass without errors. If any test fails, ensure that all dependencies are installed and that you're running a compatible version of Python.

#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Running PST unit tests..."

# Run against this checkout rather than any separately installed PST version.
PST_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PST_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PST_PROJECT_DIR}/tests"

# Run individual test scripts
python test_dust.py
python test_observables.py
python test_models.py
python test_cem_2d.py
python test_ssp.py

echo "All tests passed successfully!"

#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Running PST unit tests..."

# Navigate to the tests directory
cd "$(dirname "$0")/tests"

# Run individual test scripts
python test_dust.py
python test_observables.py
python test_models.py
python test_ssp.py

echo "All tests passed successfully!"

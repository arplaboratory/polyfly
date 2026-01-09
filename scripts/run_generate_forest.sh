#!/usr/bin/env bash
# Run the forest generation script

cd /workspace/src
python -m poly_fly.forest_planner.generate_forest "$@"


#!/usr/bin/env bash
# Run the interactive forest generation

cd /workspace/src
python -m poly_fly.forest_planner.interactive_generate_forest "$@"


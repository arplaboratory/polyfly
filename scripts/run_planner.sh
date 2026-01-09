#!/usr/bin/env bash
# Run the optimal planner

cd /workspace/src
python -m poly_fly.optimal_planner.planner "$@"


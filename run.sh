#!/usr/bin/env bash
# SkyShade launcher.
# Clears PYTHONPATH (and user-site) so the venv's matplotlib / mpl_toolkits win
# over the ROS + system copies that get injected when ROS is sourced. Without
# this, `from mpl_toolkits.mplot3d import Axes3D` loads the old system matplotlib
# (3.5.1) and the 3D rooms (Sub-2 / Sub-4 tabs) crash.
#
# Usage:  ./run.sh                 # full GUI
#         ./run.sh --no-gui --duration 10
cd "$(dirname "$0")" || exit 1
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 venv/bin/python run_sim.py "$@"

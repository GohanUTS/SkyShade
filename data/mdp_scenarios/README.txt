MDP Scenario Files
==================

This directory contains 50 scripted battery/obstacle episode definitions used
by sub4_nav/test_nav_safety.py.

Each scenario is encoded directly in test_nav_safety.py::build_scenarios() as
a Python list (no external files required for the current test suite).  If you
add new scenarios requiring custom obstacle maps, save them here as JSON files
with the following schema:

    {
        "name": "Scenario descriptive name",
        "battery_pct": 35.0,
        "distance_m": 80.0,
        "forbidden_actions": ["CONTINUE"],
        "rth_latency_check": false
    }

and update test_nav_safety.py to load and run them.

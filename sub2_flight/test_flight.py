"""
Sub-2 Flight — validation test.

Evaluates the trained Q-agent across 10 episodes for each of three scenarios:
  1. Stationary user, no wind
  2. Stationary user, gusty wind (3 m/s)
  3. Walking user, gusty wind    (3 m/s)

Pass criteria (from README):
  • Mean episode reward > +150 across 10 eval episodes per scenario
  • Hover error within 0.5 m in at least 9 of 10 episodes

Usage:
    python sub2_flight/test_flight.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sub2_flight.env.hover_env import HoverEnv, HOVER_RADIUS_M
from sub2_flight.policy import FlightPolicy

EVAL_EPISODES = 10
SCENARIOS = [
    {"name": "Stationary / no wind",   "wind": 0.0, "walk": False},
    {"name": "Stationary / gusty wind", "wind": 3.0, "walk": False},
    {"name": "Walking / gusty wind",    "wind": 3.0, "walk": True},
]

PASS_MEAN_REWARD = 150.0
PASS_HOVER_EPISODES = 9   # Out of EVAL_EPISODES


def evaluate_scenario(policy: FlightPolicy, scenario: dict, rng) -> dict:
    env = HoverEnv(render=False)
    rewards = []
    hover_successes = 0

    for ep in range(EVAL_EPISODES):
        user_pos = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]
        state = env.reset(wind_speed=scenario["wind"], user_pos=user_pos)
        total_reward = 0.0
        done = False
        step = 0
        within_hover = False

        while not done:
            action = policy.select_action(state)

            if scenario["walk"] and step % 20 == 0 and step > 0:
                env._user_pos[0] += rng.uniform(-0.2, 0.2)
                env._user_pos[1] += rng.uniform(-0.2, 0.2)

            state, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1

            dist = np.linalg.norm(env.drone_pos[:2] - env._user_pos[:2])
            if dist <= HOVER_RADIUS_M:
                within_hover = True

        rewards.append(total_reward)
        if within_hover:
            hover_successes += 1

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "hover_successes": hover_successes,
    }


def main():
    try:
        policy = FlightPolicy()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rng = np.random.default_rng(99)
    all_passed = True

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario['name']}")
        result = evaluate_scenario(policy, scenario, rng)
        mean_r = result["mean_reward"]
        hover_ok = result["hover_successes"]

        ok_reward = mean_r > PASS_MEAN_REWARD
        ok_hover = hover_ok >= PASS_HOVER_EPISODES

        status = "PASS" if (ok_reward and ok_hover) else "FAIL"
        print(f"  Mean reward      : {mean_r:+.1f}  (target > {PASS_MEAN_REWARD})  {'✓' if ok_reward else '✗'}")
        print(f"  Hover successes  : {hover_ok}/{EVAL_EPISODES}  (target >= {PASS_HOVER_EPISODES})  {'✓' if ok_hover else '✗'}")
        print(f"  → {status}")

        if not (ok_reward and ok_hover):
            all_passed = False

    print("\nSub-2", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

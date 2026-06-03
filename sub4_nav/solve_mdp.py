"""
Sub-4 Navigation and Safety — offline MDP value iteration solver.

Runs value iteration until the maximum Bellman update falls below
CONVERGENCE_DELTA, then extracts the greedy policy and saves it as a numpy
array.

Usage:
    python sub4_nav/solve_mdp.py \
        --gamma 0.95 \
        --output models/policy_table_v1.npy

Outputs:
  • <output>                    — policy table, shape (N_STATES,) int array
  • convergence_curve.png       — Bellman delta vs iteration
  • Monotone safety check printed to stdout: verifies that LAND_NOW is always
    preferred over CONTINUE when battery is CRITICAL.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sub4_nav.mdp import (
    build_transitions, N_STATES, N_ACTIONS, TERMINAL_STATE,
    ACTION_NAMES, REWARD_PER_STEP, BATTERY_CRITICAL, idx_to_state,
    ACTION_LAND_NOW, ACTION_CONTINUE, ACTION_RTH, N_DISTANCE,
)

# ── Solver config ─────────────────────────────────────────────────────────────
GAMMA = 0.95
CONVERGENCE_DELTA = 1e-6
MAX_ITERATIONS = 10_000
# ─────────────────────────────────────────────────────────────────────────────


def _make_tb_writer():
    """Create a TensorBoard writer for the MDP solve run, or return None."""
    try:
        import time
        from torch.utils.tensorboard import SummaryWriter
        _ts  = time.strftime("%Y%m%d_%H%M%S")
        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "runs", "mdp_solver", _ts)
        os.makedirs(_dir, exist_ok=True)
        logdir = os.path.abspath(os.path.join(_dir, ".."))
        print(f"\n  \033[36m[TensorBoard]\033[0m  tensorboard --logdir {logdir}"
              f"\n  Logging MDP convergence → {_dir}\n")
        return SummaryWriter(_dir)
    except Exception:
        return None


def value_iteration(T, R, gamma: float):
    """
    Standard synchronous value iteration.

    Returns (V, policy, deltas) where:
      V      : value function, shape (N_STATES,)
      policy : greedy policy, shape (N_STATES,) int
      deltas : Bellman delta per iteration (for convergence plot)

    Also streams per-iteration metrics to TensorBoard so the convergence
    curve updates live while the solver runs.
    """
    V = np.zeros(N_STATES + 1, dtype=float)  # +1 for TERMINAL absorbing state
    deltas = []
    tb = _make_tb_writer()

    for iteration in range(MAX_ITERATIONS):
        V_new = V.copy()

        for s in range(N_STATES):
            q_values = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                expected_next = sum(
                    prob * V[ns] for prob, ns in T[s][a]
                )
                q_values[a] = R[s, a] + gamma * expected_next
            V_new[s] = np.max(q_values)

        delta = float(np.max(np.abs(V_new[:N_STATES] - V[:N_STATES])))
        deltas.append(delta)
        V = V_new

        # Log to TensorBoard every iteration so the curve is visible live
        if tb is not None:
            tb.add_scalar("mdp/bellman_delta",     delta,       iteration)
            tb.add_scalar("mdp/log10_bellman_delta",
                          float(np.log10(max(delta, 1e-12))),   iteration)
            tb.add_scalar("mdp/mean_value",        float(np.mean(V[:N_STATES])), iteration)
            tb.add_scalar("mdp/max_value",         float(np.max(V[:N_STATES])), iteration)

        if delta < CONVERGENCE_DELTA:
            print(f"Converged after {iteration + 1} iterations  (Δ={delta:.2e})")
            break
    else:
        print(f"WARNING: did not converge after {MAX_ITERATIONS} iterations.")

    if tb is not None:
        tb.flush()
        tb.close()

    # Extract greedy policy
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        q_values = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            expected_next = sum(prob * V[ns] for prob, ns in T[s][a])
            q_values[a] = R[s, a] + gamma * expected_next
        policy[s] = int(np.argmax(q_values))

    return V[:N_STATES], policy, deltas


def plot_convergence(deltas):
    fig, ax = plt.subplots()
    ax.semilogy(deltas)
    ax.axhline(CONVERGENCE_DELTA, color="red", linestyle="--", label=f"Threshold {CONVERGENCE_DELTA:.0e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Bellman Δ")
    ax.set_title("MDP Value Iteration Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=100)
    print("Convergence curve saved → convergence_curve.png")


def monotone_safety_check(policy):
    """
    Verify that when battery is CRITICAL the policy never chooses CONTINUE.
    Any state with BATTERY_CRITICAL should prefer RTH or LAND_NOW.
    """
    print("\n=== Monotone Safety Check ===")
    passed = True
    for dist in range(N_DISTANCE):
        from sub4_nav.mdp import state_to_idx
        s = state_to_idx(BATTERY_CRITICAL, dist)
        action = policy[s]
        safe = action != ACTION_CONTINUE
        bat, d = idx_to_state(s)
        print(f"  State (CRITICAL, dist={dist}) → {ACTION_NAMES[action]}  {'✓' if safe else '✗ FAIL'}")
        if not safe:
            passed = False
    return passed


def print_policy_table(policy):
    battery_names = ["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    distance_names = ["NEAR", "MID", "FAR"]
    print("\n=== Policy Table ===")
    print(f"{'Battery':<12} {'Distance':<10} {'Action'}")
    print("-" * 35)
    for s in range(N_STATES):
        bat, dist = idx_to_state(s)
        print(f"{battery_names[bat]:<12} {distance_names[dist]:<10} {ACTION_NAMES[policy[s]]}")


def solve(gamma: float, output_path: str):
    T, R = build_transitions()
    V, policy, deltas = value_iteration(T, R, gamma)

    print_policy_table(policy)
    ok = monotone_safety_check(policy)

    plot_convergence(deltas)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, policy)
    print(f"\nPolicy table saved → {output_path}")

    if not ok:
        print("WARNING: Monotone safety check FAILED — review reward structure.")


def main():
    parser = argparse.ArgumentParser(description="Solve Sub-4 MDP via value iteration")
    parser.add_argument("--gamma",  type=float, default=GAMMA)
    parser.add_argument("--output", default="models/policy_table_v1.npy")
    args = parser.parse_args()
    solve(args.gamma, args.output)


if __name__ == "__main__":
    main()

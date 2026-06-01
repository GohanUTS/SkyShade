"""
Sub-4 Nav Safety — background MDP value-iteration solver thread.

Runs synchronous value iteration in a daemon thread, streaming per-iteration
convergence deltas to a queue so the Training Grounds hub can animate the
convergence curve in real time.  Saves the greedy policy table on completion
or early stop.

Queue message format
────────────────────
("iter",    iteration: int,   delta: float)
("done",    policy: ndarray,  values: ndarray)
("stopped", policy: ndarray,  values: ndarray)   # partial save on early stop
("error",   traceback_tail: str)
"""

import os
import json
import queue
import threading
import time
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class MDPSolverWorker(threading.Thread):
    """Background value-iteration solver for Sub-4 navigation safety."""

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "policy_table_v1.npy")

    GAMMA             = 0.95
    CONVERGENCE_DELTA = 1e-6
    MAX_ITER          = 10_000

    def __init__(self, progress_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self._q    = progress_queue
        self._stop_event = stop_event

    def run(self):
        try:
            from sub4_nav.mdp import build_transitions, N_STATES, N_ACTIONS

            T, R = build_transitions()
            V    = np.zeros(N_STATES + 1, dtype=float)

            for it in range(self.MAX_ITER):
                if self._stop_event.is_set():
                    break
                V_new = V.copy()
                for s in range(N_STATES):
                    q = np.array([
                        R[s, a] + self.GAMMA * sum(p * V[ns] for p, ns in T[s][a])
                        for a in range(N_ACTIONS)
                    ])
                    V_new[s] = float(np.max(q))
                delta = float(np.max(np.abs(V_new[:N_STATES] - V[:N_STATES])))
                V = V_new
                self._q.put(("iter", it + 1, delta))
                if delta < self.CONVERGENCE_DELTA:
                    break

            # Extract greedy policy from final V
            policy = np.array([
                int(np.argmax([
                    R[s, a] + self.GAMMA * sum(p * V[ns] for p, ns in T[s][a])
                    for a in range(N_ACTIONS)
                ]))
                for s in range(N_STATES)
            ], dtype=int)

            os.makedirs(_MODELS_DIR, exist_ok=True)
            np.save(self.OUTPUT_PATH, policy)
            with open(self.OUTPUT_PATH + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "subsystem": "Sub-4 Battery Safety",
                    "algorithm": "MDP value iteration",
                    "model": os.path.basename(self.OUTPUT_PATH),
                    "iterations": int(it + 1),
                    "final_delta": float(delta),
                    "trained_at": time.time(),
                }, f, indent=2)

            kind = "stopped" if self._stop_event.is_set() else "done"
            self._q.put((kind, policy.copy(), V[:N_STATES].copy()))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

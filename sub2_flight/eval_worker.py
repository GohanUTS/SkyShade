"""
Sub-2 Flight — PPO hover policy evaluation worker.

Runs the trained PPO model for N_EVAL episodes (deterministic, no exploration)
on PPOHoverEnv stage 2 (stationary user + gusty wind) and streams per-episode
results to a queue.

Queue messages
──────────────
("eval_ep", ep: int, reward: float, hover_rate: float, passed: bool, path: list)
    path : list of [x, y, z] drone positions (sampled every RECORD_EVERY steps)
("eval_done", rewards: list, passes: list, best_path: list)
("eval_error", msg: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

N_EVAL        = 5     # evaluation episodes
MAX_EP_STEPS  = 500
HOVER_PASS    = 0.30  # fraction of episode spent within hover radius = pass
RECORD_EVERY  = 5     # record drone position every N steps

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_flight_v1"
)


class FlightEvalWorker(threading.Thread):
    """Evaluate the trained PPO hover policy."""

    def __init__(self, progress_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self._q    = progress_queue
        self._stop = stop_event

    def run(self):
        try:
            from stable_baselines3 import PPO
            from sub2_flight.env.ppo_hover_env import PPOHoverEnv, HOVER_RADIUS_M

            model = PPO.load(_MODEL_PATH)

            rewards, passes, all_paths = [], [], []

            for ep in range(N_EVAL):
                if self._stop.is_set():
                    break

                # Eval on stage 2 (wind) — harder than stage 1 but without
                # the walking user so each episode is deterministic given the seed
                env = PPOHoverEnv(stage=2, max_episode_steps=MAX_EP_STEPS)
                obs, _ = env.reset(seed=ep * 17)

                total_r     = 0.0
                hover_steps = 0
                total_steps = 0
                path        = []
                done        = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(action)
                    total_r += r
                    total_steps += 1

                    pos  = env._inner_env.drone_pos
                    user = env._inner_env._user_pos
                    dist = float(np.linalg.norm(pos[:2] - user[:2]))
                    if dist <= HOVER_RADIUS_M:
                        hover_steps += 1
                    if total_steps % RECORD_EVERY == 0:
                        path.append([float(pos[0]), float(pos[1]), float(pos[2])])

                    done = term or trunc

                env.close()

                hover_rate = hover_steps / max(1, total_steps)
                passed     = hover_rate >= HOVER_PASS

                rewards.append(total_r)
                passes.append(passed)
                all_paths.append(path)

                self._q.put(("eval_ep", ep + 1, total_r, hover_rate, passed, path))

            best_idx  = int(np.argmax(rewards)) if rewards else 0
            best_path = all_paths[best_idx] if all_paths else []
            self._q.put(("eval_done", rewards, passes, best_path))

        except Exception:
            self._q.put(("eval_error", traceback.format_exc()[-300:]))

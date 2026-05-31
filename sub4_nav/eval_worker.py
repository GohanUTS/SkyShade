"""
Sub-4 Nav — SAC obstacle navigation evaluation worker.

Runs the trained SAC nav model for N_EVAL episodes and streams per-episode
results including the drone's full path so the training ground can visualise
the actual trajectory through the obstacle room.

Queue messages
──────────────
("eval_ep", ep: int, reward: float, reached_goal: bool, steps: int, path: list)
    path : list of [x, y] drone XY positions at flight altitude
("eval_done", rewards: list, successes: list, best_path: list)
("eval_error", msg: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

N_EVAL       = 5
RECORD_EVERY = 3   # record drone XY every N steps

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_nav_v1"
)


class NavEvalWorker(threading.Thread):
    """Evaluate the trained PPO obstacle-navigation policy."""

    def __init__(self, progress_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self._q    = progress_queue
        self._stop = stop_event

    def run(self):
        try:
            from stable_baselines3 import SAC
            from sub4_nav.obstacle_env import ObstacleNavEnv, GOAL_R

            try:
                model = SAC.load(_MODEL_PATH)
            except Exception:
                self._q.put(("eval_error",
                             "Model file contains old PPO weights — incompatible with SAC. "
                             "Click '🔄 Retrain from Scratch' on the Sub-4 Nav tab to train a new SAC model."))
                return

            rewards, successes, all_paths = [], [], []

            for ep in range(N_EVAL):
                if self._stop.is_set():
                    break

                env  = ObstacleNavEnv(max_steps=1_200)
                obs, _ = env.reset(seed=ep * 13)

                total_r      = 0.0
                reached_goal = False
                steps        = 0
                path         = []
                done         = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(action)
                    total_r += r
                    steps   += 1

                    xy = env._drone_pos[:2].tolist()
                    if steps % RECORD_EVERY == 0:
                        path.append(xy)

                    if term and r > 100:   # large reward only on goal arrival
                        reached_goal = True
                    done = term or trunc

                env.close()

                rewards.append(total_r)
                successes.append(reached_goal)
                all_paths.append(path)

                self._q.put(("eval_ep", ep + 1, total_r, reached_goal, steps, path))

            best_idx  = int(np.argmax(rewards)) if rewards else 0
            best_path = all_paths[best_idx] if all_paths else []
            self._q.put(("eval_done", rewards, successes, best_path))

        except Exception:
            self._q.put(("eval_error", traceback.format_exc()[-300:]))

"""
Sub-4 Nav — PPO obstacle-navigation training worker.

Trains a PPO agent on ObstacleNavEnv (10×8 m room, 7 pillars, 8 lidar rays)
in a background daemon thread.  Streams progress tuples — including a
top-down visualisation snapshot — to a queue so the Training Grounds hub
can animate the live map.

Queue message format
────────────────────
("progress", timestep: int, mean_reward: float, viz: dict)
    viz = {
        "drone_xy": np.ndarray (2,),
        "lidar":    list[float]  (8 normalised fractions),
        "path":     list[tuple]  (recent drone positions),
        "goal_xy":  np.ndarray (2,),
        "obstacles": list[tuple] (x, y, r),
        "room":     (ROOM_W, ROOM_D),
    }
("done",    model_path: str, steps_done: int)
("stopped", model_path: str, steps_done: int)
("error",   traceback_tail: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class NavTrainingWorker(threading.Thread):
    """PPO obstacle-avoidance training for Sub-4 navigation."""

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "ppo_nav_v1")

    def __init__(self, total_steps: int, progress_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self._total = total_steps
        self._q     = progress_queue
        self._stop  = stop_event

    def run(self):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from sub4_nav.obstacle_env import ObstacleNavEnv

            q_ref    = self._q
            stop_ref = self._stop
            path_buf: list = []   # recent drone XY positions for viz trail

            class _Callback(BaseCallback):
                def __init__(self_):
                    super().__init__(verbose=0)
                    self_._env_ref = None

                def _on_step(self_) -> bool:
                    if stop_ref.is_set():
                        return False
                    # Grab the unwrapped env to sample visualisation state
                    if self_._env_ref is None:
                        try:
                            self_._env_ref = self_.training_env.envs[0].unwrapped
                        except Exception:
                            pass
                    if self_._env_ref is not None:
                        try:
                            xy = self_._env_ref._drone_pos[:2].copy()
                            path_buf.append(tuple(xy))
                            if len(path_buf) > 120:
                                del path_buf[:-120]
                        except Exception:
                            pass
                    return True

                def _on_rollout_end(self_):
                    n   = self_.num_timesteps
                    buf = self_.model.ep_info_buffer
                    mr  = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0

                    viz: dict = {}
                    if self_._env_ref is not None:
                        try:
                            viz = self_._env_ref.get_viz_state()
                            viz["path"] = list(path_buf)
                        except Exception:
                            pass

                    q_ref.put(("progress", n, mr, viz))

            np.random.seed(42)
            env   = Monitor(ObstacleNavEnv())
            model = PPO(
                "MlpPolicy", env,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.02,
                policy_kwargs={"net_arch": [256, 256]},
                seed=42, verbose=0,
            )
            cb = _Callback()
            model.learn(total_timesteps=self._total, callback=cb, progress_bar=False)
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            model.save(self.OUTPUT_PATH)

            steps_done = cb.num_timesteps
            out  = self.OUTPUT_PATH + ".zip"
            kind = "stopped" if self._stop.is_set() else "done"
            self._q.put((kind, out, steps_done))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

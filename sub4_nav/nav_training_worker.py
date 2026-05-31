"""
Sub-4 Nav — PPO obstacle-navigation training worker.

Warm-starts from the existing model if one is saved, so repeated training
sessions progressively improve the policy rather than resetting.

Queue message format
────────────────────
("progress", timestep: int, mean_reward: float, viz: dict)
("done",    model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("stopped", model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("error",   traceback_tail: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _efficiency_pct(buf) -> int:
    """Estimate navigation success rate (0–100 %) from the ep_info_buffer.

    A successful run reaching the goal scores +200 completion bonus, giving
    episode reward > +100.  A crashing/wandering policy scores −200 to −500.
    We map this so the percentage is meaningful for stakeholders.
    """
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    # −500 → 0 %, +500 → 100 %
    pct = int((mean_r + 500) / 10)
    return max(0, min(100, pct))


class NavTrainingWorker(threading.Thread):
    """PPO obstacle-avoidance training for Sub-4 navigation.

    Warm-starts from an existing model if present (lower LR for fine-tuning).
    Streams per-rollout visualisation data so the 3D room map stays live.
    """

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
            path_buf: list = []

            class _Callback(BaseCallback):
                def __init__(self_):
                    super().__init__(verbose=0)
                    self_._env_ref = None

                def _on_step(self_) -> bool:
                    if stop_ref.is_set():
                        return False
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
            env = Monitor(ObstacleNavEnv())

            # ── Warm-start ────────────────────────────────────────────────────
            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            if warm:
                model = PPO.load(self.OUTPUT_PATH, env=env,
                                 learning_rate=5e-5,   # conservative fine-tuning LR
                                 clip_range=0.1)
            else:
                model = PPO(
                    "MlpPolicy", env,
                    n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.02,
                    policy_kwargs={"net_arch": [256, 256]},
                    seed=42, verbose=0,
                )

            cb = _Callback()
            model.learn(
                total_timesteps=self._total,
                callback=cb,
                progress_bar=False,
                reset_num_timesteps=not warm,
            )
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            model.save(self.OUTPUT_PATH)

            eff        = _efficiency_pct(model.ep_info_buffer)
            steps_done = cb.num_timesteps
            out        = self.OUTPUT_PATH + ".zip"
            kind       = "stopped" if self._stop.is_set() else "done"
            self._q.put((kind, out, steps_done, eff, warm))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

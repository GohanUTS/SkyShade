"""
Sub-1 Perception — SAC gimbal-tracking training worker.

Trains a Soft Actor-Critic policy on GimbalTrackingEnv: the agent learns to
drive the camera gimbal so a moving target stays centred in the frame (learned
visual servoing, on top of the classic HSV detector).

SAC is a good fit: continuous pan/tilt actions, off-policy replay for sample
efficiency, and automatic entropy tuning for exploration.

Queue message format (matches the Sub-4 SAC worker so the hub drains it the
same way):
    ("progress", timestep: int, mean_reward: float, viz: dict)
    ("done",    model_path: str, steps_done: int, accuracy_pct: int, warm: bool)
    ("stopped", model_path: str, steps_done: int, accuracy_pct: int, warm: bool)
    ("error",   traceback_tail: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# TensorBoard event logs land in <repo>/runs/ (already git-ignored).  Override
# with SKYSHADE_TB_DIR, or disable entirely with SKYSHADE_TB=0.
_TB_DIR = (None if os.environ.get("SKYSHADE_TB", "1") == "0"
           else os.environ.get(
               "SKYSHADE_TB_DIR",
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")))

REPORT_EVERY = 2048   # stream a progress update every N env steps


def _accuracy_pct(buf, episode_steps: int = 200) -> int:
    """Estimate tracking accuracy (0–100 %) from the ep_info_buffer.

    Best case the agent earns ~+1 per step (locked) → ep reward ≈ +episode_steps;
    a poor agent goes strongly negative.  Map that span onto 0–100 %.
    """
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    return max(0, min(100, int((mean_r + episode_steps) / (2 * episode_steps) * 100)))


class GimbalTrainingWorker(threading.Thread):
    """SAC visual-servoing trainer for Sub-1 perception gimbal tracking."""

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "gimbal_rl_v1")

    def __init__(self, total_steps: int, progress_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self._total = total_steps
        self._q     = progress_queue
        self._stop_ev  = stop_event

    def run(self):
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from sub1_perception.gimbal_rl_env import GimbalTrackingEnv

            q_ref    = self._q
            stop_ref = self._stop_ev

            class _Callback(BaseCallback):
                def __init__(self_):
                    super().__init__(verbose=0)
                    self_._env_ref = None
                    self_._last_n  = 0

                def _on_step(self_) -> bool:
                    if stop_ref.is_set():
                        return False
                    if self_._env_ref is None:
                        try:
                            self_._env_ref = self_.training_env.envs[0].unwrapped
                        except Exception:
                            pass
                    n = self_.num_timesteps
                    if n - self_._last_n >= REPORT_EVERY:
                        buf = self_.model.ep_info_buffer
                        mr  = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
                        viz = {}
                        if self_._env_ref is not None:
                            try:
                                viz = self_._env_ref.get_viz_state()
                            except Exception:
                                pass
                        # Domain metrics for TensorBoard (alongside SB3's built-in
                        # rollout/ and train/ scalars).
                        self_.logger.record("perception/mean_reward", mr)
                        if "lock_rate" in viz:
                            self_.logger.record("perception/lock_rate",
                                                 float(viz["lock_rate"]))
                        q_ref.put(("progress", n, mr, viz))
                        self_._last_n = n
                    return True

            np.random.seed(42)
            env = Monitor(GimbalTrackingEnv())

            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            model    = None
            if warm:
                try:
                    model = SAC.load(self.OUTPUT_PATH, env=env, learning_rate=5e-5)
                    model.tensorboard_log = _TB_DIR   # saved models don't carry this
                    q_ref.put(("progress", 0, 1, {}))
                except Exception:
                    warm = False
                    q_ref.put(("progress", -1, 1, {}))   # incompatible existing file
            if model is None:
                model = SAC(
                    "MlpPolicy", env,
                    learning_rate   = 3e-4,
                    buffer_size     = 100_000,
                    batch_size      = 256,
                    tau             = 0.005,
                    gamma           = 0.98,
                    ent_coef        = "auto",
                    learning_starts = 1_000,
                    train_freq      = 1,
                    gradient_steps  = 1,
                    policy_kwargs   = {"net_arch": [128, 128]},
                    seed            = 42,
                    verbose         = 0,
                    tensorboard_log = _TB_DIR,
                )
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                except Exception:
                    pass

            cb = _Callback()
            model.learn(
                total_timesteps     = self._total,
                callback            = cb,
                progress_bar        = False,
                reset_num_timesteps = not warm,
                log_interval        = 10,
                tb_log_name         = "sub1_gimbal_SAC",
            )
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            model.save(self.OUTPUT_PATH)

            acc        = _accuracy_pct(model.ep_info_buffer)
            steps_done = cb.num_timesteps
            kind       = "stopped" if self._stop_ev.is_set() else "done"
            self._q.put((kind, zip_path, steps_done, acc, warm))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

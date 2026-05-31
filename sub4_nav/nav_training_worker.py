"""
Sub-4 Nav — SAC obstacle-navigation training worker.

Replaces the previous PPO worker.  Soft Actor-Critic is off-policy and
sample-efficient: it stores all experience in a replay buffer and learns from
the full history, not just the most recent rollout.  The automatic entropy
tuning (the "soft" part) encourages exploration without needing a manual
curriculum — the drone naturally tries many routes through the obstacle room.

Why SAC over PPO here
─────────────────────
• Off-policy replay: reuses every transition → ~3× fewer env steps to converge
• Automatic entropy:  no manual stage thresholds; exploration self-regulates
• Continuous action:  native fit for velocity-setpoint outputs

Queue message format
────────────────────
("progress", timestep: int, mean_reward: float, viz: dict)
("done",    model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("stopped", model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("error",   traceback_tail: str)

Note: the output file is named ppo_nav_v1 for backwards compatibility but now
contains SAC weights — SAC and PPO are not interchangeable at load time.
"""

import os
import queue
import threading
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

REPORT_EVERY = 2048   # stream a progress update every N env steps


def _efficiency_pct(buf) -> int:
    """Estimate navigation success (0–100 %) from the ep_info_buffer."""
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    # −500 → 0 %, +500 → 100 %
    return max(0, min(100, int((mean_r + 500) / 10)))


class NavTrainingWorker(threading.Thread):
    """SAC obstacle-avoidance training for Sub-4 navigation.

    Warm-starts from an existing saved model (lower LR for fine-tuning).
    Streams per-interval visualisation data so the 3D room map stays live.
    """

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "ppo_nav_v1")   # legacy name kept

    def __init__(self, total_steps: int, progress_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self._total = total_steps
        self._q     = progress_queue
        self._stop  = stop_event

    def run(self):
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from sub4_nav.obstacle_env import ObstacleNavEnv

            q_ref    = self._q
            stop_ref = self._stop
            path_buf: list = []

            class _Callback(BaseCallback):
                """SAC is off-policy — progress is reported every REPORT_EVERY steps."""

                def __init__(self_):
                    super().__init__(verbose=0)
                    self_._env_ref    = None
                    self_._last_n     = 0

                def _on_step(self_) -> bool:
                    if stop_ref.is_set():
                        return False

                    # Grab unwrapped env for visualisation
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

                    # Periodic progress report
                    n = self_.num_timesteps
                    if n - self_._last_n >= REPORT_EVERY:
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
                        self_._last_n = n

                    return True

            np.random.seed(42)
            env = Monitor(ObstacleNavEnv())

            # ── Warm-start: load existing model if present ────────────────────
            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            if warm:
                # SAC warm-start: load weights, reduce learning rate for fine-tuning
                model = SAC.load(self.OUTPUT_PATH, env=env,
                                 learning_rate=5e-5)
            else:
                model = SAC(
                    "MlpPolicy", env,
                    learning_rate    = 3e-4,
                    buffer_size      = 100_000,   # replay buffer (off-policy memory)
                    batch_size       = 256,        # larger batches for stable Q-learning
                    tau              = 0.005,      # soft target-network update speed
                    gamma            = 0.99,
                    ent_coef         = "auto",     # automatic entropy: self-regulating exploration
                    learning_starts  = 1_000,      # collect this many steps before first update
                    train_freq       = 1,
                    gradient_steps   = 1,
                    policy_kwargs    = {"net_arch": [256, 256]},
                    seed             = 42,
                    verbose          = 0,
                )

            cb = _Callback()
            model.learn(
                total_timesteps    = self._total,
                callback           = cb,
                progress_bar       = False,
                reset_num_timesteps= not warm,
                log_interval       = 10,
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

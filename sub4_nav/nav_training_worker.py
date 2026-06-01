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
import json
import queue
import threading
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skyshade_mpl")

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# TensorBoard event logs land in <repo>/runs/ (already git-ignored).  Override
# with SKYSHADE_TB_DIR, or disable entirely with SKYSHADE_TB=0.
_TB_DIR = (None if os.environ.get("SKYSHADE_TB", "1") == "0"
           else os.environ.get(
               "SKYSHADE_TB_DIR",
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")))

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
                 stop_event: threading.Event, scenario: str = None):
        super().__init__(daemon=True)
        self._total = total_steps
        self._q     = progress_queue
        self._stop_event  = stop_event
        self._scenario    = scenario   # train on this scenario's obstacle layout

    def run(self):
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from sub4_nav.obstacle_env import ObstacleNavEnv, scenario_obstacles

            q_ref    = self._q
            stop_ref = self._stop_event
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
                        # Domain metric for TensorBoard (alongside SB3's built-in
                        # rollout/ and train/ scalars).
                        self_.logger.record("navigation/mean_reward", mr)
                        q_ref.put(("progress", n, mr, viz))
                        self_._last_n = n

                    return True

            np.random.seed(42)
            env = Monitor(ObstacleNavEnv(
                obstacles=scenario_obstacles(self._scenario)))

            # ── Warm-start: load existing SAC model if present ───────────────
            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            model    = None
            if warm:
                try:
                    # If this file was trained with old PPO, SAC.load() will raise —
                    # catch it and fall back to a fresh SAC model.
                    model = SAC.load(self.OUTPUT_PATH, env=env,
                                     learning_rate=5e-5)
                    model.tensorboard_log = _TB_DIR   # saved models don't carry this
                    q_ref.put(("progress", 0, 1, 0.0))
                except Exception:
                    warm = False
                    q_ref.put(("progress", -1, 1, 0.0))   # -1 signals incompatible file
            if model is None:
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
                    tensorboard_log  = _TB_DIR,
                )
                # Remove incompatible old file so next run doesn't try to load it
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                except Exception:
                    pass

            cb = _Callback()
            model.learn(
                total_timesteps    = self._total,
                callback           = cb,
                progress_bar       = False,
                reset_num_timesteps= not warm,
                log_interval       = 10,
                tb_log_name        = "sub4_nav_SAC",
            )
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            tmp_output = self.OUTPUT_PATH + ".tmp"
            tmp_zip = tmp_output + ".zip"
            final_zip = self.OUTPUT_PATH + ".zip"
            try:
                if os.path.exists(tmp_zip):
                    os.remove(tmp_zip)
                if os.path.exists(tmp_output):
                    os.remove(tmp_output)
            except OSError:
                pass
            model.save(tmp_output)
            written_tmp = tmp_zip if os.path.exists(tmp_zip) else tmp_output
            if not os.path.exists(written_tmp) or os.path.getsize(written_tmp) == 0:
                raise RuntimeError("SAC save failed: no model zip was written")
            os.replace(written_tmp, final_zip)

            eff        = _efficiency_pct(model.ep_info_buffer)
            steps_done = cb.num_timesteps
            out        = final_zip
            with open(self.OUTPUT_PATH + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "subsystem": "Sub-4 Navigation",
                    "algorithm": "SAC",
                    "model": os.path.basename(out),
                    "steps_done": int(steps_done),
                    "efficiency_pct": int(eff),
                    "warm_started": bool(warm),
                    "trained_at": time.time(),
                    "scenario": self._scenario or "default",
                    "legacy_filename": "ppo_nav_v1.zip",
                }, f, indent=2)
            kind       = "stopped" if self._stop_event.is_set() else "done"
            self._q.put((kind, out, steps_done, eff, warm))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

"""
Sub-2 Flight — background PPO training thread.

Uses 4 parallel PyBullet environments (SubprocVecEnv) so training runs ~4×
faster than a single env.  Each subprocess gets its own physics server, so
there is no shared state between workers.

Throughput (typical):
  1 env  → ~500–600 steps/sec  → 100k steps ≈  3 min
  4 envs → ~1800–2400 steps/sec → 100k steps ≈  1 min

Warm-starts from the existing model when ppo_flight_v1.zip is present, so
every training session adds improvement without discarding prior learning.

Queue message format
────────────────────
("progress", timestep: int, stage: int, mean_reward: float)
("done",     model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("stopped",  model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("error",    traceback_tail: str)
"""

import os
import json
import queue
import threading
import time
import traceback

import numpy as np

_MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# TensorBoard event logs land in <repo>/runs/ (already git-ignored).  Override
# with SKYSHADE_TB_DIR, or disable entirely with SKYSHADE_TB=0.
_TB_DIR = (None if os.environ.get("SKYSHADE_TB", "1") == "0"
           else os.environ.get(
               "SKYSHADE_TB_DIR",
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")))

# Curriculum thresholds (cumulative env steps across all parallel workers)
_STAGE2_STEPS = 200_000
_STAGE3_STEPS = 400_000

# Number of parallel PyBullet environments — ~4× throughput on 4-core CPUs
N_ENVS = 4


def _efficiency_pct(buf) -> int:
    """Estimate hover efficiency (0–100 %) from the SB3 ep_info_buffer."""
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    pct = int((mean_r + 600) / 7.5)
    return max(0, min(100, pct))


class FlightTrainingWorker(threading.Thread):
    """Background PPO training thread — 4 parallel envs for ~4× speed."""

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "ppo_flight_v1")

    def __init__(self, total_steps: int, progress_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self._total = total_steps
        self._q     = progress_queue
        self._stop_event  = stop_event

    def run(self):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            from sub2_flight.env.ppo_hover_env import PPOHoverEnv

            q_ref    = self._q
            stop_ref = self._stop_event

            class _Callback(BaseCallback):
                def __init__(self_):
                    super().__init__(verbose=0)

                def _on_step(self_) -> bool:
                    return not stop_ref.is_set()

                def _on_rollout_end(self_):
                    n     = self_.num_timesteps
                    stage = (3 if n >= _STAGE3_STEPS else
                             2 if n >= _STAGE2_STEPS else 1)
                    self_.training_env.env_method("set_stage", stage)
                    buf    = self_.model.ep_info_buffer
                    mean_r = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
                    # Domain metrics for TensorBoard (alongside SB3's built-ins).
                    self_.logger.record("flight/mean_reward", mean_r)
                    self_.logger.record("flight/curriculum_stage", stage)
                    q_ref.put(("progress", n, stage, mean_r))

            np.random.seed(0)

            # ── 4 parallel PyBullet DIRECT environments ───────────────────────
            def _make_env():
                return Monitor(PPOHoverEnv(stage=1))

            # Use 'spawn' start method to avoid fork-in-thread segfaults on Linux.
            # fork() from a daemon thread is unsafe; spawn creates a clean process.
            import multiprocessing as _mp
            try:
                _mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass   # already set

            try:
                env = make_vec_env(_make_env, n_envs=N_ENVS,
                                   vec_env_cls=SubprocVecEnv)
            except Exception:
                # Fallback to single-env if multiprocessing unavailable
                q_ref.put(("progress", 0, 1, 0.0))   # signal fallback
                env = make_vec_env(_make_env, n_envs=1,
                                   vec_env_cls=DummyVecEnv)

            # ── Warm-start: load existing model if present ────────────────────
            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            if warm:
                model = PPO.load(self.OUTPUT_PATH, env=env,
                                 learning_rate=1e-4,
                                 clip_range=0.15)
                model.tensorboard_log = _TB_DIR   # saved models don't carry this
                q_ref.put(("progress", 0, 1, 0.0))
            else:
                model = PPO(
                    "MlpPolicy", env,
                    n_steps   = 512,       # smaller rollout per env → more frequent updates
                    batch_size= 256,
                    n_epochs  = 10,
                    gamma     = 0.99,
                    gae_lambda= 0.95,
                    learning_rate = 3e-4,
                    clip_range    = 0.2,
                    ent_coef      = 0.01,
                    policy_kwargs = {"net_arch": [256, 256]},
                    seed=0, verbose=0,
                    tensorboard_log = _TB_DIR,
                )

            cb = _Callback()
            model.learn(
                total_timesteps    = self._total,
                callback           = cb,
                progress_bar       = False,
                reset_num_timesteps= not warm,
                tb_log_name        = "sub2_flight_PPO",
            )
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            model.save(self.OUTPUT_PATH)

            eff        = _efficiency_pct(model.ep_info_buffer)
            steps_done = cb.num_timesteps
            out        = self.OUTPUT_PATH + ".zip"
            with open(self.OUTPUT_PATH + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "subsystem": "Sub-2 Flight",
                    "algorithm": "PPO",
                    "model": os.path.basename(out),
                    "steps_done": int(steps_done),
                    "efficiency_pct": int(eff),
                    "warm_started": bool(warm),
                    "trained_at": time.time(),
                    "parallel_envs": int(N_ENVS),
                }, f, indent=2)
            kind       = "stopped" if self._stop_event.is_set() else "done"
            self._q.put((kind, out, steps_done, eff, warm))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

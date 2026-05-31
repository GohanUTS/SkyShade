"""
Sub-2 Flight — background PPO training thread.

Warm-starts from the existing model if one is already saved, so each
training run fine-tunes and improves the previous result rather than
starting from scratch.

Queue message format
────────────────────
("progress", timestep: int, stage: int, mean_reward: float)
("done",     model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("stopped",  model_path: str, steps_done: int, efficiency_pct: int, warm: bool)
("error",    traceback_tail: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Curriculum thresholds — match ppo_hover_env.py
_STAGE2_STEPS = 200_000
_STAGE3_STEPS = 400_000


def _efficiency_pct(buf) -> int:
    """Estimate hover efficiency (0–100 %) from the SB3 ep_info_buffer.

    Well-trained hover in stage-2 wind yields mean reward ≈ −50 to +150.
    Untrained or crashing gives −800 to −1 500.  We map this linearly so
    stakeholders see a meaningful percentage rather than a raw RL number.
    """
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    # Clamp and normalise: −600 → 0 %, +150 → 100 %
    pct = int((mean_r + 600) / 7.5)
    return max(0, min(100, pct))


class FlightTrainingWorker(threading.Thread):
    """Background PPO training thread for Sub-2 hover control.

    Warm-starts from an existing saved model if present.  The curriculum
    always runs all three stages so the model is continuously refined:
      Stage 1 (0 – 200 k steps)  : stationary user, calm air
      Stage 2 (200 k – 400 k)    : stationary user, gusty wind
      Stage 3 (400 k +)          : walking user, gusty wind
    """

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "ppo_flight_v1")

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
            from sub2_flight.env.ppo_hover_env import PPOHoverEnv

            q_ref    = self._q
            stop_ref = self._stop

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
                    q_ref.put(("progress", n, stage, mean_r))

            np.random.seed(0)
            env = Monitor(PPOHoverEnv(stage=1))

            # ── Warm-start: load existing model, otherwise create fresh ───────
            zip_path = self.OUTPUT_PATH + ".zip"
            warm     = os.path.exists(zip_path)
            if warm:
                model = PPO.load(self.OUTPUT_PATH, env=env,
                                 learning_rate=1e-4,   # lower LR for fine-tuning
                                 clip_range=0.15)
                # Signal warm-start to the UI
                q_ref.put(("progress", 0, 1, 0.0))
            else:
                model = PPO(
                    "MlpPolicy", env,
                    n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs={"net_arch": [256, 256]},
                    seed=0, verbose=0,
                )

            cb = _Callback()
            model.learn(
                total_timesteps=self._total,
                callback=cb,
                progress_bar=False,
                reset_num_timesteps=not warm,   # keep counter on fine-tune
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

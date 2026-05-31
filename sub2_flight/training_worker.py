"""
Sub-2 Flight — background PPO training thread.

Runs SB3 PPO with a 3-stage curriculum in a daemon thread so the UI stays
responsive.  Reports progress tuples to a queue after every rollout and saves
the model on completion or early stop.

Queue message format
────────────────────
("progress", timestep: int, stage: int, mean_reward: float)
("done",     model_path: str, steps_done: int)
("stopped",  model_path: str, steps_done: int)   # partial save on early stop
("error",    traceback_tail: str)
"""

import os
import queue
import threading
import traceback

import numpy as np

# Model saved here (no .zip — SB3 appends it automatically)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class FlightTrainingWorker(threading.Thread):
    """Background PPO training thread for Sub-2 hover control."""

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
            from sub2_flight.env.ppo_hover_env import PPOHoverEnv, STAGE2_STEPS, STAGE3_STEPS

            q_ref    = self._q
            stop_ref = self._stop

            class _Callback(BaseCallback):
                def __init__(self_):
                    super().__init__(verbose=0)

                def _on_step(self_) -> bool:
                    return not stop_ref.is_set()

                def _on_rollout_end(self_):
                    n     = self_.num_timesteps
                    stage = 3 if n >= STAGE3_STEPS else (2 if n >= STAGE2_STEPS else 1)
                    self_.training_env.env_method("set_stage", stage)
                    buf   = self_.model.ep_info_buffer
                    mean_r = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
                    q_ref.put(("progress", n, stage, mean_r))

            np.random.seed(0)
            env   = Monitor(PPOHoverEnv(stage=1))
            model = PPO(
                "MlpPolicy", env,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.01,
                policy_kwargs={"net_arch": [256, 256]},
                seed=0, verbose=0,
            )
            cb = _Callback()
            model.learn(total_timesteps=self._total, callback=cb, progress_bar=False)
            env.close()

            os.makedirs(_MODELS_DIR, exist_ok=True)
            model.save(self.OUTPUT_PATH)

            steps_done = cb.num_timesteps
            out        = self.OUTPUT_PATH + ".zip"
            kind       = "stopped" if self._stop.is_set() else "done"
            self._q.put((kind, out, steps_done))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))

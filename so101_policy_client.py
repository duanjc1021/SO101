#!/usr/bin/env python3
"""Run a trained pi0.5 SO101 policy on the real arm via an openpi policy server.

This is the deployment counterpart to record_pi05_dataset.py. It:
  1. connects the SO101 follower arm + top/wrist cameras,
  2. each tick builds the observation the policy expects,
  3. queries the openpi WebsocketClientPolicy (action chunks streamed one-at-a-time),
  4. sends the predicted 6-DOF action to the follower.

The policy server runs wherever the checkpoint lives (the H100, or locally):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_so101 --policy.dir=checkpoints/pi05_so101/<exp>/<step>

IMPORTANT — the openpi REPACK transform is applied to *dataset* data only, NOT at
inference. So the client must send the keys the SO101Inputs transform reads directly:
    observation/image, observation/wrist_image, observation/state, prompt
(see openpi/src/openpi/policies/so101_policy.py).

Install the lightweight client package on this machine first (no JAX needed):
    pip install -e /home/jincheng/Desktop/vla/openpi/packages/openpi-client

ALWAYS dry-run first (prints obs/actions, never moves the arm):
    python so101_policy_client.py --dry-run
Then go live (start conservative; keep the e-stop within reach):
    python so101_policy_client.py --host <server-ip> --port 8000 --max-relative-target 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

# Reuse the exact camera/arm setup from the recorder so the runtime matches training.
from record_pi05_dataset import (
    DEFAULT_FPS,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_HEIGHT,
    DEFAULT_TASK,
    DEFAULT_TOP_CAMERA,
    DEFAULT_WIDTH,
    DEFAULT_WRIST_CAMERA,
    read_rgb_frame,
)
from record_training_videos import (
    import_lerobot_so101_classes,
    open_camera,
    require_calibration,
)

# Must match the action_horizon in the pi05_so101 TrainConfig.
DEFAULT_ACTION_HORIZON = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained pi0.5 SO101 policy on the arm.")
    p.add_argument("--host", default="0.0.0.0", help="Policy server host (H100 IP or localhost).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--task", default=DEFAULT_TASK, help="Language instruction (must match training).")
    p.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON,
                   help="Must equal the trained config's action_horizon.")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = run until Ctrl+C).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print observations + predicted actions but DO NOT move the arm.")
    p.add_argument("--max-relative-target", type=float, default=5.0,
                   help="LeRobot safety clamp: max degrees per command per joint. Lower = safer.")
    p.add_argument("--top-camera", default=DEFAULT_TOP_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    p.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    p.add_argument("--connect-without-calibration", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from openpi_client import websocket_client_policy as _wcp
    from openpi_client import action_chunk_broker as _acb

    from pathlib import Path
    calib_dir = Path(__file__).resolve().parent
    require_calibration(calib_dir, args.follower_id)
    SO101Follower, SO101FollowerConfig, _SO101Leader, _SO101LeaderConfig = import_lerobot_so101_classes()

    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            calibration_dir=calib_dir,
            # Safety clamp so a bad prediction can't slam a joint.
            max_relative_target=args.max_relative_target,
        )
    )

    captures: dict[str, cv2.VideoCapture] = {}
    try:
        follower.connect(calibrate=not args.connect_without_calibration)
        captures["top"] = open_camera(args.top_camera, args.width, args.height, args.fps)
        captures["wrist"] = open_camera(args.wrist_camera, args.width, args.height, args.fps)

        # Build the SAME state-feature ordering the recorder used, so observation.state
        # is assembled identically to training (joint order must match exactly).
        state_features = hw_to_dataset_features(follower.observation_features, OBS_STR, use_video=False)
        # The follower action dict keys, in order, to remap the policy's 6-vector output.
        action_keys = list(follower.action_features)

        if args.dry_run:
            logging.warning("DRY RUN: the arm will NOT move. Printing observations + predicted actions.")
        policy = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
        logging.info("Connected to policy server. Metadata: %s", policy.get_server_metadata())
        agent = _acb.ActionChunkBroker(policy, action_horizon=args.action_horizon)

        print(f"\nTask: {args.task!r} | action_horizon={args.action_horizon} | dry_run={args.dry_run}")
        print("Ctrl+C to stop.\n", flush=True)

        frame_period = 1.0 / args.fps
        next_t = time.perf_counter()
        step = 0
        while args.max_steps == 0 or step < args.max_steps:
            observation = follower.get_observation()
            frames = {name: read_rgb_frame(name, cap, args.width, args.height) for name, cap in captures.items()}

            # observation.state, assembled exactly as in training.
            state = build_dataset_frame(state_features, observation, OBS_STR)["observation.state"]

            obs = {
                "observation/image": frames["top"],
                "observation/wrist_image": frames["wrist"],
                "observation/state": np.asarray(state, dtype=np.float32),
                "prompt": args.task,
            }

            result = agent.infer(obs)
            action_vec = np.asarray(result["actions"], dtype=np.float32).reshape(-1)
            if action_vec.shape[0] != len(action_keys):
                raise RuntimeError(
                    f"Policy returned {action_vec.shape[0]} actions but follower expects "
                    f"{len(action_keys)} ({action_keys}). Check the config action dim."
                )
            action = {k: float(v) for k, v in zip(action_keys, action_vec)}

            if args.dry_run:
                if step % args.fps == 0:  # ~once per second
                    logging.info("step %d state=%s action=%s", step,
                                 np.round(state, 1).tolist(), {k: round(v, 1) for k, v in action.items()})
            else:
                follower.send_action(action)

            step += 1
            next_t += frame_period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

        return 0
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        return 0
    finally:
        try:
            if follower.is_connected:
                follower.disconnect()
        except Exception:
            logging.exception("Failed to disconnect follower cleanly")
        for cap in captures.values():
            cap.release()


if __name__ == "__main__":
    sys.exit(main())

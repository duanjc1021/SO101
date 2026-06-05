#!/usr/bin/env python3
"""Record SO101 teleop demonstrations into a LeRobotDataset for pi0.5 fine-tuning.

Each 30 Hz frame stores:
  observation.state          float32[6]  follower joint positions (.pos)
  
  action                     float32[6]  leader commanded joint positions (.pos)
  observation.images.top     RGB video   RealSense D435i (overhead / scene)
  observation.images.wrist   RGB video   USB2.0 CAM1 (wrist)
  task                       str         language instruction (pi0.5 is language-conditioned)

Data is written under SO101trainingvideo/<dataset-name> in the LeRobotDataset format,
which is exactly what openpi (pi0 / pi0.5 fine-tuning) and LeRobot policies consume:
normalization stats, per-camera encoded videos, and language tasks are produced for free.

Workflow (operator drives the leader arm; the follower mirrors it):
  python record_pi05_dataset.py --task "fold the towel" --num-episodes 30

  ENTER  start recording an episode
  ENTER  stop + save the current episode
  q      quit the session (between episodes)

Re-run the same command later with --resume to append more episodes to the dataset.
"""

from __future__ import annotations

import argparse
import logging
import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

# Reuse the hardened camera-open logic (MJPG->native fallback, stable warmup) and
# the SO101 import/calibration helpers from the simple video recorder.
from record_training_videos import (
    import_lerobot_so101_classes,
    open_camera,
    require_calibration,
)


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

# Default language instruction. Keep the wording identical across every episode of
# the same task — pi0.5 is language-conditioned, so inconsistent strings hurt training.
DEFAULT_TASK = "put the blue cube in the brown box"
DEFAULT_NUM_EPISODES = 35

# Camera roles identified by unplugging the top camera (the RealSense disappeared):
#   top   = RealSense D435i RGB *color* camera = "video-index0" (YUYV)
#   wrist = USB2.0 CAM1 (Sonix)
# IMPORTANT: the D435i exposes several nodes. "video-index0" is the RGB color stream;
# "video-index2" is an *infrared* camera and records grayscale, so it must NOT be used
# for the top view. Pinned to stable /dev/v4l/by-id paths (device numbers shuffle).
DEFAULT_TOP_CAMERA = (
    "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_"
    "Intel_R__RealSense_TM__Depth_Camera_435i_252443060783-video-index0"
)
DEFAULT_WRIST_CAMERA = (
    "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
)

DEFAULT_DATASET_ROOT = Path("/home/jincheng/Desktop/vla/SO101trainingvideo")

# Stable by-id serial paths for the arms (ttyACM* numbers can swap on replug).
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111036-if00"
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031758-if00"
DEFAULT_FOLLOWER_ID = "bimanual_follower_0_left"
DEFAULT_LEADER_ID = "bimanual_leader_0_left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record SO101 teleop demonstrations into a LeRobotDataset for pi0.5."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Language instruction for the episodes. Use the SAME wording for every episode of a task.")
    parser.add_argument("--dataset-name", default="so101_pi05", help="Dataset folder name under the dataset root.")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id. Defaults to so101/<dataset-name>.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Episodes to record this session.")
    parser.add_argument(
        "--episode-time-s",
        type=float,
        default=0.0,
        help="Auto-stop each episode after this many seconds. 0 = stop manually with ENTER.",
    )
    parser.add_argument("--resume", action="store_true", help="(No longer needed: appending is automatic when the dataset folder already exists.)")
    parser.add_argument("--top-camera", default=DEFAULT_TOP_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument("--calibration-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Optional LeRobot safety limit in degrees per command for follower joints.",
    )
    parser.add_argument(
        "--connect-without-calibration",
        action="store_true",
        help="Call connect(calibrate=False). Use only if calibration is already written to the motors.",
    )
    return parser.parse_args()


class KeyPoller:
    """Non-blocking single-key reader for a terminal (cbreak mode).

    Degrades to a no-op when stdin is not a real TTY (e.g. run under a debugger,
    a pipe, or fixed-duration mode), so the recorder still works via --episode-time-s.
    """

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.old: list | None = None

    def __enter__(self) -> "KeyPoller":
        if self.enabled:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.enabled and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get_key(self) -> str | None:
        if not self.enabled:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def read_rgb_frame(name: str, cap: cv2.VideoCapture, width: int, height: int) -> np.ndarray:
    """Read one frame and return it as a contiguous RGB uint8 array (LeRobot expects RGB)."""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame from camera '{name}'")
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_features(follower, leader, width: int, height: int) -> dict:
    """Build the LeRobot feature schema: 6-DOF state + action plus two RGB video cameras."""
    camera_shapes = {
        "top": (height, width, 3),
        "wrist": (height, width, 3),
    }
    obs_hw = {**follower.observation_features, **camera_shapes}
    obs_features = hw_to_dataset_features(obs_hw, OBS_STR, use_video=True)
    action_features = hw_to_dataset_features(leader.action_features, ACTION, use_video=True)
    return {**obs_features, **action_features}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not sys.stdin.isatty() and args.episode_time_s <= 0:
        raise SystemExit(
            "No interactive terminal detected. Provide --episode-time-s to record "
            "fixed-length episodes, or run this script directly in a terminal."
        )

    repo_id = args.repo_id or f"so101/{args.dataset_name}"
    dataset_root = (args.dataset_root / args.dataset_name).expanduser().resolve()

    calibration_dir = args.calibration_dir.expanduser().resolve()
    require_calibration(calibration_dir, args.follower_id)
    require_calibration(calibration_dir, args.leader_id)
    SO101Follower, SO101FollowerConfig, SO101Leader, SO101LeaderConfig = import_lerobot_so101_classes()

    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            calibration_dir=calibration_dir,
            max_relative_target=args.max_relative_target,
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(port=args.leader_port, id=args.leader_id, calibration_dir=calibration_dir)
    )

    captures: dict[str, cv2.VideoCapture] = {}
    dataset: LeRobotDataset | None = None
    created_fresh = False
    episodes_done = 0
    try:
        calibrate_on_connect = not args.connect_without_calibration
        logging.info("Connecting leader and follower (align follower near leader pose first)...")
        leader.connect(calibrate=calibrate_on_connect)
        follower.connect(calibrate=calibrate_on_connect)

        logging.info("Opening cameras: top=%s wrist=%s", args.top_camera, args.wrist_camera)
        captures["top"] = open_camera(args.top_camera, args.width, args.height, args.fps)
        captures["wrist"] = open_camera(args.wrist_camera, args.width, args.height, args.fps)

        features = build_features(follower, leader, args.width, args.height)

        # Open the dataset:
        #  - valid dataset already on disk (has meta/info.json) -> append to it.
        #    resume() loads metadata locally and only contacts the HuggingFace Hub
        #    when that local metadata is missing, so this stays fully offline.
        #  - an incomplete leftover dir (no meta/info.json, e.g. from a crashed run)
        #    can be neither resumed nor created over (create() raises FileExistsError),
        #    so remove the empty shell and create fresh.
        created_fresh = False
        if (dataset_root / "meta" / "info.json").is_file():
            dataset = LeRobotDataset.resume(repo_id, root=dataset_root, image_writer_threads=4)
            logging.info(
                "Appending to existing dataset at %s (%d episode(s) already recorded)",
                dataset_root,
                dataset.meta.total_episodes,
            )
        else:
            if dataset_root.exists():
                logging.warning("Removing incomplete dataset dir (no meta/info.json): %s", dataset_root)
                shutil.rmtree(dataset_root)
            logging.info("Creating dataset at %s", dataset_root)
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=args.fps,
                features=features,
                root=dataset_root,
                robot_type="so101",
                use_videos=True,
                image_writer_threads=4,
            )
            created_fresh = True

        frame_period = 1.0 / args.fps
        episodes_done = 0
        state = "idle"
        next_t = time.perf_counter()
        episode_start = next_t
        episode_frames = 0

        interactive = sys.stdin.isatty()
        print(f"\nTask: {args.task!r} | target {args.num_episodes} episode(s)", flush=True)
        if interactive:
            print("ENTER = start episode | ENTER = stop & save | d = discard episode | q = quit\n", flush=True)
            print("[idle] move the arm to the start pose, then press ENTER to record.", flush=True)
        else:
            print(
                f"Non-interactive mode: auto-recording {args.episode_time_s:.0f}s episodes.\n",
                flush=True,
            )

        with KeyPoller() as keys:
            while episodes_done < args.num_episodes:
                # --- teleop tick (runs in both idle and recording so the arm is live) ---
                action = leader.get_action()
                follower.send_action(action)
                observation = follower.get_observation()
                frames = {name: read_rgb_frame(name, cap, args.width, args.height) for name, cap in captures.items()}

                if state == "recording":
                    obs_values = {**observation, **frames}
                    obs_frame = build_dataset_frame(dataset.features, obs_values, OBS_STR)
                    action_frame = build_dataset_frame(dataset.features, action, ACTION)
                    dataset.add_frame({**obs_frame, **action_frame, "task": args.task})
                    episode_frames += 1

                # --- control ---
                key = keys.get_key()
                now = time.perf_counter()
                if state == "idle":
                    # Non-interactive mode auto-starts; interactive waits for ENTER.
                    if (not interactive) or key in ("\r", "\n"):
                        state = "recording"
                        episode_start = now
                        episode_frames = 0
                        print(f"[REC] episode {episodes_done + 1}/{args.num_episodes} ... press ENTER to stop.", flush=True)
                    elif key == "q":
                        break
                elif state == "recording":
                    if key == "d":
                        # Discard a botched demo without saving it (keeps the dataset clean).
                        print(f"[discard] dropping current episode ({episode_frames} frames, not saved).", flush=True)
                        dataset.clear_episode_buffer()
                        state = "idle"
                        print(f"[idle] discarded. {episodes_done}/{args.num_episodes} saved. ENTER to record, q to quit.", flush=True)
                        next_t = time.perf_counter()
                        continue
                    auto_stop = args.episode_time_s > 0 and (now - episode_start) >= args.episode_time_s
                    if key in ("\r", "\n") or auto_stop:
                        print(f"[save] encoding {episode_frames} frames ...", flush=True)
                        dataset.save_episode()
                        episodes_done += 1
                        state = "idle"
                        if episodes_done >= args.num_episodes:
                            break
                        print(f"[idle] saved. {episodes_done}/{args.num_episodes} done. ENTER to record next, q to quit.", flush=True)
                        next_t = time.perf_counter()
                        continue

                # --- pace to fps ---
                next_t += frame_period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

        logging.info("Recorded %d episode(s) into %s", episodes_done, dataset_root)
        return 0
    finally:
        # finalize() flushes the metadata buffer and writes parquet footers; without
        # it the dataset is left invalid (empty meta/). It is idempotent and must run
        # even on Ctrl+C, so it lives here in finally.
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                logging.exception("Failed to finalize dataset")
            if created_fresh and episodes_done == 0 and dataset_root.exists():
                # A brand-new dataset with nothing recorded would just be a broken shell.
                shutil.rmtree(dataset_root, ignore_errors=True)
                logging.info("Removed empty dataset dir (no episodes recorded).")
        for name, device in (("leader", leader), ("follower", follower)):
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception:
                logging.exception("Failed to disconnect %s cleanly", name)
        for cap in captures.values():
            cap.release()


if __name__ == "__main__":
    sys.exit(main())

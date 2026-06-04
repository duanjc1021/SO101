#!/usr/bin/env python3
"""Record training camera videos at 640x480 and 30 FPS.

Examples:
  python record_training_videos.py
  python record_training_videos.py --camera /dev/video0 --camera /dev/video2 --duration 60
  python record_training_videos.py --output-dir /home/jincheng/Desktop/vla/SO101trainingvideo/towel_fold --episode-name demo_001

Stop an unlimited recording with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30.0
# Scene cameras matching view_cameras.py: RealSense D435i RGB + USB2.0 CAM1.
# Pin to stable /dev/v4l/by-id paths instead of /dev/videoN numbers, which shuffle
# on reboot/replug (the RealSense exposes 4 nodes; its RGB index has been video4
# and video2 on different boots). The RealSense RGB color stream is its
# "video-index2" node; the other RealSense indices are depth/IR and either won't
# open or deliver non-RGB frames, so they are deliberately not used here.
DEFAULT_CAMERAS = (
    "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_252443060783-video-index2",
    "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
)
# Pin to stable by-id paths (serial numbers) instead of volatile ttyACM* numbers,
# which can swap between the two arms on reboot/replug.
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111036-if00"
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031758-if00"
DEFAULT_FOLLOWER_ID = "bimanual_follower_0_left"
DEFAULT_LEADER_ID = "bimanual_leader_0_left"


def parse_camera(value: str) -> str | int:
    """Accept either a Linux camera path like /dev/video0 or an integer index."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one or more training videos at 640x480 @ 30 FPS."
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help=(
            "Camera device or index. Repeat for multiple cameras. "
            f"Default: {', '.join(DEFAULT_CAMERAS)}"
        ),
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording length in seconds. Use 0 to record until Ctrl+C.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jincheng/Desktop/vla/SO101trainingvideo"),
        help="Directory where episode folders will be written.",
    )
    parser.add_argument(
        "--episode-name",
        default=None,
        help="Optional episode folder name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV video codec. Use MJPG if your camera needs MJPEG capture.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live camera preview windows while recording.",
    )
    parser.add_argument(
        "--record-robot-data",
        action="store_true",
        help="Also record SO101 follower observation/state and leader action data.",
    )
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing <follower-id>.json and <leader-id>.json.",
    )
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


def _try_warmup(cap: cv2.VideoCapture, attempts: int, delay_s: float) -> bool:
    """Read a few frames to confirm the negotiated format actually streams.

    A camera that accepts a format it cannot deliver (e.g. a RealSense color node
    asked for MJPG) blocks in V4L2 select() and never returns a frame, so we cap
    the number of attempts rather than waiting forever.
    """
    for _ in range(attempts):
        ok, _frame = cap.read()
        if ok:
            return True
        time.sleep(delay_s)
    return False


def open_camera(device: str | int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera: {device}\n"
            "It may be in use by another process (only one program can hold a "
            "UVC camera at a time). Check with: fuser /dev/video*"
        )

    # Prefer MJPG (compressed, sustains 640x480@30 on the USB webcam), but fall
    # back to the camera's native format. Forcing MJPG on a camera that cannot
    # produce it — notably the RealSense color node, which is UYVY-only — makes
    # read() hang in V4L2 select() forever (the "select() timeout" symptom).
    # Set FOURCC before resolution so the driver negotiates the pixel format first.
    def configure(fourcc: str | None) -> None:
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    configure("MJPG")
    # Short probe: if MJPG doesn't stream quickly, it isn't supported here.
    if not _try_warmup(cap, attempts=10, delay_s=0.05):
        logging.warning(
            "Camera %s did not stream with MJPG; falling back to native format.",
            device,
        )
        # Reopen cleanly — a wedged MJPG negotiation can leave the handle unusable.
        cap.release()
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen camera: {device}")
        configure(None)
    # NOTE: do not force CAP_PROP_BUFFERSIZE=1 here. The USB2.0 CAM1 then blocks
    # per read and the synchronized camera pair drops from 30 to ~20 FPS.

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> (8 * k)) & 0xFF) for k in range(4))
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(
        "Opened camera %s at reported %sx%s @ %.2f FPS (%s)",
        device,
        actual_width,
        actual_height,
        actual_fps,
        fourcc_str,
    )

    if actual_width != width or actual_height != height:
        logging.warning(
            "Camera %s reported %sx%s instead of requested %sx%s.",
            device,
            actual_width,
            actual_height,
            width,
            height,
        )

    # Confirm the final configuration produces frames before recording starts.
    warmed_up = _try_warmup(cap, attempts=30, delay_s=0.05)
    if not warmed_up:
        cap.release()
        raise RuntimeError(f"Could not read a warmup frame from camera: {device}")

    return cap


def make_writer(path: Path, width: int, height: int, fps: float, fourcc: str) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc[:4]),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def require_calibration(calibration_dir: Path, device_id: str) -> Path:
    calibration_path = calibration_dir / f"{device_id}.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Missing calibration file: {calibration_path}\n"
            f"Expected a JSON file named '{device_id}.json' in {calibration_dir}."
        )
    return calibration_path


def import_lerobot_so101_classes():
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import the official 'lerobot' Python package.\n"
            "Run this script with the LeRobot environment Python, for example:\n"
            "  ./.miniforge3/envs/lerobot/bin/python record_training_videos.py"
        ) from exc

    return SO101Follower, SO101FollowerConfig, SO101Leader, SO101LeaderConfig


def to_jsonable(value):
    """Convert tensors, arrays, and nested containers into JSON-safe values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cameras = [parse_camera(camera) for camera in (args.camera or list(DEFAULT_CAMERAS))]
    episode_name = args.episode_name or datetime.now().strftime("episode_%Y%m%d_%H%M%S")
    episode_dir = args.output_dir.expanduser().resolve() / episode_name
    episode_dir.mkdir(parents=True, exist_ok=False)

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    captures: list[cv2.VideoCapture] = []
    writers: list[cv2.VideoWriter] = []
    video_paths: list[Path] = []
    robot_data_file = None
    follower = None
    leader = None

    try:
        if args.record_robot_data:
            calibration_dir = args.calibration_dir.expanduser().resolve()
            follower_calibration = require_calibration(calibration_dir, args.follower_id)
            leader_calibration = require_calibration(calibration_dir, args.leader_id)

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
                SO101LeaderConfig(
                    port=args.leader_port,
                    id=args.leader_id,
                    calibration_dir=calibration_dir,
                )
            )

            logging.info("Follower port: %s", args.follower_port)
            logging.info("Leader port: %s", args.leader_port)
            logging.info("Follower calibration: %s", follower_calibration)
            logging.info("Leader calibration: %s", leader_calibration)
            logging.info("Align the follower near the leader pose before recording.")

            calibrate_on_connect = not args.connect_without_calibration
            leader.connect(calibrate=calibrate_on_connect)
            follower.connect(calibrate=calibrate_on_connect)
            robot_data_file = (episode_dir / "robot_data.jsonl").open("w", encoding="utf-8")

        for idx, camera in enumerate(cameras):
            captures.append(open_camera(camera, args.width, args.height, args.fps))
            video_path = episode_dir / f"camera_{idx}.mp4"
            writers.append(make_writer(video_path, args.width, args.height, args.fps, args.fourcc))
            video_paths.append(video_path)

        metadata = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "duration_s": args.duration,
            "cameras": [str(camera) for camera in cameras],
            "videos": [str(path) for path in video_paths],
            "fourcc": args.fourcc[:4],
            "robot_data": {
                "enabled": args.record_robot_data,
                "path": str(episode_dir / "robot_data.jsonl") if args.record_robot_data else None,
                "follower_port": args.follower_port if args.record_robot_data else None,
                "leader_port": args.leader_port if args.record_robot_data else None,
                "follower_id": args.follower_id if args.record_robot_data else None,
                "leader_id": args.leader_id if args.record_robot_data else None,
            },
        }
        write_metadata(episode_dir / "metadata.json", metadata)

        logging.info("Recording to %s", episode_dir)
        logging.info("Press Ctrl+C to stop.")

        frame_count = 0
        start_time = time.perf_counter()
        next_frame_time = start_time
        frame_period = 1.0 / args.fps

        # Tolerate brief camera hiccups instead of discarding the whole episode on
        # the first dropped frame. Abort only if a camera stalls for this many
        # consecutive reads (the V4L2 "select() timeout" symptom when a device is
        # contended or wedged), which at 30 FPS is a few seconds of no frames.
        max_consecutive_read_failures = max(1, int(args.fps * 3))
        consecutive_failures = [0 for _ in captures]

        while not stop_requested:
            if args.duration > 0 and time.perf_counter() - start_time >= args.duration:
                break

            now = time.perf_counter()
            timestamp_s = now - start_time

            frames = []
            read_failed = False
            for cam_idx, (camera, cap) in enumerate(zip(cameras, captures, strict=True)):
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures[cam_idx] += 1
                    if consecutive_failures[cam_idx] >= max_consecutive_read_failures:
                        raise RuntimeError(
                            f"Camera {camera} returned no frames for "
                            f"{consecutive_failures[cam_idx]} consecutive reads; "
                            "it is likely disconnected or in use by another process."
                        )
                    logging.warning(
                        "Dropped frame from camera %s (%d/%d before abort)",
                        camera,
                        consecutive_failures[cam_idx],
                        max_consecutive_read_failures,
                    )
                    read_failed = True
                    break
                consecutive_failures[cam_idx] = 0
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                frames.append(frame)

            if read_failed:
                # Skip this tick entirely so cameras stay frame-aligned; retry next loop.
                continue

            observation = None
            action = None
            if args.record_robot_data:
                observation = follower.get_observation()
                action = leader.get_action()
                follower.send_action(action)

            for idx, (writer, frame) in enumerate(zip(writers, frames, strict=True)):
                writer.write(frame)
                if args.preview:
                    cv2.imshow(f"camera_{idx}", frame)

            if robot_data_file is not None:
                row = {
                    "frame_index": frame_count,
                    "timestamp_s": timestamp_s,
                    "observation": to_jsonable(observation),
                    "action": to_jsonable(action),
                }
                robot_data_file.write(json.dumps(row, separators=(",", ":")) + "\n")

            frame_count += 1

            if args.preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break

            next_frame_time += frame_period
            sleep_s = next_frame_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If capture falls behind, resync rather than accumulating delay forever.
                next_frame_time = time.perf_counter()

        elapsed_s = time.perf_counter() - start_time
        metadata.update(
            {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "frames_written_per_camera": frame_count,
                "elapsed_s": round(elapsed_s, 3),
                "measured_fps": round(frame_count / elapsed_s, 3) if elapsed_s > 0 else 0,
            }
        )
        write_metadata(episode_dir / "metadata.json", metadata)
        logging.info(
            "Finished: %d frames per camera in %.2fs (%.2f FPS)",
            frame_count,
            elapsed_s,
            frame_count / elapsed_s if elapsed_s > 0 else 0,
        )
        return 0
    finally:
        if robot_data_file is not None:
            robot_data_file.close()
        for device_name, device in (("leader", leader), ("follower", follower)):
            if device is None:
                continue
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception:
                logging.exception("Failed to disconnect %s cleanly", device_name)
        for writer in writers:
            writer.release()
        for cap in captures:
            cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

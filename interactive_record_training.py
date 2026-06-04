#!/usr/bin/env python3
"""Interactive training recorder.

Press Enter to start recording, then press Space to stop.
This script wraps record_training_videos.py so the actual camera and SO101
cleanup logic stays in one place.
"""

from __future__ import annotations

import argparse
import select
import signal
import subprocess
import sys
import termios
import tty
from pathlib import Path


DEFAULT_CAMERA = "/dev/video2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start training recording with Enter and stop with Space."
    )
    parser.add_argument("--camera", action="append", default=None, help="Camera device. Repeat for multiple cameras.")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--episode-name", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    # Stable by-id serial paths (ttyACM* numbers can swap the arms on reboot/replug).
    parser.add_argument("--follower-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111036-if00")
    parser.add_argument("--leader-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031758-if00")
    parser.add_argument("--follower-id", default="bimanual_follower_0_left")
    parser.add_argument("--leader-id", default="bimanual_leader_0_left")
    parser.add_argument("--calibration-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-robot-data", action="store_true", help="Record video only.")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    script_path = Path(__file__).resolve().parent / "record_training_videos.py"
    command = [
        sys.executable,
        str(script_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--output-dir",
        str(args.output_dir),
        "--follower-port",
        args.follower_port,
        "--leader-port",
        args.leader_port,
        "--follower-id",
        args.follower_id,
        "--leader-id",
        args.leader_id,
        "--calibration-dir",
        str(args.calibration_dir),
    ]

    for camera in args.camera or [DEFAULT_CAMERA]:
        command.extend(["--camera", camera])

    if args.episode_name:
        command.extend(["--episode-name", args.episode_name])
    if not args.no_robot_data:
        command.append("--record-robot-data")
    if args.preview:
        command.append("--preview")

    return command


def wait_for_space() -> None:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == " ":
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    args = parse_args()
    input("Press Enter to START recording...")

    command = build_command(args)
    print("Recording started. Press Space to STOP.")
    process = subprocess.Popen(command)

    try:
        wait_for_space()
        print("\nStopping recording...")
        process.send_signal(signal.SIGINT)
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping recording...")
        process.send_signal(signal.SIGINT)
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())

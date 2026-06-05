#!/home/jincheng/Desktop/vla/lerobot/.miniforge3/envs/lerobot/bin/python
"""Teleoperate an SO101 follower arm from an SO101 leader arm.

Defaults for this setup (pinned to stable by-id serial paths so the
ttyACM0/ttyACM1 enumeration can never swap the arms on reboot/replug):
  follower: serial 5B14111036, calibration bimanual_follower_0_left.json
  leader:   serial 5B14031758, calibration bimanual_leader_0_left.json

Run:
  python leader_follower_teleop.py

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


# Pin to stable by-id paths (serial numbers) instead of volatile ttyACM* numbers,
# which can swap between the two arms on reboot/replug.
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111036-if00"
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031758-if00"
DEFAULT_FOLLOWER_ID = "bimanual_follower_0_left"
DEFAULT_LEADER_ID = "bimanual_leader_0_left"
DEFAULT_FPS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the follower arm as the leader arm moves using the official LeRobot SO101 API."
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
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Target control loop frequency.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help=(
            "Optional LeRobot safety limit in degrees per command for follower joints. "
            "Leave unset for the official direct teleoperation behavior."
        ),
    )
    parser.add_argument(
        "--connect-without-calibration",
        action="store_true",
        help="Call connect(calibrate=False). Use only if calibration is already written to the motors.",
    )
    return parser.parse_args()


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
            "This environment still has the LeRobot console scripts, but the editable source package "
            "appears to be missing.\n\n"
            "Restore/install the official package, then run this script again, for example:\n"
            "  python -m pip install --force-reinstall 'lerobot==0.5.2'\n"
            "or restore the deleted LeRobot source tree in this repository."
        ) from exc

    return SO101Follower, SO101FollowerConfig, SO101Leader, SO101LeaderConfig


def sleep_to_hold_fps(loop_start_s: float, fps: float) -> None:
    if fps <= 0:
        return
    elapsed_s = time.perf_counter() - loop_start_s
    time.sleep(max((1.0 / fps) - elapsed_s, 0.0))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    calibration_dir = args.calibration_dir.expanduser().resolve()
    follower_calibration = require_calibration(calibration_dir, args.follower_id)
    leader_calibration = require_calibration(calibration_dir, args.leader_id)

    SO101Follower, SO101FollowerConfig, SO101Leader, SO101LeaderConfig = import_lerobot_so101_classes()

    follower_config = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        calibration_dir=calibration_dir,
        max_relative_target=args.max_relative_target,
    )
    leader_config = SO101LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
        calibration_dir=calibration_dir,
    )

    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)
    calibrate_on_connect = not args.connect_without_calibration

    logging.info("Follower port: %s", args.follower_port)
    logging.info("Leader port: %s", args.leader_port)
    logging.info("Follower calibration: %s", follower_calibration)
    logging.info("Leader calibration: %s", leader_calibration)
    logging.info("Align the follower near the leader pose before continuing to avoid a sudden jump.")

    try:
        # This mirrors Hugging Face's official LeRobot teleoperation API:
        # action = leader.get_action(); follower.send_action(action)
        leader.connect(calibrate=calibrate_on_connect)
        follower.connect(calibrate=calibrate_on_connect)

        logging.info("Teleoperation started. Move the leader arm; press Ctrl+C to stop.")
        while True:
            loop_start_s = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            sleep_to_hold_fps(loop_start_s, args.fps)
    except KeyboardInterrupt:
        logging.info("Stopping teleoperation.")
    finally:
        for device_name, device in (("leader", leader), ("follower", follower)):
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception:
                logging.exception("Failed to disconnect %s cleanly", device_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())

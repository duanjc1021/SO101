#!/home/jincheng/Desktop/vla/lerobot/.miniforge3/envs/lerobot/bin/python
"""Calibrate the SO101 follower arm and save the calibration JSON.

Run:
  python calibrate_follower.py

Options are the same as leader_follower_teleop.py for the follower side.
Stop with Ctrl+C if needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


DEFAULT_FOLLOWER_PORT = "/dev/ttyACM0"
DEFAULT_FOLLOWER_ID = "bimanual_follower_0_left"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the SO101 follower arm.")
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where <follower-id>.json will be written.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ModuleNotFoundError as exc:
        raise SystemExit("Could not import lerobot. Activate the lerobot conda env first.") from exc

    calibration_dir = args.calibration_dir.expanduser().resolve()
    calibration_dir.mkdir(parents=True, exist_ok=True)

    config = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        calibration_dir=calibration_dir,
    )
    follower = SO101Follower(config)

    logging.info("Follower port: %s", args.follower_port)
    logging.info("Calibration will be saved to: %s", calibration_dir / f"{args.follower_id}.json")

    try:
        follower.connect(calibrate=True)
        logging.info("Calibration complete. Disconnecting.")
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        try:
            if follower.is_connected:
                follower.disconnect()
        except Exception:
            logging.exception("Failed to disconnect cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

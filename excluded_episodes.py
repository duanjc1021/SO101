"""Episodes to EXCLUDE from the so101_pi05 LeRobotDataset during training.

Episodes 6 and 9 have a video/data frame mismatch: their concatenated mp4 segments
hold ~41 and ~49 more frames than their recorded data rows (the data timestamps
themselves are clean 30 Hz). They are NOT physically deleted — lerobot 0.5.2 has no
delete utility and re-encoding the v3.0 videos is blocked by a broken local
torchcodec/ffmpeg. Instead we exclude them at load/train time via the dataset's
`episodes=` filter, which is safe and fully reversible.

Usage (local LeRobot):
    from excluded_episodes import keep_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root, episodes=keep_episodes(ds_meta.total_episodes))

Usage (openpi pi05_so101 config): pass keep_episodes(total) as the data config's
`episodes` list so the model never sees episodes 6 and 9.
"""

from __future__ import annotations

# Bad episodes (video-vs-data frame mismatch: concatenated mp4 segment longer than
# data row count; data timestamps themselves are clean 30 Hz). Found via the full
# integrity sweep over all 65 episodes. 34 and 52 have the largest surplus (~3 s).
EXCLUDED_EPISODES: tuple[int, ...] = (6, 9, 33, 34, 43, 48, 52)


def keep_episodes(total_episodes: int) -> list[int]:
    """Return the list of episode indices to TRAIN on (all except EXCLUDED_EPISODES)."""
    return [e for e in range(total_episodes) if e not in EXCLUDED_EPISODES]

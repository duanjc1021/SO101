# pi0.5 fine-tuning on SO101 — H100 runbook

End-to-end steps to fine-tune pi0.5 on the `so101_pi05` dataset (58 good episodes,
eps 6/9/33/34/43/48/52 excluded) using a rented H100, then run the policy on the arm.

## 0. What's already built (in this repo's local openpi clone)
The config does not exist upstream — it lives in **4 local changes** under
`/home/jincheng/Desktop/vla/openpi`. These must travel to the H100 (they are NOT in
upstream openpi):

1. `src/openpi/policies/so101_policy.py`  *(new)* — SO101 6-DOF + 2-camera transforms
2. `src/openpi/training/config.py` — `LeRobotSO101DataConfig`, `pi05_so101` TrainConfig,
   `DataConfig.episodes` field, `so101_policy` import
3. `src/openpi/training/data_loader.py` — passes `episodes=` to the LeRobot dataset
4. (this runbook)

Easiest way to get them onto the H100: commit your openpi clone to your own git remote
and clone it there, **or** `scp` those changed files over after a fresh upstream clone.

## 1. Set up openpi on the H100
```bash
git clone --recurse-submodules <your openpi remote>   # must include the 4 changes above
cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync          # installs JAX + CUDA12
```

## 2. Put the dataset where LeRobot can find it
The dataset's repo_id is `so101/so101_pi05`, so LeRobot looks for it at
`$HF_LEROBOT_HOME/so101/so101_pi05`. Copy the dataset (116 MB+) accordingly:
```bash
export HF_LEROBOT_HOME=/data/lerobot
mkdir -p $HF_LEROBOT_HOME/so101
rsync -av /local/SO101trainingvideo/so101_pi05  $HF_LEROBOT_HOME/so101/
```
(Alternative: `ds.push_to_hub("duanjc1021/so101_pi05")` from the laptop, then set
`repo_id="duanjc1021/so101_pi05"` in the config and skip HF_LEROBOT_HOME.)

> If you record MORE good episodes before training, bump the `range(65)` in the
> `pi05_so101` config's `episodes=` list to the new total.

## 3. Compute normalization stats (the q01/q99 step)
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_so101
```

## 4. Train (single H100 80GB, full fine-tune)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_so101 --exp-name=so101_cube_v1
```
- Checkpoints land in `checkpoints/pi05_so101/so101_cube_v1/<step>/`.
- 58 episodes is small — watch the loss and keep an earlier checkpoint if it overfits.
- OOM? lower `batch_size` (32 → 16) in the config.

## 5. Serve the trained policy
```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_so101 \
  --policy.dir=checkpoints/pi05_so101/so101_cube_v1/29999
```

## 6. Run on the SO101 arm (client on the laptop)
A lightweight client reads the two cameras + follower joint state, calls the policy
server (local or over the network to the H100), and sends actions to the follower.
See `openpi/examples/simple_client/` as the template; map:
- `observation/image`       ← top camera (RealSense color)
- `observation/wrist_image` ← wrist camera
- `observation/state`       ← follower `.pos` (6,)
- `prompt`                  ← "put the blue cube in the brown box"
The server returns a 6-DOF action chunk; stream it to `follower.send_action(...)`.

## 7. Evaluate
Run 10–20 trials, log success rate, and feed failures back as new recorded episodes
(DAgger-style) if needed.
```

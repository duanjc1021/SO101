#!/usr/bin/env bash
# Preflight check for pi05_so101 training on a rented cloud GPU (e.g. A800 80GB).
# Run from inside the openpi repo AFTER `uv sync` and AFTER placing the dataset.
# Catches the common failures (GPU/VRAM, JAX, disk, config, dataset path) in seconds,
# before committing to a multi-hour run.
#
#   cd openpi
#   HF_LEROBOT_HOME=/data/lerobot bash /path/to/cloud_preflight.sh
set -uo pipefail
ok=0; warn=0; fail=0
pass(){ echo "  ✅ $1"; ok=$((ok+1)); }
warning(){ echo "  ⚠️  $1"; warn=$((warn+1)); }
bad(){ echo "  ❌ $1"; fail=$((fail+1)); }

echo "== 1. GPU visible + VRAM =="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "${vram:-0}" -ge 70000 ] && pass "VRAM ${vram} MiB (>=70GB, fits full fine-tune @ batch 32)" \
    || warning "VRAM ${vram} MiB < 70GB — full fine-tune @ batch 32 may OOM; lower batch_size to 16"
else
  bad "nvidia-smi not found — no GPU?"
fi

echo "== 2. JAX sees the GPU =="
uv run python -c "import jax; d=jax.devices(); print('  devices:', d); assert any('cuda' in str(x).lower() or 'gpu' in str(x).lower() for x in d), 'JAX sees no GPU'" \
  && pass "JAX has a CUDA device" || bad "JAX cannot see the GPU (CUDA/JAX version mismatch) — training would run on CPU"

echo "== 3. Disk free (checkpoints are ~35GB each for full fine-tune) =="
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  free: ${free_gb} GB on $(pwd)"
[ "${free_gb:-0}" -ge 120 ] && pass "disk >=120GB (room for several full checkpoints)" \
  || warning "disk ${free_gb}GB is tight for full checkpoints (~35GB each) — stream to your computer + delete, or enlarge volume"

echo "== 4. Config loads + key params =="
uv run python -c "
from openpi.training import config as c
t = c.get_config('pi05_so101')
print('  name=%s batch=%d steps=%d ema=%s' % (t.name, t.batch_size, t.num_train_steps, t.ema_decay))
print('  episodes kept: %d (expect 58)' % len(t.data.base_config.episodes))
assert t.batch_size == 32 and t.num_train_steps == 30000
assert len(t.data.base_config.episodes) == 58
" && pass "pi05_so101 config valid (batch 32, 30k steps, 58 episodes)" || bad "config failed to load/validate"

echo "== 5. Dataset resolves via repo_id (HF_LEROBOT_HOME) =="
echo "  HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-<unset>}"
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as M
m = M('so101/so101_pi05')
print('  found dataset: %d episodes, fps=%s' % (m.total_episodes, m.fps))
assert m.total_episodes == 65, 'expected 65 episodes'
" && pass "dataset 'so101/so101_pi05' resolves (65 episodes)" \
  || bad "dataset not found — set HF_LEROBOT_HOME so \$HF_LEROBOT_HOME/so101/so101_pi05 exists"

echo "== 6. wandb-disable flag exists =="
uv run python scripts/train.py pi05_so101 --help 2>/dev/null | grep -q "no-wandb-enabled" \
  && pass "--no-wandb-enabled flag available" || warning "could not confirm --no-wandb-enabled (check train.py --help)"

echo
echo "==== preflight: ${ok} passed, ${warn} warnings, ${fail} failures ===="
[ "$fail" -eq 0 ] && echo "READY — start with:" && \
  echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py --config-name pi05_so101" && \
  echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_so101 --exp-name=so101_cube_v1 --no-wandb-enabled" \
  || echo "FIX THE ❌ ITEMS ABOVE BEFORE TRAINING."
exit "$fail"

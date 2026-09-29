#!/usr/bin/env bash
# top500-emix-timeratio: render the flow-style visualization (HTML + PNG) for
# every task from the time-span-filtered indexes staged under /tmp/tr_index.
# Runs under the shared flow lock so it serializes with any running flow
# phase (e.g. the homefilter vis) instead of competing for all CPU cores.
set -euo pipefail
cd /home/rogerspyke/projects/EMIX/Embodied-Mixup
PY=/home/rogerspyke/miniforge3/envs/emix/bin/python
OUT=outputs/intersections_timeratio
mkdir -p "$OUT"

# Same task order the flow uses (fewest average frames per episode first).
TASKS="place_bread_skillet place_bread_basket place_can_basket stack_bowls_two
stack_blocks_two place_burger_fries place_dual_shoes place_cans_plasticbox
blocks_ranking_rgb blocks_ranking_size"

for base in $TASKS; do
  task=${base}_left_cen_arm_top_cam
  html=$OUT/$task.html
  png=${html%.html}.png
  if [ -f "$html" ] && [ -s "$html" ] && [ -f "$png" ] && [ $(stat -c%s "$png") -gt 1024 ]; then
    echo "[vis-tr] reusing complete visualization: $html"
    continue
  fi
  echo "[vis-tr] rendering $task"
  flock logs/.flow-cuda-0-1.lock -c "$PY -u -m emix visualize \
    --repo_id $task --root data/lerobot/$task \
    --index /tmp/tr_index/$task/emix_index.json \
    --output $html --camera isometric"
  flock logs/.flow-cuda-0-1.lock -c "$PY scripts/render_vis_png.py $html $png \
    --width 2400 --height 2800 --device-scale-factor 2 \
    --virtual-time-budget 120000 --margin 24 --timeout 600"
done
echo "[vis-tr] all tasks rendered"

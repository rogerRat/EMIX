#!/usr/bin/env bash
# =============================================================================
# Serial collection + visualization driver for the 10 active left-arm families.
#
# Runs every remaining stage end-to-end in ONE tmux session, strictly serially:
#   collect -> convert -> build -> visualize (10k points) -> render PNG
#
# Stage guards: each stage is protected by a marker file under logs/serial_pipeline.
# A stage whose marker exists is skipped, so an interrupted run resumes without
# redoing hours of collection.  A partially-created target dir is removed before
# a stage reruns.
#
# Alphabetical order (first-letter) per the workflow directive.  The six
# families whose 500-episode collections already exist only run the
# convert -> build -> vis -> render tail; the three bread/burger families run
# the full chain including a fresh collection.
# =============================================================================
set -u

ROOT="/home/rogerspyke/projects/EMIX/Embodied-Mixup"
cd "$ROOT" || exit 1

EMIX_TWIN="/home/rogerspyke/miniforge3/envs/emix-twin/bin/python"
EPISODES="${EPISODES:-300}"
EMIX="/home/rogerspyke/miniforge3/envs/emix/bin/python"
MARKERS="$ROOT/logs/serial_pipeline"
COLLECTION_ROOT="$ROOT/data/robotwin"
DATASET_ROOT="$ROOT/data/lerobot"
INDEX_ROOT="$ROOT/data/emix_index"
VIS_ROOT="$ROOT/outputs/serial_left"
mkdir -p "$MARKERS" "$VIS_ROOT"

# Pin every stage to GPU 0 (CUDA 0 = RTX 4070).  The 5060 needs the
# LD_LIBRARY_PATH strip; GPU 0 avoids it entirely and serial use keeps VRAM low.
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*"; }

stage_done() { [ -f "$MARKERS/$1" ]; }
mark_done() { touch "$MARKERS/$1"; }

collection_complete() {
    local dir="$COLLECTION_ROOT/$1/$1"
    [ -d "$dir/data" ] && [ "$(ls "$dir"/data/episode*.hdf5 2>/dev/null | wc -l)" -ge 500 ]
}

# --- Pre-seed markers for stages whose output already exists on disk ---------
# rgb: collection, dataset and index all exist; only re-visualize at 10k.
touch "$MARKERS/blocks_ranking_rgb.collect"
touch "$MARKERS/blocks_ranking_rgb.convert"
touch "$MARKERS/blocks_ranking_rgb.build"
# The other five families with a complete collection: skip the collect stage.
for t in blocks_ranking_size place_can_basket place_cans_plasticbox \
         place_dual_shoes stack_blocks_two stack_bowls_two; do
    touch "$MARKERS/$t.collect"
done

TASKS=(
    blocks_ranking_rgb
    blocks_ranking_size
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_dual_shoes
    stack_blocks_two
    stack_bowls_two
)

for task in "${TASKS[@]}"; do
    log "==================== TASK $task ===================="

    # ---- collect -----------------------------------------------------------
    if ! stage_done "$task.collect"; then
        if collection_complete "$task"; then
            log "  collect: reuse existing complete collection"
            mark_done "$task.collect"
        else
            log "  collect: running (GPU 0 / 4070, emix-twin)..."
            "$EMIX_TWIN" data_collection/robotwin.py "${task}_left" \
                --task-config "${task}_left" --python "$EMIX_TWIN" --cuda 0 --episodes "$EPISODES" \
            || { log "  FAIL: collect $task"; exit 1; }
            mark_done "$task.collect"
        fi
    else
        log "  collect: already done"
    fi

    # ---- convert (RoboTwin hdf5 -> LeRobot 84x84) ---------------------------
    if ! stage_done "$task.convert"; then
        if [ -f "$DATASET_ROOT/${task}_left/meta/info.json" ]; then
            log "  convert: reuse existing dataset"
            mark_done "$task.convert"
        else
            rm -rf "$DATASET_ROOT/${task}_left"
            log "  convert: converting (emix)..."
            "$EMIX" data_processing/convert_to_lerobot.py \
                --src "$COLLECTION_ROOT/${task}_left/${task}_left" \
                --tgt "$DATASET_ROOT/${task}_left" \
                --repo-id "${task}_left" \
                --image-width 84 --image-height 84 --fps 17 --left-qpos-dim 7 \
            || { log "  FAIL: convert $task"; exit 1; }
            mark_done "$task.convert"
        fi
    else
        log "  convert: already done"
    fi

    # ---- build EmixIndex -----------------------------------------------------
    if ! stage_done "$task.build"; then
        rm -rf "$INDEX_ROOT/${task}_left"
        log "  build: building EmixIndex (emix)..."
        "$EMIX" -m emix build \
            --root "$DATASET_ROOT/${task}_left" \
            --repo_id "${task}_left" \
            --output_dir "$INDEX_ROOT/${task}_left" \
        || { log "  FAIL: build $task"; exit 1; }
        mark_done "$task.build"
    else
        log "  build: already done"
    fi

    # ---- visualize (10k-point cap) + render PNG -------------------------------
    if ! stage_done "$task.vis"; then
        log "  vis: generating HTML (--max-points 10000)..."
        "$EMIX" -m emix visualize \
            --repo_id "${task}_left" \
            --root "$DATASET_ROOT/${task}_left" \
            --index "$INDEX_ROOT/${task}_left/emix_index.json" \
            --output "$VIS_ROOT/${task}_left.html" \
            --camera isometric \
            --skip-click-images \
            --max-points 10000 \
        || { log "  FAIL: visualize $task"; exit 1; }
        log "  vis: rendering PNG (headless Chrome)..."
        "$EMIX" scripts/render_vis_png.py \
            "$VIS_ROOT/${task}_left.html" "$VIS_ROOT/${task}_left.png" \
            --width 2400 --height 2800 --device-scale-factor 2 \
            --virtual-time-budget 120000 --margin 24 --timeout 600 \
        || { log "  FAIL: render $task"; exit 1; }
        mark_done "$task.vis"
    else
        log "  vis: already done"
    fi

    log "  --- TASK $task COMPLETE ---"
done

log "======================================================"
log "ALL 10 TASKS COMPLETE"
log "======================================================"

#!/usr/bin/env bash
# =============================================================================
# dataflow.sh — one-task end-to-end data pipeline:
#   collect (RoboTwin) -> convert (LeRobot 84x84) -> build (EmixIndex)
#   -> visualize (HTML) -> render (PNG)
#
# Usage:
#   ./scripts/dataflow.sh --task blocks_ranking_rgb --num 500
#
#   --task <name>   A v2 task family, either the base name (blocks_ranking_rgb)
#                   or the full task name (blocks_ranking_rgb_left_central_wide_cam).
#                   Must be one of the ten active families.
#   --num  <n>      Number of episodes to collect (default: 500).  Written into
#                   third_party/RoboTwin/task_config/left/<variant>/<task>.yml as
#                   episode_num and used to gate the collection completeness check.
#   --cuda <n>      CUDA device for the collect/convert stages: 0 -> RTX 4070,
#                   1 -> RTX 5060.  Default: 0.  NOTE: nvidia-smi labels the
#                   cards REVERSED from CUDA (nvidia-smi 0 = 5060, 1 = 4070);
#                   this flag mirrors CUDA_VISIBLE_DEVICES, not nvidia-smi.
#                   --cuda 1 additionally strips /usr/local/cuda from
#                   LD_LIBRARY_PATH so cuBLASLt resolves to the env build.
#                   The build/vis stages stay CPU-only.
#   --seed <n>      Fix the collection's scene seed: writes seed.txt with
#                   n, n+1, ..., n+num-1 into the collection dir and sets
#                   use_seed: true in the task config, so re-running the same
#                   --seed reproduces identical demonstrations.  CAVEAT: the
#                   seeds must yield successful trajectories — a bad seed aborts
#                   the collect stage.  For a guaranteed-valid list, collect once
#                   WITHOUT --seed (RoboTwin discovers working seeds into
#                   seed.txt), then set use_seed: true to replay it.  Default:
#                   unset -> use_seed stays false (RoboTwin auto-discovers).
#   --camera <p>    Camera preset for the 3D vis (default: isometric).
#   --max-points <n>  Cap on cross-point groups rendered in the HTML
#                   (default: 10000; --max-points is a CLI-only viz flag).
#
# Idempotent / resumable: every stage is skipped when its artifact already
# exists (a complete collection, a LeRobot meta/info.json, an emix_index.json,
# or the HTML+PNG pair).  A partial/incomplete artifact is removed first so the
# stage restarts cleanly.
#
# Env split (do not merge): collection runs in emix-twin (the only env with the
# vendored curobo/RoboTwin runtime); conversion, EmixIndex build and the
# visualization run in emix.  The build/vis stages are CPU-bound and release
# the GPU (CUDA_VISIBLE_DEVICES="").
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

EMIX_TWIN="/home/rogerspyke/miniforge3/envs/emix-twin/bin/python"  # RoboTwin collection
EMIX="/home/rogerspyke/miniforge3/envs/emix/bin/python"            # convert / index / vis

COLLECTION_ROOT="$ROOT/data/robotwin"
DATASET_ROOT="$ROOT/data/lerobot"
INDEX_ROOT="$ROOT/data/emix_index"
VIS_ROOT="$ROOT/outputs/intersections"
mkdir -p "$VIS_ROOT"

# The ten active v2 families (task_config/left/<name>_left_central_wide_cam).
ACTIVE_FAMILIES=(
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

TASK=""
NUM=500
CUDA_IDX=0
SEED=""
CAMERA="isometric"
MAX_POINTS=10000

log() { echo "[$(date '+%F %T')] $*"; }

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="${2:-}"; shift 2 ;;
        --num) NUM="${2:-}"; shift 2 ;;
        --cuda) CUDA_IDX="${2:-}"; shift 2 ;;
        --gpu) CUDA_IDX="${2:-}"; shift 2 ;;  # deprecated alias for --cuda
        --seed) SEED="${2:-}"; shift 2 ;;
        --camera) CAMERA="${2:-}"; shift 2 ;;
        --max-points) MAX_POINTS="${2:-}"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1"; usage 1 ;;
    esac
done

# ---- validate inputs -------------------------------------------------------
[ -n "$TASK" ] || { log "FAIL: --task is required"; usage 1; }
[[ "$NUM" =~ ^[0-9]+$ ]] && [ "$NUM" -ge 1 ] || { log "FAIL: --num must be a positive integer (got '$NUM')"; exit 1; }
case "$CUDA_IDX" in 0|1) ;; *) log "FAIL: --cuda must be 0 (RTX 4070) or 1 (RTX 5060), got '$CUDA_IDX'"; exit 1 ;; esac
if [ -n "$SEED" ] && ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    log "FAIL: --seed must be a non-negative integer (got '$SEED')"; exit 1
fi

# Accept either the base family name or the full v2 task name.
if [[ "$TASK" == *_left_cen_arm_right_wide_cam ]]; then
    TASK_NAME="$TASK"
    BASE="${TASK%_left_cen_arm_right_wide_cam}"
elif [[ "$TASK" == *_left_central_wide_cam ]]; then
    TASK_NAME="$TASK"
    BASE="${TASK%_left_central_wide_cam}"
else
    BASE="$TASK"
    TASK_NAME="${TASK}_left_cen_arm_right_wide_cam"
fi

case " ${ACTIVE_FAMILIES[*]} " in
    *" $BASE "*) ;;
    *) log "FAIL: '$BASE' is not one of the ten active v2 families:"; printf '  %s\n' "${ACTIVE_FAMILIES[@]}"; exit 1 ;;
esac

# ---- prepare: point RoboTwin's task config at the requested episode count ----
# Task configs mirror the envs/left layout: one subfolder per camera variant
# (base/ holds the default central-cam configs).
TASK_CONFIG=""
for folder in cen_arm_right_wide_cam cen_arm_near_side_cam cen_arm_side_cam cen_arm_front_cam \
              cen_arm_top_cam central_wide_cam left_oppo_cam base ""; do
    candidate="$ROOT/third_party/RoboTwin/task_config/left/${folder:+$folder/}${TASK_NAME}.yml"
    if [ -f "$candidate" ]; then TASK_CONFIG="$candidate"; break; fi
done
if [ -z "$TASK_CONFIG" ]; then
    log "FAIL: missing RoboTwin task config for $TASK_NAME under task_config/left/*/"
    exit 1
fi
if ! grep -q '^episode_num:' "$TASK_CONFIG"; then
    log "FAIL: no 'episode_num:' line in $TASK_CONFIG"
    exit 1
fi
sed -i -E "s/^episode_num:[[:space:]]*[0-9]+/episode_num: $NUM/" "$TASK_CONFIG"
log "config: episode_num -> $NUM in $TASK_CONFIG"

# ---- prepare: pin the collection scene seed (--seed) -------------------------
# RoboTwin collection seed = use_seed flag + seed.txt (one scene seed per
# episode).  With --seed, write seed.txt as a contiguous range starting at
# SEED and flip use_seed to true so the same --seed replays identical demos.
if [ -n "$SEED" ]; then
    if ! grep -q '^use_seed:' "$TASK_CONFIG"; then
        log "FAIL: no 'use_seed:' line in $TASK_CONFIG"
        exit 1
    fi
    sed -i -E "s/^use_seed:[[:space:]]*(true|false)/use_seed: true/" "$TASK_CONFIG"
    seed_file="$ROOT/third_party/RoboTwin/data/$TASK_NAME/$TASK_NAME/seed.txt"
    mkdir -p "$(dirname "$seed_file")"
    {
        for ((i = 0; i < NUM; i++)); do
            printf '%s ' "$((SEED + i))"
        done
    } > "$seed_file"
    log "seed: use_seed=true, wrote $NUM seeds ($SEED..$((SEED + NUM - 1))) to $seed_file"
fi

# ---- GPU runtime ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$CUDA_IDX"
# Self-document the physical card behind this CUDA index.  On this machine
# nvidia-smi's enumeration is REVERSED from CUDA's: CUDA 0 -> RTX 4070 (shown as
# nvidia-smi GPU 1), CUDA 1 -> RTX 5060 (shown as nvidia-smi GPU 0).  Print the
# real name so "--cuda N" maps to the card the user expects.
gpu_name="$(CUDA_VISIBLE_DEVICES="$CUDA_IDX" "$EMIX_TWIN" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'probe failed')"
log "cuda: CUDA_VISIBLE_DEVICES=$CUDA_IDX -> $gpu_name"
if [ "$CUDA_IDX" = "1" ]; then
    # RTX 5060 (Blackwell): a stale /usr/local/cuda cuBLASLt.so.12.1 shadows
    # torch's bundled build via LD_LIBRARY_PATH and breaks every GEMM.  Drop
    # those entries (same rule as scripts/oidn_env.py strip_stale_cuda).
    KEPT=""
    IFS=':' read -r -a ENTRIES <<< "${LD_LIBRARY_PATH:-}"
    for entry in "${ENTRIES[@]}"; do
        case "$entry" in
            *"/usr/local/cuda"*) ;;
            *) [ -n "$KEPT" ] && KEPT="$KEPT:$entry" || KEPT="$entry" ;;
        esac
    done
    if [ -n "$KEPT" ]; then export LD_LIBRARY_PATH="$KEPT"; else unset LD_LIBRARY_PATH; fi
fi

# ---- artifact guards ---------------------------------------------------------
collection_complete() {
    local dir="$COLLECTION_ROOT/$1/$1/data"
    local n
    [ -d "$dir" ] || return 1
    n="$(ls -1 "$dir"/episode*.hdf5 2>/dev/null | wc -l)"
    [ "$n" -ge "$2" ]
}

collection_dir="$COLLECTION_ROOT/$TASK_NAME/$TASK_NAME"
dataset_dir="$DATASET_ROOT/$TASK_NAME"
index_dir="$INDEX_ROOT/$TASK_NAME"
vis_html="$VIS_ROOT/$TASK_NAME.html"
vis_png="$VIS_ROOT/$TASK_NAME.png"

log "==================== TASK $TASK_NAME ($NUM episodes) ===================="

# ---- collect (RoboTwin hdf5) -------------------------------------------------
if collection_complete "$TASK_NAME" "$NUM"; then
    log "  collect: reuse existing complete collection: $collection_dir"
else
    if [ -d "$collection_dir" ]; then
        log "  collect: removing incomplete collection: $collection_dir"
        rm -rf "$collection_dir"
    fi
    log "  collect: running (cuda $CUDA_IDX, emix-twin)..."
    "$EMIX_TWIN" data_collection/robotwin.py "$TASK_NAME" \
        --task-config "$TASK_NAME" --python "$EMIX_TWIN" --cuda "$CUDA_IDX" \
    || {
        log "  FAIL: collect $TASK_NAME"
        if [ -n "$SEED" ]; then
            # A --seed replay aborted on a bad seed; drop the whole in-flight
            # source so a retry starts clean (seed.txt no longer matches files).
            rm -rf "$ROOT/third_party/RoboTwin/data/$TASK_NAME/$TASK_NAME"
            log "  collect: removed in-flight source after failed seed replay: third_party/RoboTwin/data/$TASK_NAME/$TASK_NAME"
        fi
        exit 1
    }
    if ! collection_complete "$TASK_NAME" "$NUM"; then
        log "  FAIL: collect did not reach $NUM episodes: $collection_dir"
        exit 1
    fi
fi

# ---- convert (RoboTwin hdf5 -> LeRobot 84x84) ---------------------------------
if [ -f "$dataset_dir/meta/info.json" ]; then
    log "  convert: reuse existing dataset: $dataset_dir"
else
    log "  convert: converting (emix)..."
    rm -rf "$dataset_dir"
    "$EMIX" data_processing/convert_to_lerobot.py \
        --src "$collection_dir" \
        --tgt "$dataset_dir" \
        --repo-id "$TASK_NAME" \
        --image-width 224 --image-height 224 --fps 17 --left-qpos-dim 7 \
    || { log "  FAIL: convert $TASK_NAME"; exit 1; }
    [ -f "$dataset_dir/meta/info.json" ] || { log "  FAIL: convert produced no meta/info.json"; exit 1; }
fi

# ---- build EmixIndex ----------------------------------------------------------
if [ -f "$index_dir/emix_index.json" ]; then
    log "  build: reuse existing EmixIndex: $index_dir"
else
    log "  build: building EmixIndex (emix, CPU)..."
    rm -rf "$index_dir"
    CUDA_VISIBLE_DEVICES="" "$EMIX" -m emix build \
        --root "$dataset_dir" \
        --repo_id "$TASK_NAME" \
        --output_dir "$index_dir" \
    || { log "  FAIL: build $TASK_NAME"; exit 1; }
    [ -f "$index_dir/emix_index.json" ] || { log "  FAIL: build produced no emix_index.json"; exit 1; }
fi

# ---- visualize HTML + render PNG ------------------------------------------------
if [ -s "$vis_html" ] && [ -s "$vis_png" ]; then
    log "  vis: reuse existing visualization: $vis_html"
else
    log "  vis: generating HTML (--max-points $MAX_POINTS, camera $CAMERA)..."
    rm -f "$vis_html" "$vis_png"
    CUDA_VISIBLE_DEVICES="" "$EMIX" -m emix visualize \
        --repo_id "$TASK_NAME" \
        --root "$dataset_dir" \
        --index "$index_dir/emix_index.json" \
        --output "$vis_html" \
        --camera "$CAMERA" \
        --skip-click-images \
        --max-points "$MAX_POINTS" \
    || { log "  FAIL: visualize $TASK_NAME"; exit 1; }
    log "  vis: rendering PNG (headless Chrome)..."
    CUDA_VISIBLE_DEVICES="" "$EMIX" scripts/render_vis_png.py \
        "$vis_html" "$vis_png" \
        --width 2400 --height 2800 --device-scale-factor 2 \
        --virtual-time-budget 120000 --margin 24 --timeout 600 \
    || { log "  FAIL: render $TASK_NAME"; exit 1; }
    [ -s "$vis_png" ] || { log "  FAIL: render produced no PNG"; exit 1; }
fi

log "==================== TASK $TASK_NAME COMPLETE ===================="
log "  collection: $collection_dir"
log "  dataset:    $dataset_dir"
log "  index:      $index_dir/emix_index.json"
log "  vis:        $vis_html"
log "  png:        $vis_png"

#!/usr/bin/env bash
# Chain launcher: wait for the v0-noise defaultDP supervisor (PID 3757142) to finish,
# verify it completed successfully, then start the emix-noise (emixDP arm, var0025)
# supervisor per docs/_exp.md experiment 2. Detached via setsid; logs to stdout.
set -u
ROOT=/home/rogerspyke/projects/EMIX/Embodied-Mixup
V0_PID=3757142
V0_STATUS="$ROOT/outputs/run_v0-noise-suite-20260928-030125/status.json"

echo "[chain] waiting for v0-noise supervisor PID $V0_PID to exit..."
while kill -0 "$V0_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] v0-noise supervisor exited at $(date '+%F %T')"

ok=$(python3 - "$V0_STATUS" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
stages_ok = all(st.get("exit_code") == 0 for st in s.get("stages", [])) and len(s.get("stages", [])) == 2
tn_en_done = all(v.get("result") for v in s.get("results", {}).get("tn_en", {}).values())
print("1" if (s.get("status") in ("complete", "completed") and stages_ok and tn_en_done) else "0")
EOF
)

if [ "$ok" != "1" ]; then
    echo "[chain] v0-noise batch did not complete cleanly; NOT launching emix-noise. status.json:"
    cat "$V0_STATUS"
    exit 1
fi
echo "[chain] v0-noise batch completed cleanly; launching emix-noise supervisor."

stamp=$(date +%Y%m%d-%H%M%S)
exec /home/rogerspyke/miniforge3/bin/conda run --no-capture-output -n emix python -u \
    "$ROOT/scripts/run_noise_experiment.py" \
    --status-prefix emix-noise \
    --output "$ROOT/outputs/run_emix-noise-suite-$stamp"

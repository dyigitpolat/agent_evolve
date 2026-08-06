#!/usr/bin/env bash
# Wait for src/ to go quiet, verify the sealed closure, fire the live cell,
# and re-verify the closure afterwards so any drift is attributable rather than
# mysterious.
#
# Option 1 per the coordinator: the closure is NOT narrowed and the run is NOT
# isolated by worktree.  Both were rejected on provenance grounds by the party
# who would have benefited.  This just waits.
set -u

AE=/home/yigit/repos/research_stuff/agent_evolve
OUT=/home/yigit/repos/research_stuff/papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/benchmark_q1/timeloop_codesign/full_support_g6
RUN_ID="${RUN_ID:-aug02_timeloop_seats_luna_smoke_r2}"
QUIET_SECONDS="${QUIET_SECONDS:-180}"
MAX_WAIT="${MAX_WAIT:-3600}"
LOG="$AE/.probe_launch/quiet_launch.log"

cd "$AE" || exit 1
: > "$LOG"

closure() {
  ./.venv/bin/python -c "
import importlib
m=importlib.import_module('examples.development.run_timeloop_v2_frontier_probe_live')
s=m.source_identity(m._source_paths(), relative_to=m.WORKSPACE_ROOT)
print(s['aggregate_sha256'], s['file_count'])
" 2>/dev/null | tail -1
}

newest_src_write() {
  find src/agent_evolve examples/benchmarks/timeloop_codesign/v2 -name '*.py' -printf '%T@\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d. -f1
}

echo "[$(date -Is)] waiting for ${QUIET_SECONDS}s of quiet on the sealed closure" >> "$LOG"
start=$(date +%s)
while true; do
  now=$(date +%s)
  last=$(newest_src_write)
  quiet=$(( now - last ))
  if [ "$quiet" -ge "$QUIET_SECONDS" ]; then
    echo "[$(date -Is)] quiet for ${quiet}s -- proceeding" >> "$LOG"
    break
  fi
  if [ $(( now - start )) -ge "$MAX_WAIT" ]; then
    echo "[$(date -Is)] gave up after ${MAX_WAIT}s; last write ${quiet}s ago" >> "$LOG"
    exit 2
  fi
  sleep 15
done

BEFORE=$(closure)
echo "[$(date -Is)] closure before: $BEFORE" >> "$LOG"

# Provenance: record any closure file that is dirty relative to HEAD BEFORE the
# run, so the sealed closure hash can be reconciled against git history later.
# A dirty-but-stable file means the closure pins content that is not any
# committed revision; a future reader diffing the hash against git would
# otherwise find nothing and reasonably suspect corruption.
DIRTY=$(git status --porcelain -- src/agent_evolve examples/benchmarks/timeloop_codesign/v2 \
         examples/development/run_timeloop_v2_frontier_probe_live.py \
         examples/development/run_timeloop_v2_provider_free_campaign.py 2>/dev/null)
mkdir -p "$AE/.probe_launch"
{
  echo "{"
  echo "  \"note\": \"closure provenance for run $RUN_ID\","
  echo "  \"head_commit\": \"$(git rev-parse HEAD)\","
  echo "  \"closure_sha256_before\": \"$(echo "$BEFORE" | awk '{print $1}')\","
  echo "  \"closure_file_count\": $(echo "$BEFORE" | awk '{print $2}'),"
  echo "  \"dirty_closure_files\": ["
  echo "$DIRTY" | sed '/^$/d' | awk '{printf "    \"%s %s\",\n", $1, $2}' | sed '$ s/,$//'
  echo "  ],"
  echo "  \"warning\": \"if any dirty_closure_files are listed, the sealed closure pins content that is NOT any committed revision. Acceptable for a smoke cell proving the seats work; NOT acceptable for a campaign of record, which must be relaunched against a committed tree.\""
  echo "}"
} > "$AE/.probe_launch/closure_provenance_${RUN_ID}.json"
echo "[$(date -Is)] dirty closure files: $(echo "$DIRTY" | sed '/^$/d' | wc -l)" >> "$LOG"

set -a; . /home/yigit/repos/research_stuff/.env 2>/dev/null; set +a
export AGENT_EVOLVE_PROTECTED_ACQUISITION_MODE=botorch_qlognehvi
export AGENT_EVOLVE_PROTECTED_ACQUISITION_BATCH_SIZE=4
export AGENT_EVOLVE_MODEL_PROFILE=gpt_luna

# prepare emits the preregistration contract that live must be handed back.
# It is generated, not authored: _validate_preregistration recomputes it from
# the replicate seed, the source closure, the construction probe, the gate-A
# baseline and the qualification evidence, and refuses any file that differs.
# So the closure must stay still across BOTH runs, not just the live one -- the
# prereg is bound to source_sha256.
echo "[$(date -Is)] prepare: emitting the preregistration contract" >> "$LOG"
./.venv/bin/python examples/development/run_timeloop_v2_frontier_probe_live.py \
  prepare --run-id "${RUN_ID}_prepare" --replicate-seed 20260810 >> "$LOG" 2>&1
PREP=$?
PREREG="$OUT/${RUN_ID}_prepare/preregistration_template.json"
if [ "$PREP" -ne 0 ] || [ ! -f "$PREREG" ]; then
  echo "[$(date -Is)] prepare failed (exit=$PREP) or no prereg at $PREREG" >> "$LOG"
  echo "DONE status=prepare_failed"
  exit 1
fi
MID=$(closure)
if [ "$BEFORE" != "$MID" ]; then
  echo "[$(date -Is)] closure drifted between prepare and live; aborting before spend" >> "$LOG"
  echo "DONE status=drift_before_live"
  exit 1
fi
echo "[$(date -Is)] prereg written; closure still $MID; firing live" >> "$LOG"
./.venv/bin/python examples/development/run_timeloop_v2_frontier_probe_live.py \
  live --run-id "$RUN_ID" --replicate-seed 20260810 --prereg "$PREREG" >> "$LOG" 2>&1
STATUS=$?

AFTER=$(closure)
echo "[$(date -Is)] exit=$STATUS" >> "$LOG"
echo "[$(date -Is)] closure after : $AFTER" >> "$LOG"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "[$(date -Is)] CLOSURE DRIFTED DURING THE RUN" >> "$LOG"
else
  echo "[$(date -Is)] closure stable across the run" >> "$LOG"
fi
echo "DONE status=$STATUS"

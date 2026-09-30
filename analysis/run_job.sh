#!/usr/bin/env bash
# usage: code_repo/analysis/run_job.sh <name> <python module> [args...]
# logs to ${DANE_OUT:-paper_figures_v2}/logs/<name>.{log,rc}, so a run pointed at a different
# DANE_OUT (e.g. a different dataset) never touches another run's bookkeeping.
set -o pipefail
code_root="$(cd "$(dirname "$0")/.." && pwd)"
project="$(cd "$code_root/.." && pwd)"
cd "$project"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
out_dir=${DANE_OUT:-$project/paper_figures_v2}
name=$1; shift
mkdir -p "$out_dir/logs"
log="$out_dir/logs/$name.log"
echo "[$(date '+%F %T')] START $*" | tee "$log"
python -W ignore -m "$@" 2>&1 | tee -a "$log"
rc=$?
echo "[$(date '+%F %T')] END rc=$rc" | tee -a "$log"
echo "$rc" > "$out_dir/logs/$name.rc"

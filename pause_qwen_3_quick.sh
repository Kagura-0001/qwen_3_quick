#!/usr/bin/env bash
# Stop a running qwen 3 quick training job. The next launch starts from scratch.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-qwen3_quick_4b_zero3_lora_lowmem}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_DIR/output/$RUN_ID}"
CONFIG_PATH="${CONFIG_PATH:-${TMPDIR:-/tmp}/${RUN_ID}.yaml}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
FORCE_KILL="${FORCE_KILL:-1}"
REMOVE_CHECKPOINTS_ON_PAUSE="${REMOVE_CHECKPOINTS_ON_PAUSE:-1}"

is_train_cmd() {
  case "$1" in
    *"swift sft"* | *"swift/cli/sft.py"* | *"torchrun"* | *"torch.distributed.run"* | *"torch/distributed/run.py"*)
      return 0
      ;;
  esac
  return 1
}

cmd_matches_run() {
  local text="$1"
  local arg

  [[ "$text" == *"$OUTPUT_DIR"* ]] && return 0
  [[ "$text" == *"$CONFIG_PATH"* ]] && return 0
  [[ "$text" == *"$RUN_ID"* ]] && return 0

  for arg in $text; do
    case "$arg" in
      *.yaml | *.yml)
        [[ -r "$arg" ]] || continue
        grep -F "output_dir: $OUTPUT_DIR" "$arg" >/dev/null 2>&1 && return 0
        ;;
    esac
  done

  return 1
}

find_train_pids() {
  local cmdline pid text
  for cmdline in /proc/[0-9]*/cmdline; do
    pid="${cmdline#/proc/}"
    pid="${pid%/cmdline}"
    [[ "$pid" == "$$" ]] && continue
    [[ -r "$cmdline" ]] || continue
    text="$(tr '\0' ' ' 2>/dev/null <"$cmdline" || true)"
    [[ -z "$text" ]] && continue
    is_train_cmd "$text" || continue
    cmd_matches_run "$text" || continue
    printf '%s\n' "$pid"
  done | sort -n | uniq
}

cleanup_checkpoints() {
  if [[ "$REMOVE_CHECKPOINTS_ON_PAUSE" == "1" ]]; then
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} + 2>/dev/null || true
  fi
}

mapfile -t pids < <(find_train_pids)

if ((${#pids[@]} == 0)); then
  echo "No qwen 3 quick training process found for OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

echo "Stopping qwen 3 quick training for OUTPUT_DIR=$OUTPUT_DIR"
printf 'PIDs: %s\n' "${pids[*]}"

kill -TERM "${pids[@]}" >/dev/null 2>&1 || true

deadline=$((SECONDS + WAIT_SECONDS))
while ((SECONDS < deadline)); do
  mapfile -t remaining < <(find_train_pids)
  if ((${#remaining[@]} == 0)); then
    cleanup_checkpoints
    echo "Stopped."
    exit 0
  fi
  sleep 1
done

mapfile -t remaining < <(find_train_pids)
if ((${#remaining[@]} > 0)); then
  if [[ "$FORCE_KILL" == "1" ]]; then
    printf 'Force killing PIDs: %s\n' "${remaining[*]}"
    kill -KILL "${remaining[@]}" >/dev/null 2>&1 || true
    cleanup_checkpoints
    echo "Stopped with SIGKILL."
  else
    printf 'Still running PIDs: %s\n' "${remaining[*]}" >&2
    exit 1
  fi
else
  cleanup_checkpoints
  echo "Stopped."
fi

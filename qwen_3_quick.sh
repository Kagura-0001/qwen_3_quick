#!/usr/bin/env bash
# One-file launcher for qwen 3 quick: env, model, dataset, config, train.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="${ENV_DIR:-$HOME/.venv/qwen_3_quick}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-$ENV_DIR/bin/swift}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3-4B}"
DATASET_ID="${DATASET_ID:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
DATASET_PATH="${DATASET_PATH:-$HF_DATASETS_CACHE/qwen_3_quick/alpaca_cleaned/train.jsonl}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"

RUN_ID="${RUN_ID:-qwen3_quick_4b_zero3_lora_lowmem}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_DIR/output/$RUN_ID}"
PROFILE_DIR="${PROFILE_DIR:-$BUNDLE_DIR/output/gpu_profiles}"
CONFIG_PATH="${CONFIG_PATH:-${TMPDIR:-/tmp}/${RUN_ID}.yaml}"

AUTO_SETUP="${AUTO_SETUP:-1}"
AUTO_MODEL="${AUTO_MODEL:-1}"
AUTO_DATASET="${AUTO_DATASET:-1}"
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
PROFILE="${PROFILE:-1}"
RESUME="${RESUME:-0}"
DEEPSPEED="${DEEPSPEED-zero3}"

export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME
export HF_DATASETS_CACHE

usage() {
  cat <<'EOF'
qwen 3 quick

Usage:
  ./qwen_3_quick.sh

Common overrides:
  PREPARE_ONLY=1 ./qwen_3_quick.sh        # install/download/convert only
  DRY_RUN=1 ./qwen_3_quick.sh             # generate config only
  ./pause_qwen_3_quick.sh                 # stop training; next run starts over

Default locations:
  ENV_DIR=~/.venv/qwen_3_quick
  MODEL_DIR=~/models/Qwen3-4B
  HF_DATASETS_CACHE=~/.cache/huggingface/datasets
  DATASET_PATH=~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl
  OUTPUT_DIR=./output/qwen3_quick_4b_zero3_lora_lowmem

Training defaults:
  MAX_STEPS=100000000
  SAVE_STRATEGY=steps
  SAVE_STEPS=MAX_STEPS
  MAX_LENGTH=768
  BATCH_SIZE=1
  GRAD_ACC=1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC_PER_NODE=8
  DEEPSPEED=zero3                         # set DEEPSPEED= to disable
  LORA_RANK=8
  LORA_ALPHA=16
  TARGET_MODULES=all-linear               # comma-separated LoRA target modules

Notes:
  No intermediate checkpoint is saved by default. The last-step model-only save
  is flattened into OUTPUT_DIR and checkpoint-* is removed. If paused, the next
  run starts from scratch.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

uv_cmd() {
  local uv_args=(--no-config)
  local host
  for host in ${QWEN_3_QUICK_UV_INSECURE_HOSTS:-${OMNI_SFT_UV_INSECURE_HOSTS:-}}; do
    uv_args+=(--allow-insecure-host "$host")
  done
  uv "${uv_args[@]}" "$@"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  echo "uv not found in PATH; installing to $HOME/.local/bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available; cannot install uv." >&2
    exit 2
  fi
  export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    echo "uv installation finished but uv is still not in PATH." >&2
    exit 2
  }
}

ensure_env() {
  if [[ -x "$PYTHON_BIN" && -x "$SWIFT_BIN" && "${FORCE_ENV_INSTALL:-0}" != "1" ]]; then
    echo "Env ready: $ENV_DIR"
    return
  fi

  ensure_uv
  echo "Creating/updating env: $ENV_DIR"
  uv_cmd venv "$ENV_DIR" --python 3.11 --allow-existing

  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"

  echo "Installing build helpers"
  uv_cmd pip install -U pip setuptools wheel packaging ninja

  echo "Installing PyTorch CUDA 12.9 stack"
  uv_cmd pip install \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0

  echo "Installing ms-swift and Qwen SFT dependencies"
  uv_cmd pip install \
    "ms-swift @ git+https://github.com/modelscope/ms-swift.git@cbb0afb07c688405f5b38b3b6b7894b996cd5a10" \
    "transformers==5.8.1" \
    "datasets==3.6.0" \
    "deepspeed==0.19.0" \
    "accelerate" \
    "peft" \
    "trl" \
    "tensorboard" \
    "wandb==0.27.0" \
    "pyyaml>=5.4" \
    "omegaconf" \
    "modelscope" \
    "huggingface_hub[cli]" \
    "sentencepiece" \
    "tiktoken"

  if command -v nvidia-smi >/dev/null 2>&1; then
    ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | awk 'NF && !seen[$1]++ {print $1}' | paste -sd ';' -)"
    if [[ -n "$ARCH_LIST" ]]; then
      export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH_LIST}"
    fi
  fi
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
  export MAX_JOBS="${MAX_JOBS:-32}"

  if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
    echo "Installing flash-attn"
    uv_cmd pip install --no-build-isolation "flash-attn==2.8.3"
  else
    echo "Skipping flash-attn because SKIP_FLASH_ATTN=1"
  fi

  echo "Sanity checking env"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import torch
import swift
import transformers
import datasets

print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
print("swift", swift.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
try:
    import flash_attn
    print("flash-attn", flash_attn.__version__)
except Exception as exc:
    print("flash-attn import failed:", repr(exc))
print("huggingface-hub", metadata.version("huggingface-hub"))
PY
}

find_hf_bin() {
  if [[ -n "${HF_BIN:-}" ]]; then
    echo "$HF_BIN"
  elif [[ -x "$ENV_DIR/bin/hf" ]]; then
    echo "$ENV_DIR/bin/hf"
  elif [[ -x "$ENV_DIR/bin/huggingface-cli" ]]; then
    echo "$ENV_DIR/bin/huggingface-cli"
  elif command -v hf >/dev/null 2>&1; then
    command -v hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    command -v huggingface-cli
  else
    return 1
  fi
}

ensure_model() {
  mkdir -p "$MODEL_DIR"
  if [[ -f "$MODEL_DIR/config.json" \
      && ( -f "$MODEL_DIR/model.safetensors" || -f "$MODEL_DIR/model.safetensors.index.json" ) \
      && "${FORCE_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    echo "Model ready: $MODEL_DIR"
    return
  fi

  local hf_bin
  hf_bin="$(find_hf_bin)" || {
    echo "Cannot find hf/huggingface-cli. Install env first or set HF_BIN." >&2
    exit 2
  }
  echo "Downloading $MODEL_ID to $MODEL_DIR"
  "$hf_bin" download "$MODEL_ID" \
    --repo-type model \
    --local-dir "$MODEL_DIR"
  test -f "$MODEL_DIR/config.json"
}

ensure_dataset() {
  if [[ -s "$DATASET_PATH" && "${FORCE_DATASET_CONVERT:-0}" != "1" ]]; then
    echo "Dataset ready: $DATASET_PATH"
    return
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing python for dataset conversion: $PYTHON_BIN" >&2
    exit 2
  fi

  mkdir -p "$(dirname "$DATASET_PATH")"
  echo "Downloading/converting dataset $DATASET_ID:$DATASET_SPLIT to $DATASET_PATH"
  "$PYTHON_BIN" - "$DATASET_ID" "$DATASET_SPLIT" "$DATASET_PATH" "$SYSTEM_PROMPT" "${DATASET_LIMIT:-0}" <<'PY'
import json
import sys
from pathlib import Path

from datasets import load_dataset

dataset_id, split, out_path, system_prompt, limit_s = sys.argv[1:]
limit = int(limit_s)
out = Path(out_path)

ds = load_dataset(dataset_id, split=split)
if limit > 0:
    ds = ds.select(range(min(limit, len(ds))))

count = 0
with out.open("w", encoding="utf-8") as f:
    for item in ds:
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        response = str(item.get("output", "")).strip()
        if not instruction or not response:
            continue
        query = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        row = {"system": system_prompt, "query": query, "response": response}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

if count == 0:
    raise SystemExit(f"no usable rows converted from {dataset_id}:{split}")
print(f"converted {count} rows -> {out}")
PY
}

filter_cuda_ld_library_path() {
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    return
  fi
  OLD_IFS="$IFS"
  IFS=':' read -r -a LD_PARTS <<<"$LD_LIBRARY_PATH"
  IFS="$OLD_IFS"
  FILTERED=()
  for part in "${LD_PARTS[@]}"; do
    case "$part" in
      /usr/local/cuda | /usr/local/cuda/* | /usr/local/cuda-* | /usr/local/cuda-*/*) ;;
      *) FILTERED+=("$part") ;;
    esac
  done
  if ((${#FILTERED[@]})); then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${FILTERED[*]}")"
  else
    unset LD_LIBRARY_PATH
  fi
}

setup_runtime_env() {
  unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ALL_PROXY all_proxy
  filter_cuda_ld_library_path

  PY_SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_path("purelib"))
PY
)"
  [[ -d "$PY_SITE_PACKAGES/nvidia/cudnn/lib" ]] && export CUDNN_HOME="${CUDNN_HOME:-$PY_SITE_PACKAGES/nvidia/cudnn/lib}"
  [[ -d "$PY_SITE_PACKAGES/nvidia/cuda_nvrtc/lib" ]] && export NVRTC_HOME="${NVRTC_HOME:-$PY_SITE_PACKAGES/nvidia/cuda_nvrtc/lib}"
  [[ -d "$PY_SITE_PACKAGES/nvidia/curand/lib" ]] && export CURAND_HOME="${CURAND_HOME:-$PY_SITE_PACKAGES/nvidia/curand/lib}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$("$PYTHON_BIN" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
  fi
  export MASTER_PORT

  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
}

latest_checkpoint() {
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed 's#.*/checkpoint-##;s#^#checkpoint-#' \
    | sort -V \
    | tail -1 \
    | sed "s#^#$OUTPUT_DIR/#"
}

write_config() {
  mkdir -p "$OUTPUT_DIR" "$PROFILE_DIR" "$(dirname "$CONFIG_PATH")"
  MAX_STEPS_VALUE="${MAX_STEPS:-100000000}"
  SAVE_STEPS_VALUE="${SAVE_STEPS:-$MAX_STEPS_VALUE}"

  RESUME_PATH="${RESUME_FROM_CHECKPOINT:-}"
  if [[ "$RESUME" == "1" && -z "$RESUME_PATH" ]]; then
    RESUME_PATH="$(latest_checkpoint || true)"
  fi

  cat >"$CONFIG_PATH" <<YAML
model: $MODEL_DIR
dataset:
  - $DATASET_PATH
columns:
  system: system
  query: query
  response: response
tuner_type: lora
tuner_backend: peft
target_modules:
YAML

  OLD_IFS="$IFS"
  IFS=',' read -r -a TARGET_MODULE_ARRAY <<<"${TARGET_MODULES:-all-linear}"
  IFS="$OLD_IFS"
  for module in "${TARGET_MODULE_ARRAY[@]}"; do
    module="${module#"${module%%[![:space:]]*}"}"
    module="${module%"${module##*[![:space:]]}"}"
    [[ -n "$module" ]] && printf '  - %s\n' "$module" >>"$CONFIG_PATH"
  done

  cat >>"$CONFIG_PATH" <<YAML
lora_rank: ${LORA_RANK:-8}
lora_alpha: ${LORA_ALPHA:-16}
lora_dropout: 0.0
torch_dtype: bfloat16
bf16: true
attn_impl: flash_attn
gradient_checkpointing: true
use_logits_to_keep: true
packing: true
packing_length: ${MAX_LENGTH:-768}
packing_num_proc: 8
lazy_tokenize: false
max_length: ${MAX_LENGTH:-768}
per_device_train_batch_size: ${BATCH_SIZE:-1}
gradient_accumulation_steps: ${GRAD_ACC:-1}
learning_rate: ${LEARNING_RATE:-2.0e-4}
num_train_epochs: 1
max_steps: $MAX_STEPS_VALUE
warmup_ratio: 0.03
logging_steps: ${LOGGING_STEPS:-10}
save_strategy: "${SAVE_STRATEGY:-steps}"
save_steps: $SAVE_STEPS_VALUE
eval_steps: 100000
save_total_limit: ${SAVE_TOTAL_LIMIT:-2}
split_dataset_ratio: 0.0
dataset_num_proc: 8
dataloader_num_workers: 4
dataloader_prefetch_factor: 4
dataloader_persistent_workers: true
ddp_find_unused_parameters: false
remove_unused_columns: false
report_to:
  - tensorboard
output_dir: $OUTPUT_DIR
add_version: false
save_only_model: ${SAVE_ONLY_MODEL:-true}
create_checkpoint_symlink: false
YAML

  if [[ -n "${DEEPSPEED:-}" ]]; then
    cat >>"$CONFIG_PATH" <<YAML
deepspeed: $DEEPSPEED
YAML
  fi

  if [[ -n "$RESUME_PATH" ]]; then
    cat >>"$CONFIG_PATH" <<YAML
resume_from_checkpoint: $RESUME_PATH
resume_only_model: false
YAML
  fi
}

print_manifest() {
  echo "qwen 3 quick"
  echo "ENV_DIR: $ENV_DIR"
  echo "MODEL_DIR: $MODEL_DIR"
  echo "DATASET_ID: $DATASET_ID"
  echo "DATASET_SPLIT: $DATASET_SPLIT"
  echo "DATASET_PATH: $DATASET_PATH"
  echo "OUTPUT_DIR: $OUTPUT_DIR"
  echo "CONFIG_PATH: $CONFIG_PATH"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  echo "NPROC_PER_NODE: ${NPROC_PER_NODE:-8}"
  echo "DEEPSPEED: ${DEEPSPEED:-<disabled>}"
  echo "TARGET_MODULES: ${TARGET_MODULES:-all-linear}"
  echo "LORA_RANK: ${LORA_RANK:-8}"
  echo "LORA_ALPHA: ${LORA_ALPHA:-16}"
  echo "MAX_STEPS: ${MAX_STEPS_VALUE:-${MAX_STEPS:-100000000}}"
  echo "SAVE_STRATEGY: ${SAVE_STRATEGY:-steps}"
  echo "SAVE_STEPS: ${SAVE_STEPS_VALUE:-${SAVE_STEPS:-${MAX_STEPS:-100000000}}}"
  if [[ -n "${RESUME_PATH:-}" ]]; then
    echo "RESUME_FROM_CHECKPOINT: $RESUME_PATH"
  else
    echo "RESUME_FROM_CHECKPOINT: <none>"
  fi
}

run_train() {
  GPU_CSV="$PROFILE_DIR/${RUN_ID}_$(date +%Y%m%d_%H%M%S)_nvidia_smi.csv"
  DMON_LOG="$PROFILE_DIR/${RUN_ID}_$(date +%Y%m%d_%H%M%S)_dmon.log"

  monitor_pid=""
  dmon_pid=""
  cleanup_monitors() {
    for pid in "$monitor_pid" "$dmon_pid"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
      fi
    done
  }
  trap cleanup_monitors EXIT

  if [[ "$PROFILE" == "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "PROFILE=1 but nvidia-smi is missing." >&2
      exit 2
    fi
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 >"$GPU_CSV" &
    monitor_pid="$!"
    nvidia-smi dmon -i "$CUDA_VISIBLE_DEVICES" -s pucm -o DT -d 1 -f "$DMON_LOG" &
    dmon_pid="$!"
  fi

  "$SWIFT_BIN" sft "$CONFIG_PATH"

  cleanup_monitors
  trap - EXIT

  final_checkpoint="$(latest_checkpoint || true)"
  if [[ -n "$final_checkpoint" ]]; then
    find "$final_checkpoint" -maxdepth 1 -type f \( \
      -name '*.safetensors' \
      -o -name '*.bin' \
      -o -name 'adapter_config.json' \
      -o -name 'config.json' \
      -o -name 'generation_config.json' \
      -o -name 'tokenizer*' \
      -o -name 'vocab.json' \
      -o -name 'merges.txt' \
      -o -name 'args.json' \
      -o -name 'README.md' \
      -o -name 'training_args.bin' \
    \) -exec cp -f {} "$OUTPUT_DIR/" \;
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} +
  fi

  if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" || -f "$OUTPUT_DIR/model.safetensors" ]]; then
    echo "Final weights: $OUTPUT_DIR"
  fi

  if [[ "$PROFILE" == "1" ]]; then
    echo "GPU CSV: $GPU_CSV"
    echo "SM dmon: $DMON_LOG"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  write_config
  print_manifest
  echo "DRY_RUN=1, generated config only."
  exit 0
fi

if [[ "$AUTO_SETUP" == "1" ]]; then
  ensure_env
fi
if [[ ! -x "$PYTHON_BIN" || ! -x "$SWIFT_BIN" ]]; then
  echo "Missing python/swift in env: $ENV_DIR" >&2
  echo "Set AUTO_SETUP=1 or provide ENV_DIR/PYTHON_BIN/SWIFT_BIN." >&2
  exit 2
fi

if [[ "$AUTO_MODEL" == "1" ]]; then
  ensure_model
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model missing: $MODEL_DIR" >&2
  echo "Set AUTO_MODEL=1 or provide MODEL_DIR." >&2
  exit 2
fi

if [[ "$AUTO_DATASET" == "1" ]]; then
  ensure_dataset
fi
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "Dataset missing: $DATASET_PATH" >&2
  echo "Set AUTO_DATASET=1 or provide DATASET_PATH." >&2
  exit 2
fi

setup_runtime_env
write_config
print_manifest

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "PREPARE_ONLY=1, skipping training."
  exit 0
fi

run_train

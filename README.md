# qwen 3 quick

Standalone Qwen3-0.6B LoRA low-memory SFT launcher.

Main training entrypoint:

```bash
./qwen_3_quick.sh
```

Stop a running training job:

```bash
./pause_qwen_3_quick.sh
```

The launcher can install the env, download the model, download and convert a
Hugging Face dataset, generate the Swift config, and launch training. Paused
runs are not resumed; the next launch starts from scratch.

## One-Click Run

```bash
cd /mnt/bn/strategy-mllm-train/intern/users/weisong/repo/omni/qwen_3_quick
./qwen_3_quick.sh
```

Defaults:

- Env: `~/.venv/qwen_3_quick`
- Model: `~/models/Qwen3-0.6B`
- Dataset: `yahma/alpaca-cleaned`
- Converted JSONL: `~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl`
- Output: `./output/qwen3_quick_alpaca_lora_lowmem`
- Final weights: `./output/qwen3_quick_alpaca_lora_lowmem`

## Training Defaults

- 8 GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- No TP
- LoRA all-linear
- bf16
- flash attention
- gradient checkpointing
- max length 768
- batch size 1
- gradient accumulation 1
- `MAX_STEPS=100000000`
- `SAVE_STRATEGY=steps`
- `SAVE_STEPS=MAX_STEPS`
- `SAVE_ONLY_MODEL=true`

## Pause

Pause means stop the current training process and release GPUs. It does not keep
a resume checkpoint.

```bash
./pause_qwen_3_quick.sh
```

If training uses a custom output directory, pass the same value:

```bash
OUTPUT_DIR=/path/to/output ./pause_qwen_3_quick.sh
```

The next run starts from scratch:

```bash
./qwen_3_quick.sh
```

## Checkpoints

Intermediate checkpoints are disabled by default. The launcher sets
`SAVE_STEPS=MAX_STEPS`, so Swift only saves once at the final step. After a
normal completed run, the launcher flattens the final LoRA weights into
`OUTPUT_DIR` and removes `checkpoint-*`.

If you explicitly want checkpoint/resume behavior, enable it manually:

```bash
SAVE_STEPS=1000 SAVE_ONLY_MODEL=false RESUME=1 ./qwen_3_quick.sh
```

## Common Overrides

```bash
ENV_DIR=~/.venv/qwen_3_quick
MODEL_DIR=~/models/Qwen3-0.6B
DATASET_ID=yahma/alpaca-cleaned
DATASET_SPLIT=train
HF_DATASETS_CACHE=~/.cache/huggingface/datasets
DATASET_PATH=~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl
OUTPUT_DIR=./output/qwen3_quick_alpaca_lora_lowmem
MAX_STEPS=100000000
SAVE_STRATEGY=steps
SAVE_STEPS=100000000
SAVE_ONLY_MODEL=true
PROFILE=1
```

Prepare only:

```bash
PREPARE_ONLY=1 ./qwen_3_quick.sh
```

Config-only dry run:

```bash
DRY_RUN=1 ./qwen_3_quick.sh
```

Use a preconverted dataset:

```bash
AUTO_DATASET=0 DATASET_PATH=/path/to/train.jsonl ./qwen_3_quick.sh
```

The converted dataset schema is:

```json
{"system": "...", "query": "...", "response": "..."}
```

## Raw GPU/SM Logs

Profile is enabled by default. Raw logs are written to:

```bash
./output/gpu_profiles/
```

Files:

- `*_nvidia_smi.csv`: timestamp, GPU index, GPU util, memory used
- `*_dmon.log`: raw `nvidia-smi dmon` output including SM utilization

No profile summary script is included.

## Cleanup

Remove generated runtime outputs:

```bash
rm -rf output
```

Remove converted dataset:

```bash
rm -f ~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl
```

Remove the environment:

```bash
rm -rf ~/.venv/qwen_3_quick
```

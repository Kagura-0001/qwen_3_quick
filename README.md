# qwen 3 quick

Standalone one-file launcher for Qwen3-0.6B LoRA low-memory SFT.

The only public shell entrypoint is:

```bash
./qwen_3_quick.sh
```

It can install the env, download the model, download and convert a Hugging Face
dataset, generate the Swift config, launch training, and resume after
interruption.

## One-Click Run

```bash
cd /mnt/bn/strategy-mllm-train/intern/users/weisong/repo/omni/qwen_3_quick
./qwen_3_quick.sh
```

Defaults:

- Env: `~/.venv/qwen_3_quick`
- Model: `~/models/Qwen3-0.6B`
- Dataset: `yahma/alpaca-cleaned`
- Converted JSONL: `./data/alpaca_cleaned/train.jsonl`
- Output: `./output/qwen3_quick_alpaca_lora_lowmem`
- Final weights: `./output/qwen3_quick_alpaca_lora_lowmem/final_weight`

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
- `SAVE_STEPS=1000`
- `SAVE_TOTAL_LIMIT=2`

## Resume

Resume is enabled by default:

```bash
RESUME=1 ./qwen_3_quick.sh
```

The script reuses the stable output dir and automatically picks the latest:

```bash
./output/qwen3_quick_alpaca_lora_lowmem/checkpoint-*
```

To resume from a specific checkpoint:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-1000 ./qwen_3_quick.sh
```

Training interruption cannot be resumed unless real checkpoint directories
exist. For that reason the script saves resumable checkpoints during training.
It keeps at most two by default. When a run finishes successfully, the latest
weights are copied to `final_weight/`, and checkpoint directories are removed by
default.

If you want to keep checkpoints after successful completion:

```bash
CLEAN_CHECKPOINTS_AFTER_SUCCESS=0 ./qwen_3_quick.sh
```

## Common Overrides

```bash
ENV_DIR=~/.venv/qwen_3_quick
MODEL_DIR=~/models/Qwen3-0.6B
DATASET_ID=yahma/alpaca-cleaned
DATASET_SPLIT=train
DATASET_PATH=./data/alpaca_cleaned/train.jsonl
OUTPUT_DIR=./output/qwen3_quick_alpaca_lora_lowmem
MAX_STEPS=100000000
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=2
CLEAN_CHECKPOINTS_AFTER_SUCCESS=1
FINAL_WEIGHT_DIR=./output/qwen3_quick_alpaca_lora_lowmem/final_weight
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
rm -rf data output
```

Remove the environment:

```bash
rm -rf ~/.venv/qwen_3_quick
```

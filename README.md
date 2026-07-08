# qwen 3 quick

Standalone Qwen3 LoRA SFT launcher. The default path is tuned for 8x H100
utilization: Qwen3-4B + DDP + bf16 + flash attention.

The environment path installs flash-attn from a prebuilt wheel by default. It
does not compile flash-attn locally unless `FLASH_ATTN_INSTALL=source` is set
explicitly.

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
- Model: `~/models/Qwen3-4B`
- Dataset: `yahma/alpaca-cleaned`
- Converted JSONL: `~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl`
- Output: `./output/qwen3_quick_4b_ddp_lora_b8_l1024_flash`
- Final weights: `./output/qwen3_quick_4b_ddp_lora_b8_l1024_flash`

## Training Defaults

- 8 GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- No TP
- DDP: `DEEPSPEED=`; set `DEEPSPEED=zero3` for the low-memory path
- PyTorch 2.8.0 + CUDA 12.8 wheels
- flash-attn 2.8.3 official prebuilt wheel
- LoRA all-linear
- LoRA rank 8 / alpha 16
- bf16
- flash attention
- gradient checkpointing
- max length 1024
- batch size 8
- gradient accumulation 1
- `MAX_STEPS=100000000`
- `SAVE_STRATEGY=steps`
- `SAVE_STEPS=MAX_STEPS`
- `SAVE_ONLY_MODEL=true`

## Recommended High-Utilization Run

This is the current recommended one-click run:

```bash
git clone https://github.com/Kagura-0001/qwen_3_quick.git
cd qwen_3_quick
./qwen_3_quick.sh
```

It downloads `Qwen/Qwen3-4B` to `~/models/Qwen3-4B` and trains with DDP.
On this 8x H100 machine, the best current candidate is:

| Config | Peak memory | SM avg | GPU util avg | Steps/s |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B, DDP, length 1024, batch 8, rank 8 | 73.3 GiB | 90.2% active / 80.6% all | 90.3% active / 81.0% all | 0.974 |
| Qwen3-4B, ZeRO-3, length 1024, batch 4, rank 8 | 41.8 GiB | 69.6% active / 49.9% all | 66.6% active / 48.1% all | 0.583 |

The launcher starts GPU monitors before model loading, so short smoke tests
include startup and shutdown 0% samples. For long training runs, the all-sample
average converges toward the active training average as startup is amortized.

Use the 0.6B DDP path explicitly if you need the older fast/small baseline:

```bash
MODEL_ID=Qwen/Qwen3-0.6B \
MODEL_DIR=~/models/Qwen3-0.6B \
DEEPSPEED= \
LORA_RANK=32 \
LORA_ALPHA=64 \
RUN_ID=qwen3_quick_0p6b_lora_lowmem \
OUTPUT_DIR=./output/qwen3_quick_0p6b_lora_lowmem \
./qwen_3_quick.sh
```

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
TORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
TORCHAUDIO_VERSION=2.8.0
PYTORCH_CUDA=cu128
FLASH_ATTN_VERSION=2.8.3
FLASH_ATTN_WHEEL=                    # optional local wheel path or URL
FLASH_ATTN_INSTALL=wheel             # wheel by default; source only if explicit
MODEL_ID=Qwen/Qwen3-4B
MODEL_DIR=~/models/Qwen3-4B
DATASET_ID=yahma/alpaca-cleaned
DATASET_SPLIT=train
HF_DATASETS_CACHE=~/.cache/huggingface/datasets
DATASET_PATH=~/.cache/huggingface/datasets/qwen_3_quick/alpaca_cleaned/train.jsonl
RUN_ID=qwen3_quick_4b_ddp_lora_b8_l1024_flash
OUTPUT_DIR=./output/qwen3_quick_4b_ddp_lora_b8_l1024_flash
DEEPSPEED=
LORA_RANK=8
LORA_ALPHA=16
MAX_LENGTH=1024
BATCH_SIZE=8
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

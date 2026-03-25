# LIBERO Rollout Evaluation

## Prerequisites

```bash
conda activate flower
# LIBERO + deps should already be installed (submodule + pip install -e LIBERO/ --no-deps)
python -c "from libero.libero.envs import OffScreenRenderEnv; print('OK')"
```

## 1. Download Checkpoint from Leonardo

You need the Hydra config (`.hydra/`) and model weights (`model.safetensors`).
You do NOT need optimizer/scheduler/sampler/random_states files.

```bash
# Create local destination
mkdir -p checkpoints/runs/2026-03-17/03-23-38/checkpoint_30000

# Download Hydra config
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_P01_047/project/output/checkpoints/runs/2026-03-17/03-23-38/.hydra \
    checkpoints/runs/2026-03-17/03-23-38/

# Download model weights (main model)
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_P01_047/project/output/checkpoints/runs/2026-03-17/03-23-38/checkpoint_30000/model.safetensors \
    checkpoints/runs/2026-03-17/03-23-38/checkpoint_30000/

# Optional: download EMA weights (use with --use_ema)
# rsync -avP leonardo:/.../checkpoint_30000/model_1.safetensors \
#     checkpoints/runs/2026-03-17/03-23-38/checkpoint_30000/
```

Expected local structure:
```
checkpoints/runs/2026-03-17/03-23-38/
├── .hydra/
│   └── config.yaml
└── checkpoint_30000/
    └── model.safetensors
```

## 2. Run Evaluation

**Important:** `--checkpoint_dir` must be an **absolute path** (the Hydra config loader resolves paths relative to the inference wrapper source file, not the working directory).

### Smoke test (1 rollout, quick sanity check)

```bash

```

### Full evaluation (20 rollouts per task)

```bash
python -m flower_vla.eval.libero.libero_eval \
    --checkpoint_dir $(pwd)/checkpoints/meanflower_checkpoint_80000/ \
    --checkpoint_name checkpoint_80000 \
    --benchmark_name libero_10 \
    --n_eval 20 \
    --num_videos 3 \
    --pred_action_horizon 10 \
    --multistep 5 \
    --num_sampling_steps 4
```

### Cross-benchmark evaluation

```bash
# Replace libero_10 with: libero_spatial, libero_object, libero_goal
python -m flower_vla.eval.libero.libero_eval \
    --checkpoint_dir $(pwd)/checkpoints/runs/2026-03-17/03-23-38 \
    --checkpoint_name checkpoint_30000 \
    --benchmark_name libero_spatial \
    --n_eval 20
```

## 3. CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_dir` | required | Path to run dir containing `.hydra/` |
| `--checkpoint_name` | required | Checkpoint folder name (e.g. `checkpoint_30000`) |
| `--benchmark_name` | `libero_10` | `libero_10`, `libero_spatial`, `libero_object`, `libero_goal` |
| `--n_eval` | `20` | Rollouts per task |
| `--max_steps` | `520` | Max env steps per episode |
| `--num_videos` | `3` | Rollouts to record per task |
| `--device` | `0` | CUDA device index |
| `--pred_action_horizon` | `10` | Action chunk size |
| `--multistep` | `5` | Steps before re-predicting |
| `--num_sampling_steps` | `1` | Flow matching sampling steps (1 for MeanFlow) |
| `--use_ema` | `false` | Load EMA weights (`model_1.safetensors`) |
| `--use_torch_compile` | `false` | Compile model with `torch.compile` |
| `--ensemble_strategy` | `false` | `false`, `act`, or `cogact` |
| `--cfg_lambda` | `1.0` | Classifier-free guidance weight |
| `--wandb_project` | `flower_libero_eval` | W&B project name |
| `--seed` | `42` | Random seed |
| `--log_dir` | `results/libero` | Output directory for videos and logs |

## 4. Output

- **Terminal**: Per-task success rates + overall average
- **W&B**: Per-task success rates, running mean, results table, rollout videos
- **Local**: `results/libero/videos/<task_name>/ep<N>_ok.mp4` or `ep<N>_fail.mp4`

## 5. Reference Numbers

| Benchmark | FLOWER paper (reported) |
|---|---|
| LIBERO-10 | 94.5% |
| LIBERO-Spatial | - |
| LIBERO-Object | - |
| LIBERO-Goal | - |

# FLOWER + Improved MeanFlow (iMF)

**Single-step flow-matching action heads for Vision-Language-Action policies.**

KTH Royal Institute of Technology — Master's thesis, Emil Hed (2026).

This repository extends [FLOWER](https://www.arxiv.org/pdf/2509.04996), a compact
(~950M parameter) Vision-Language-Action (VLA) policy, by replacing its multi-step
**Rectified Flow (RF)** action head with a single-step **Improved MeanFlow (iMF)**
head. It contains the pretraining pipeline and the LIBERO evaluation used in the
thesis. CALVIN fine-tuning and evaluation live in the sibling repository
[flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin).

---

## Motivation

The diffusion and flow-matching action heads used in modern VLA policies achieve
strong performance, but they typically require **multiple sequential sampling
steps** to generate each action, which increases inference latency on
resource-constrained robots.

This thesis asks: *can the multi-step Rectified Flow head in FLOWER be replaced
with a single-step Improved MeanFlow head without sacrificing task performance?*

**Answer.** Across both benchmarks the single-step iMF head matches the multi-step
RF baseline on task success — the two are statistically interchangeable on LIBERO,
and iMF stays competitive at matched one-step compute on CALVIN while recovering
RF's best multi-step performance with a *single* sampling step. End-to-end latency
drops by roughly **2×** relative to the published four-step default. One-step
MeanFlow heads are a viable lower-latency replacement for multi-step flow heads in
VLA policies.

---

## What's new versus upstream FLOWER

The iMF head predicts an **average** velocity field, enabling one-step (1-NFE)
sampling instead of RF's iterative integration:

- **Dual-head architecture** — a shared DiT trunk feeding parallel `u`-head
  (average velocity) and `v`-head (instantaneous velocity, used as a regularizer).
- **Compound velocity target** `V = u + h·sg(du/dt)`, where `h = t − r` is the
  integration interval, `sg(·)` is stop-gradient, and `du/dt` is a Jacobian
  correction computed via a Jacobian-vector product (JVP).
- **Single-step inference** (1 NFE) vs. RF's 4 sampling steps.
- Production config: integral-mode fraction `ρ = 0.5`, head depth `L_head = 8`.

Implementation:

- `flower_vla/agents/flower.py` — RF baseline (upstream FLOWER head).
- `flower_vla/agents/meanflower.py` — iMF / MeanFlow head.
- `flower_vla/agents/networks/meanflower_transformers.py` — JVP-aware
  transformer backbone (flash-attention Triton kernel).
- `CHANGELOG.md` — detailed development log of the iMF improvements.

---

## Results

All evaluations are **single-seed** and **simulation-only** (no real-robot
validation). Latency measured on a single RTX 4090, batch size 1, action chunk
H = 10.

### Inference latency (RTX 4090)

| Variant | NFE | VLM (ms) | FlowTransformer (ms) | Total (ms) | Speed-up |
|---------|----:|---------:|---------------------:|-----------:|---------:|
| RF      | 4   | 22.10    | 46.20                | **67.00**  | 1.00× (ref) |
| RF      | 3   | 22.16    | 34.55                | 56.82      | 1.18× |
| RF      | 2   | 22.13    | 23.18                | 45.27      | 1.48× |
| RF      | 1   | 22.07    | 11.13                | 33.30      | 2.01× |
| **iMF** | **1** | 22.00  | 10.60                | **32.85**  | **2.04×** |

At matched one-step compute, RF@1 and iMF@1 cost essentially the same
(33.30 vs 32.85 ms) — the 2× speed-up comes from removing sampling steps, not from
a cheaper head.

### LIBERO (in-distribution)

Per-suite success rate (%), 20 rollouts/task (200 per suite). LIBERO tasks appear
in the pretraining mix, so these results are largely in-distribution.

| Suite    | RF@4 | iMF@1 | Δ |
|----------|-----:|------:|---:|
| Spatial  | 99.0 | 98.0  | −1.0 |
| Object   | 98.0 | 96.0  | −2.0 |
| Goal     | 95.0 | 96.0  | +1.0 |
| Long-10  | 91.0 | 96.0  | +5.0 |
| **Macro avg** | **95.8** | **96.5** | **+0.7** |

The two heads are statistically interchangeable (no significant per-suite
differences; Long-10 borderline at p ≈ 0.053).

### CALVIN (out-of-distribution)

CALVIN is entirely absent from pretraining, making it a strict OOD test. Metric:
mean rollout chain length L̄ ∈ [0, 5] (and success-at-K), 1000 paired chains.
**These numbers were produced in the sibling
[flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)
repository.**

At matched one-step compute (default `ρ = 0.5`), RF@1 and iMF@1 reach statistical
parity on both the in-environment (CALVIN-D) and cross-environment (ABC→D) splits.
A `ρ = 0.75` iMF variant — tuned post-hoc on CALVIN-D — opens a long-horizon
advantage that grows monotonically with chain length:

| Split    | Variant            |  L̄   | Δ vs RF@1 | K=5 success |
|----------|--------------------|-----:|----------:|------------:|
| CALVIN-D | RF@1               | 4.100 | —        | 66.4 |
| CALVIN-D | iMF@1 (ρ=0.5)      | 4.041 | −0.059   | 64.9 |
| CALVIN-D | iMF@1 (ρ=0.75)     | **4.307** | **+0.207** | **74.2** (+7.8pp) |
| ABC→D    | RF@1               | 4.208 | —        | 67.0 |
| ABC→D    | iMF@1 (ρ=0.5)      | 4.182 | −0.026   | 65.3 |
| ABC→D    | iMF@1 (ρ=0.75)     | 4.278 | +0.070   | 70.7 (+3.7pp) |

Tuned iMF@1 also recovers RF's best *multi-step* (RF@4) chain length on CALVIN-D
(4.307 vs 4.350) and exceeds nominal RF@4 on ABC→D (4.278 vs 4.214) — with a
single sampling step. The K=5 gain on CALVIN-D is significant
(McNemar p ≈ 1×10⁻⁵).

**Takeaway.** One-step iMF matches multi-step RF on task success at ~2× lower
latency; its sharper single-step actions especially help long-horizon OOD chains.

**Caveats.** All fine-tunes are single-seed and simulation-only; the strongest
CALVIN-D advantage uses `ρ = 0.75` tuned post-hoc on CALVIN-D itself (the ABC→D
column applies that value unchanged as a robustness check). iMF also produces
measurably jerkier actions than multi-step RF (the precision/smoothness
trade-off).

---

## Repository layout

```
flower_vla/
  agents/
    flower.py                          # RF baseline head
    meanflower.py                      # iMF / MeanFlow head
    networks/
      transformers.py                  # standard DiT backbone
      meanflower_transformers.py       # JVP-aware backbone (Triton flash attn)
    lang_encoders/                     # Florence-2 token encoder
    utils/                             # EMA, geometry, optimizer hooks
  dataset/
    oxe/                               # Open X-Embodiment mixes, configs, transforms
    datamodule.py, dataset.py          # data loading
  eval/
    libero/libero_eval.py              # LIBERO benchmark evaluation
    simpler/, kitchen/                 # optional SimplerEnv / kitchen eval
  training.py, finetuning.py           # entry points
conf/
  flower_training.yaml                 # RF baseline pretraining
  iMF_training.yaml                    # Improved MeanFlow pretraining
  meanflower_training.yaml             # MeanFlow pretraining
  finetuning.yaml
scripts/leonardo/                      # SLURM launchers (CINECA cluster)
docs/                                  # CINECA setup, LIBERO eval notes
```

---

## Installation

### Requirements
- Python 3.10
- CUDA 11.8+
- 24 GB+ GPU memory for training (more is better)
- <8 GB GPU memory for inference

### Setup
```bash
conda create -n flower python=3.10
conda activate flower

git clone --recurse-submodules <repo-url>
cd flower_vla_pret

pip install -r requirements_train.txt   # training
# pip install -r requirements_simpler.txt  # for SimplerEnv eval
```

> **SimplerEnv** evaluation is optional. If you want it, initialize the submodule
> with `git submodule update --init --recursive`.

---

## Pretraining

> **All thesis experiments were run on the CINECA [Leonardo](https://www.hpc.cineca.it/systems/hardware/leonardo/)
> supercomputer**, on the `boost_usr_prod` partition (1 node × 4 A100 64 GB GPUs)
> using the SLURM `sbatch` scripts in [`scripts/leonardo/`](scripts/leonardo/).
> The recommended way to reproduce the runs is via those scripts (below); the raw
> `accelerate launch` commands they wrap are documented afterwards for local use.

FLOWER uses HuggingFace `accelerate` for multi-GPU training with Hydra configs.

### Running on CINECA Leonardo (SLURM)

The scripts assume the project lives across Leonardo's two storage tiers — code,
checkpoints and wandb on `$LEONARDO_FAST`, the venv/datasets/HF cache on
`$LEONARDO_WORK` — and a pip-installed virtualenv (no `cineca-ai` module). Set the
allocation paths once, then submit:

```bash
export LEONARDO_FAST=/leonardo_scratch/fast/<ACCOUNT>
export LEONARDO_WORK=/leonardo_work/<ACCOUNT>

# Quick 30-min sanity job (boost_qos_dbg) before long runs
sbatch scripts/leonardo/sbatch_debug.sh    [meanflower | flower]

# Full pretraining run (positional arg selects the head)
sbatch scripts/leonardo/sbatch_train.sh    [meanflower | flower | imf]

# Fine-tuning
sbatch scripts/leonardo/sbatch_finetune.sh

# iMF ablations (Hydra overrides)
sbatch scripts/leonardo/sbatch_ablation.sh [ratio05 | heads12 | both]

squeue -u $USER                 # monitor
tail -f flowervla_*.out         # logs
```

Step-by-step login, environment activation, log-watching and checkpoint-download
instructions are in [`docs/CINECA.md`](docs/CINECA.md). One-time data/model setup
helpers also live in `scripts/leonardo/` (`setup.sh`, `download_oxe_data.sh`,
`download_hf_models.py`).

### Running locally / on other clusters

The sbatch scripts ultimately call `accelerate launch` on the Hydra entry point.
To run directly:

### Dataset preparation
Create a central dataset directory and download the datasets not bundled in the
public Google Cloud mix:
```bash
export DATA_DIR=~/tensorflow_datasets

# Bridge (Berkeley), not part of OXE
wget -r -np -nd -A '*' \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/ \
  -P $DATA_DIR/bridge_dataset
```

### Accelerate config
```bash
accelerate config   # multi-GPU, bf16 mixed precision, DDP
```

### Train
```bash
# Improved MeanFlow (iMF)
accelerate launch flower_vla/training.py --config-name=iMF_training

# RF baseline (upstream FLOWER)
accelerate launch flower_vla/training.py --config-name=flower_training
```

Resume from a checkpoint:
```bash
accelerate launch flower_vla/training.py --config-name=iMF_training \
  +step=100000 +continue_training=/path/to/checkpoint_100000
```

On Leonardo the equivalent run is `sbatch scripts/leonardo/sbatch_train.sh imf`
(see above), which sets the same Hydra config and launches `accelerate` across the
4 GPUs.

### Debugging the data pipeline
The TensorFlow dataset transforms are easier to debug directly:
```bash
python flower_vla/test_dataloader.py
python flower_vla/debug_transforms.py
```

---

## Evaluation

LIBERO evaluation is driven by `flower_vla/eval/libero/libero_eval.py`. See
[docs/libero_eval.md](docs/libero_eval.md) for setup and per-suite details. For
CALVIN, use the [flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)
repository.

---

## Custom dataset mixes

The OXE data code is based on [Octo](https://github.com/octo-models/octo) and
[OpenVLA](https://github.com/openvla/openvla). To add a dataset:

1. Define a config in `flower_vla/dataset/oxe/configs.py`
2. Define a transform in `flower_vla/dataset/oxe/transforms.py`
3. Add its control frequency to `flower_vla/dataset/utils/frequency_mapping.py`
4. Register it in `flower_vla/dataset/utils/dataset_index.py`
5. Set its action-chunk length in `flower_vla/dataset/utils/act_seq_mapping.py`

Edit mixes in `flower_vla/dataset/oxe/mixes.py`. Use `debug_transforms.py` to test.

---

## Citation

This work builds on FLOWER:

```bibtex
@inproceedings{reuss2025flower,
  title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
  author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025},
  url={https://openreview.net/forum?id=JeppaebLRD}
}
```

The MeanFlow head is based on the MeanFlow / Improved MeanFlow formulation
(Geng et al., 2025).

If you use this thesis work, please also cite:

```bibtex
@mastersthesis{hed2026imf,
  title  = {Single-Step Flow-Matching Action Heads for Vision-Language-Action Policies},
  author = {Emil Hed},
  school = {KTH Royal Institute of Technology},
  year   = {2026},
  note   = {TRITA TBD}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This work builds on the following open-source projects and datasets:

- [FLOWER VLA](https://github.com/intuitive-robots/flower_vla_calvin) (Intuitive Robots Lab, KIT)
- [Octo](https://github.com/octo-models/octo)
- [OpenVLA](https://github.com/openvla/openvla)
- [mimictest](https://github.com/EDiRobotics/mimictest) by [Starcycle](https://github.com/StarCycle)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and CALVIN benchmarks

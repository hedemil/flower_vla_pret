# MeanFlower VLA - Changelog

## [2026-03-13] Integrate Decoupled MeanFlow (DMF) with log-variance loss weighting

**Motivation**: Standard iMF uses a single DiT with shared t/h conditioning and adaptive loss normalization `loss/(loss+eps)^p`. DMF (Kyungmin Lee, 2025) decouples the DiT into encoder (t-conditioned) and decoder (r-conditioned) blocks, enabling cleaner single-step inference at `r=0`. It also replaces the adaptive weighting with a learned log-variance head, which provides per-sample loss scaling without the hand-tuned `eps`/`p` hyperparameters that were a recurring source of instability (see entries from 2026-03-01 and 2026-03-10).

**Architecture changes**:

| Component | Before (iMF) | After (DMF) |
|-----------|-------------|-------------|
| DiT backbone | `nn.ModuleList` of 12 FlowBlocks, all conditioned on `t` | `DMFTransformer`: 8 encoder blocks (t-conditioned) + 4 decoder blocks (r-conditioned) |
| Timestep conditioning | `t_embedder` only; `h = t - r` passed to MeanFlowDecoder | `t_embedder` (encoder) + `r_embedder` (decoder); `r` passed directly |
| AdaLN | Single `adaln` / `adaln` dict | `adaln_t` (encoder) + `adaln_r` (decoder), or per-action `adaln` dict called twice |
| Action decoder | Plain `nn.Linear` | Same — r-conditioning now handled by decoder blocks |
| Loss weighting | Adaptive `loss/(loss.detach()+eps)^p` | Learned log-variance: `log(mse/exp(lv) + eps) + lv` |
| Loss structure | Single MF branch with JVP | Dual-loss: FM branch (r=t, no JVP) + MF branch (r≠t, JVP) |
| Time sampling | Single logit-normal, then split by ratio | Separate logit-normals for t and r with independent means |
| Inference | `u(z, t=1, h=1)` | `u(z, t=1, r=0)` |

**Log-variance head**: Stateless sinusoidal embeddings of `(t, r)` → concat → `nn.Linear(256, 1)` → scalar `lv`. Zero-initialized so `exp(lv)=1` at init (no scaling). Only 257 new learnable parameters.

**Weight mapping**: `map_flower_to_meanflower()` maps FlowerVLA checkpoint keys to DMF format:
- `dit.{i}.*` → `dit.encoder_blocks.{i}.*` (layers 0-7) or `dit.decoder_blocks.{i-8}.*` (layers 8-11)
- `t_embedder.*` → copied to `r_embedder.*`
- Per-action `adaln.*` keys kept as-is (shared structure with `action_type_adaln=True`)
- `logvar_linear` zero-init'd by constructor, no pretrained weights needed

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Remove `MeanFlowDecoder` import | `meanflower.py` | Class doesn't exist, already using `nn.Linear` |
| Remove dead `self.adaln` (shared path) | `meanflower.py` | Was created then overwritten by `adaln_t`/`adaln_r` |
| Add `logvar_timestep_embedding()`, `log_lv_loss()` | `meanflower.py` | Module-level utilities for DMF loss |
| Add `logvar_linear` component | `meanflower.py` | Zero-init learned log-variance head |
| Rewrite `dit_forward_meanflow(h→r, +return_logvar)` | `meanflower.py` | Pass `r` directly to `r_embedder`; optionally return log-variance |
| Rewrite `meanflow_loss` with DMF dual-loss | `meanflower.py` | FM branch (r=t) + MF branch (r≠t, JVP), both use `log_lv_loss` |
| Fix sampling: `h=1` → `r=0` | `meanflower.py` | DMF single-step inference uses `r=0` |
| Update `meanflow_eval_loss` | `meanflower.py` | Use `sample_times()`, pass `r` directly |
| Remove deprecated `sample_tr` | `meanflower.py` | Replaced by `sample_times()` |
| Add `map_flower_to_meanflower()` | `utils/model_loading.py` | Structural key mapping for checkpoint loading |
| Add `load_pretrained_weights()` with `map_type` | `utils/model_loading.py` | Generic loading with optional key transformation |
| Thread `map_type` through trainer | `flower_trainer.py`, `finetuning.py` | End-to-end wiring from config to weight mapper |
| Add `map_type: flower_to_dmf` | `finetuning.yaml` | Enable structural mapping when loading FlowerVLA checkpoint |
| Add DMF distribution params | `meanflower_vla.yaml` | `P_mean_t=-0.2`, `P_std_t=1.0`, `P_mean_r=0.2`, `P_std_r=1.0` |

**New wandb metrics:**

| Metric | What it measures |
|--------|-----------------|
| `fm_mse` | FM branch MSE (flow matching quality) |
| `mf_mse` | MF branch MSE (mean flow quality) |
| `fm_log_loss` / `mf_log_loss` | Log-variance weighted losses per branch |
| `lv_fm` / `lv_mf` | Mean log-variance per branch (tracks learned scaling) |
| `dudt_norm` | du/dt magnitude (MF branch) |
| `cos_u_v` | Cosine similarity u vs v (single-step convergence) |

**Files changed**: `flower_vla/agents/meanflower.py`, `flower_vla/agents/networks/meanflower_transformers.py`, `flower_vla/utils/model_loading.py`, `flower_vla/trainers/flower_trainer.py`, `flower_vla/finetuning.py`, `conf/finetuning.yaml`, `conf/trainer/agent/meanflower_vla.yaml`

## [2026-03-11] Add MeanFlow v_loss during eval, fix double-squeeze bug

**Problem**: Val loss (action-space MSE from single-step sampling) increases monotonically despite training `v_loss` improving. The two metrics are **fundamentally different**: training measures `||u - u_tgt||²` in latent space with adaptive normalization; eval measures action-space MSE. We need a comparable eval metric to determine whether the model is actually overfitting or if the sampling procedure is the bottleneck.

**Bugfix**: `ddp_wrapper.forward()` mutates `batch[target_modality]` in-place via `[:, -1]` (squeezing history dim). When `evaluate()` calls `evaluate_step` multiple times on the same batch (EMA → online → MF_EMA → MF_online), the second+ calls squeeze an already-squeezed tensor (4D→3D→2D), causing `IndexError: too many indices for tensor of dimension 2`. This was a latent bug from the recent dual-eval (EMA + online) addition.

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Add `EVALUATION_MF = 4` mode | `ddp_wrapper.py` | New eval mode that calls `meanflow_eval_loss_step` |
| Guard `discard_action_history` with `.dim() == 4` | `ddp_wrapper.py` | Prevent double-squeeze when same batch is reused across eval calls |
| Add `meanflow_eval_loss_step()` | `meanflower.py` | Wrapper: encode observations, delegate to core loss, attach `dataset_index` |
| Add `meanflow_eval_loss()` | `meanflower.py` | Core: sample `t,r`, build `z_t`, single forward pass, compute `\|\|u - v\|\|²` per sample (no JVP needed) |
| Add `mf_metrics_tracker` / `mf_online_metrics_tracker` | `flower_trainer.py` | Track per-dataset v_loss for EMA and online weights |
| Add `mode` param to `evaluate_step()` | `flower_trainer.py` | Pass through to `ddp_wrapper` instead of hardcoded `Mode.EVALUATION` |
| Run MF eval in `evaluate()` loop | `flower_trainer.py` | Two extra calls per batch (EMA + online) with `Mode.EVALUATION_MF` |
| Merge MF metrics in `compute_and_log_metrics()` | `flower_trainer.py` | Log all four trackers to wandb |

**New wandb metrics:**

| Metric | What it measures |
|--------|-----------------|
| `val_vloss/overall` | `\|\|u - v\|\|²` with EMA weights — comparable to training `v_loss` |
| `val_vloss_online/overall` | `\|\|u - v\|\|²` with online weights |

**Files changed**: `flower_vla/agents/ddp_wrapper.py`, `flower_vla/agents/meanflower.py`, `flower_vla/trainers/flower_trainer.py`

## [2026-03-10] Fix inverted ratio semantics, align with py-meanflow reference

**Problem**: Training metrics (raw_mse, cos_u_utgt) kept improving but val_loss **increased** over training: 0.175 (10k) → 0.271 (20k) → 0.313 (30k). Per-dataset val_losses were also identical across all tasks (~0.123) due to a separate metrics bug.

**Root causes**:

1. **Inverted `data_proportion` semantics.** Our `data_proportion=0.75` meant 75% of samples got `r=t` (h=0, instantaneous velocity) and only 25% were integral (h>0). The py-meanflow reference's `ratio=0.75` means the **opposite**: 75% integral, 25% velocity. The model was trained almost exclusively at h≈0 but evaluated with single-step sampling at h=1 — completely out-of-distribution. As the model specialized for h≈0 during training, its h=1 predictions degraded.
2. **Adaptive loss too aggressive.** `norm_p=1.0` with `norm_eps=0.01` made `loss/(loss+0.01)^1.0 ≈ 1.0` for all samples, flattening all gradient signals. Reference uses `norm_p=0.75`, `norm_eps=0.001` for softer normalization that preserves relative loss magnitudes.
3. **Per-dataset val_loss bug.** `validation_step` returned a scalar loss that got broadcast to all samples in `DatasetMetricsTracker`, making all per-dataset losses identical.

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Rename `data_proportion` → `ratio`, flip semantics | `meanflower.py` | Match py-meanflow reference: `ratio=0.75` = 75% integral (h>0), 25% velocity (h=0) |
| Stochastic per-sample masking in `sample_tr` | `meanflower.py` | Reference uses `torch.rand` per sample, not deterministic first-N |
| `norm_p`: 1.0 → **0.75**, `norm_eps`: 0.01 → **0.001** | `meanflower.py` | Match CIFAR10 reference config for softer adaptive weighting |
| Return per-sample loss from `validation_step` | `meanflower.py`, `flower.py` | Fix per-dataset metrics — `DatasetMetricsTracker` needs `[B]` losses, not scalar |

**Config changes** (`conf/trainer/agent/meanflower_vla.yaml`):

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `data_proportion: 0.75` | 75% velocity, 25% integral | **`ratio: 0.75`** = 75% integral, 25% velocity | Match py-meanflow CIFAR10 config (closest dataset size) |

**Reference comparison** (py-meanflow CIFAR10 v0):

| Parameter | Reference | Ours (after) |
|-----------|-----------|-------------|
| ratio (integral %) | 0.75 (75%) | 0.75 (75%) |
| norm_p | 0.75 | 0.75 |
| norm_eps | 0.001 | 0.001 |
| ema_decay | 0.9999 | 0.999 |
| P_mean | -2.0 | -0.4 |
| P_std | 2.0 | 1.0 |

**Files changed**: `flower_vla/agents/meanflower.py`, `flower_vla/agents/flower.py`, `conf/trainer/agent/meanflower_vla.yaml`

## [2026-03-09] Detach t/h conditioning in JVP, zero-init, increase weight decay

**Problem**: Mean Flow training metrics (raw_mse ~1000+, cos_u_utgt ~0.3, dudt_norm ~2000-5000) were inflated because the JVP tangent flowed through the sinusoidal t-embedder and h-embedder, creating enormous ∂u/∂t that dominated the self-consistency target. Val_loss was competitive (0.0659 vs FLOWER 0.0672 at 20k) but the model overfitted after 10k steps (0.1178 → 0.1691 at 20k).

**Root causes**:

1. **JVP tangent through t/h embedders.** The `u_func` passed raw `t_input` and `h_input` to the network, so `torch.func.jvp` computed `du/dt` including the sinusoidal embedder path — not just the physically meaningful `∂u/∂z · v` term.
2. **Non-zero initialization.** adaLN modulation and MeanFlowDecoder output layers had random init, so blocks were not identity and decoder output was non-zero at step 0.
3. **Weak transformer weight decay.** 0.01 vs FLOWER's 0.1 — insufficient regularization without dropout.

**Results after fix** (at 4k steps): dudt_norm 244 (was 2000-5000), cos_u_utgt 0.89 (was 0.3), raw_mse 25.8 (was 1000+).

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Detach `t_input` and `h_input` in `u_func` | `meanflower.py` | Zero JVP tangent through t/h embedders, leaving only ∂u/∂z · v |
| Zero-init `FlowBlock.adaLN_modulation[-1]` | `meanflower_transformers.py` | Blocks start as identity (shift=0, scale=0, gate=0) |
| Zero-init `MeanFlowDecoder.decoder[-1]` | `meanflower_transformers.py` | u_pred = 0 at initialization |
| Dtype-aware fill value in cross-attention | `meanflower_transformers.py` | Consistent bf16-safe `-inf` masking |

**Config changes** (`conf/trainer/meanflower_trainer.yaml`):

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| transformer_weight_decay | 0.01 | **0.1** | Match FLOWER baseline for proper regularization |

**Files changed**: `flower_vla/agents/meanflower.py`, `flower_vla/agents/networks/meanflower_transformers.py`, `conf/trainer/meanflower_trainer.yaml`

## [2026-03-07] Fix overfitting: LR decay, early stopping, best model saving

**Problem**: Training metrics (v_loss, cos_u_v) converge well, but val_loss diverges after step 20k (0.0799 → 0.2706 by 50k). The model overfits because LR never decays during training and there is no mechanism to stop or save the best checkpoint.

**Root causes**:

1. **LR never decays.** TriStageLRScheduler had `total_steps=600k` with `phase_ratio=(0.01, 0.39, 0.6)`, meaning warmup ends at 6k and hold continues until 240k. At 50k steps the model is still at full LR (1e-4).
2. **No early stopping.** Training runs for `max_train_steps` regardless of val_loss.
3. **No best model saving.** Only periodic checkpoints every 10k steps — best model may be overwritten.
4. **MeanFlowDecoder has no regularization.** Dropout cannot be used (breaks JVP), and weight decay was the same low 0.01 as the transformer.

**Config changes** (`conf/trainer/meanflower_trainer.yaml`):

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| DiT LR total_steps | 600000 | **100000** | Start decay much sooner to match best val_loss at ~20k |
| DiT LR phase_ratio | (0.01, 0.39, 0.6) | **(0.03, 0.17, 0.8)** | Warmup 0→3k, hold 3k→20k, cosine decay 20k→100k |
| VLM LR total_steps | 400000 | **100000** | Proportional adjustment |
| VLM LR phase_ratio | (0.1, 0.89, 0.01) | **(0.05, 0.15, 0.8)** | Warmup 0→5k, hold 5k→20k, cosine decay 20k→100k |
| decoder_weight_decay | (none) | **0.05** | Higher regularization for decoder (only lever since dropout breaks JVP) |
| early_stopping_patience | (none) | **5** | Stop after 5 evals (50k steps) without val_loss improvement |

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Early stopping + best model saving | `flower_trainer.py` | Track `best_val_loss` and `patience_counter`. Save best checkpoint on improvement, break loop when patience exhausted. |
| Decoder weight decay param group | `meanflower.py` | Separate `action_decoders` params into own optimizer group with `decoder_weight_decay` (0.05 vs 0.01). |

**Files changed**: `conf/trainer/meanflower_trainer.yaml`, `flower_vla/trainers/flower_trainer.py`, `flower_vla/agents/meanflower.py`

## [2026-02-28] Fix MeanFlow training instability

**Problem**: Training collapses after ~20k steps. Val MSE: 0.127 (20k) → 0.562 (30k), a 4.4x regression. The MeanFlow loss is self-referential (`u_tgt = v - h * du/dt`), making it sensitive to hyperparameters that standard rectified flow tolerates.

**Root cause**: Several hyperparameters diverged from the MeanFlow paper (Geng et al., 2025) and FLOWER paper (Reuss et al., 2025).

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| DiT weight decay | 0.1 | **0.0** | MeanFlow paper. Weight decay shifts the bootstrap target `u_tgt` by shrinking weights that `du/dt` depends on. |
| attn_pdrop | 0.3 | **0.0** | MeanFlow paper. High dropout during JVP creates noisy `du/dt` derivatives. |
| resid_pdrop | 0.1 | **0.0** | MeanFlow paper uses zero dropout everywhere. |
| mlp_pdrop | 0.1 | **0.0** | MeanFlow paper uses zero dropout everywhere. |
| EMA | False | **True (0.9999)** | MeanFlow paper. Smooths weight updates, prevents val loss spikes. |
| DiT LR scheduler | InverseSqrt | **TriStage (warmup+const+cosine)** | FLOWER paper. Phases [0.01, 0.39, 0.6] over 600k steps. |
| n_layers | 12 | **18** | FLOWER paper Table 5. More capacity for the harder MeanFlow objective. |
| n_heads | 8 | **16** | FLOWER paper Table 5. |
| VLM LR | 2e-5 | **1e-5** | FLOWER paper Table 8. |
| VLM weight decay | 1e-9 | **0.001** | FLOWER paper Table 8. |

**Files changed**: `conf/trainer/agent/meanflower_vla.yaml`, `conf/trainer/meanflower_trainer.yaml`

## [2026-03-01] Fix OOM from larger DiT + EMA

**Problem**: CUDA OOM on 4x A100 64GB. The larger DiT (18 layers, 16 heads) combined with EMA and `torch.func.jvp` (which doubles forward pass memory for dual numbers) exceeded 64GB per GPU.

**Fix**: Halve `batch_size` and double `gradient_accumulation_steps` to preserve the same effective batch size while reducing peak activation memory.

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| batch_size | 600 | **300** | Halved to fit in 64GB with larger DiT + EMA + JVP |
| gradient_accumulation_steps | 2 | **4** | Doubled to keep effective batch size at 1200 |

**Files changed**: `conf/training.yaml`

## [2026-03-01] Fix val MSE stall: scale mismatch between MeanFlow ImageNet params and 128-dim actions

**Problem**: Training v_loss decreases steadily (1.18 → 0.26 over 25k steps), but validation MSE is flat at ~0.673 across all 9 datasets — indistinguishable from predicting the action mean. The previous smaller DiT (12 layers, 8 heads) reached val MSE 0.127 at 20k before collapsing.

**Root cause analysis** (with input from senior DL engineer):

1. **Overparameterized DiT for the output space.** The 18-layer, 1024-dim DiT (~310M params) was taken from FLOWER Table 5, which uses standard rectified flow. MeanFlow's self-referential loss `u_tgt = v - h * du/dt` allows a large model to find a "flat" local minimum where `du/dt ≈ 0`. In this regime `u ≈ v` (instantaneous velocity only), which satisfies the training loss but makes single-step sampling `z_0 = z_1 - u(z_1, 1, 1)` fail — it degenerates to a standard flow that needs multi-step ODE integration. The 128-dim action output (16 timesteps x 8 action dims) is 1500x smaller than ImageNet 256x256 — the model capacity vastly exceeds what's needed.

2. **Adaptive loss masking gradient magnitude.** The adaptive weighting `loss / (loss + eps)^p` normalizes every sample's loss to ≈1.0, turning the optimizer into an effective sign-based update. Early in training when the error is large, this suppresses the gradient signal needed to learn the `du/dt` correction term that distinguishes MeanFlow from standard flow.

3. **EMA decay too slow for faster-converging task.** `decay=0.9999` was tuned for ImageNet training over millions of steps. With a 128-dim target that converges faster, the EMA weights lag too far behind the online weights, producing poor validation predictions when evaluating with EMA.

4. **Zero weight decay + zero dropout = no regularization.** MeanFlow paper disables dropout because stochastic masks create noise in `du/dt` during JVP — this is correct. But with zero weight decay AND zero dropout, there is no regularization at all. For a low-dimensional output with fewer training samples than ImageNet, moderate weight decay provides necessary regularization without interfering with JVP.

**Changes:**

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| n_layers | 18 | **12** | Reduce overparameterization. Prevents flat `du/dt` solution. Previous 12-layer model learned faster (val MSE 0.127 at 20k). |
| dit_dim | 1024 | **768** | Right-size capacity for 128-dim output. Still larger than DiT-S used in 3D MNIST MeanFlow experiments. |
| n_heads | 16 | **12** | Match reduced dim (768/12 = 64 head dim, same as before). |
| transformer_weight_decay | 0.0 | **0.01** | Regularization without interfering with JVP (unlike dropout). Prevents overfitting in the low-dim regime. |
| EMA decay | 0.9999 | **0.999** | Faster EMA adaptation for a task that converges faster than ImageNet. |
| batch_size | 300 | **600** | Reverted: smaller model fits in 64GB again. |
| gradient_accumulation_steps | 4 | **2** | Reverted with batch_size. Effective batch size stays at 1200. |

**Code changes:**

| Change | File | Rationale |
|--------|------|-----------|
| Log `dudt_norm` metric | `meanflower.py`, `flower_trainer.py` | Diagnose whether `du/dt` is vanishing or exploding. |
| Add `_train_step` counter | `meanflower.py` | Internal step tracking. |

**Files changed**: `conf/trainer/agent/meanflower_vla.yaml`, `conf/trainer/meanflower_trainer.yaml`, `conf/training.yaml`, `flower_vla/agents/meanflower.py`, `flower_vla/trainers/flower_trainer.py`

## [2026-03-01] Revert adaptive loss warmup — du/dt is exploding, not vanishing

**Problem**: After disabling adaptive weighting for the first 10k steps (raw MSE), `dudt_norm` exploded from 439 (step 1) to 13,354 (step 2k), and the loss diverged from 167 to 21,149.

**Finding**: The hypothesis that `du/dt` was vanishing (causing the model to degenerate to standard flow) was **wrong**. The actual dynamic is the opposite: without adaptive normalization, MeanFlow's self-referential target creates a positive feedback loop: large `du/dt` → large `u_tgt = v - h * du/dt` → huge MSE → huge gradients → even larger `du/dt`. The adaptive weighting `loss / (loss + eps)^p ≈ 1.0` is a necessary stabilizer that caps the effective gradient, preventing this runaway. It is not merely a convenience — it is critical for MeanFlow convergence.

**Fix**: Reverted: adaptive weighting is now active from step 0 (as in the MeanFlow paper). The `dudt_norm` logging remains to monitor the derivative magnitude going forward.

**Files changed**: `flower_vla/agents/meanflower.py`

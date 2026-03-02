# MeanFlower VLA - Changelog

## [2026-03-02] Switch from original MeanFlow to Improved Mean Flow (iMF)

**Problem**: Training loss decreases but validation MSE degrades after ~10k steps (0.137 at 10k → 0.291 at 30k). The root cause is the original MeanFlow's self-referential target `u_tgt = v - h * du/dt`: as network weights shift, the target drifts, creating bootstrap instability that adaptive weighting can mitigate but not eliminate.

**Solution**: Implement the Improved Mean Flow (iMF) formulation from Geng, Lu et al. (2025, arXiv:2512.02012). The key insight is that the original MF loss can be reformulated with a **network-independent target** `(e - x)` via a compound function `V = u + (t-r) * sg(du/dt)`.

**Algorithm changes:**
1. **Boundary condition**: extra forward pass `v_pred = u(z, t, t)` (h=0) to get the network's velocity estimate
2. **JVP tangent**: use `v_pred` (network's v estimate) instead of `e - x`
3. **Compound function**: `V = u_pred + h * sg(du/dt)` replaces self-referential `u_tgt`
4. **Loss**: `loss = loss_u + loss_v` where `loss_u = ||V - (e-x)||^2` (compound function) and `loss_v = ||v_pred - (e-x)||^2` (auxiliary, supervises boundary condition). Both use adaptive weighting.
5. **Cost**: ~50% more compute (one extra forward pass), NOT 2x

**Config changes**: None. Keeping current config (12L/768d/8h, wd=0.01, ema=0.999) to change one thing at a time.

**Files changed**: `flower_vla/agents/meanflower.py`

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

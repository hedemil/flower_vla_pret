# MeanFlower VLA - Changelog

## [2026-03-04] Move token prepend before shared blocks for full ∂u/∂t path

**Problem**: With `max_dudt_norm=150` and no clipping, `v_loss` still degrades (0.44→0.73 over steps 2k-5k). The clip was not the root cause — unbounded `dudt` growth is. Token conditioning (t/h) was only prepended at the u-head (after shared blocks), so `∂u/∂t` through the bounded token/attention path only flowed through 8 of 12 layers.

**Root cause**: The material derivative `du/dt` needs `∂u/∂t` to propagate through all layers that process the input. With tokens prepended only before u-head blocks, the shared blocks (4 layers) had no token-based `∂u/∂t` path — only the adaLN path (which is detached during JVP). Moving tokens before shared blocks gives the full 12-layer path via bounded attention, without adaLN amplification risk.

**Fix**: Restructure `dit_forward_meanflow` to prepend t/h tokens before shared blocks instead of before u-head blocks:
- Renamed `u_head_t_token_embedder` → `cond_t_token_embedder`, `u_head_h_token_embedder` → `cond_h_token_embedder`
- Token prepend, custom attention mask, and position_ids now computed before shared block loop
- Shared blocks run with tokens + custom mask (adaLN still active — tokens get modulated too, harmless)
- v-head strips tokens after shared blocks → runs standard causal adaLN, unaffected
- u-head continues with tokens through all 8 blocks, strips before decode
- `else` branch (no vector gates) unchanged

**Key properties**:
- `detach_time_cond=True` still zeroes tangent through adaLN chain — no (1+scale)^4 amplification
- `∂u/∂t` through token attention flows through all 12 layers — bounded by softmax
- v-head architecture completely unaffected (strips tokens, uses standard causal mask)
- Zero-init gates still ensure `dudt_norm ≈ 0` at step 1

**Files changed**: `flower_vla/agents/meanflower.py`

## [2026-03-03] Fix dudt_norm explosion: clip JVP tangent per-sample norm

**Problem**: Despite `detach_time_cond` fix (20x improvement), `dudt_norm` still grows unboundedly and destabilizes training around step 3-4k:
- Step 1: dudt_norm=0 (zero-init works)
- Step 2k: dudt_norm=177 (healthy)
- Step 4k: dudt_norm=665 (destabilized, v_loss turning up)

**Root cause**: The remaining dudt growth comes from `∂u/∂z · v_c` — the Jacobian of u w.r.t. input z, scaled by v-head prediction. This propagates through all DiT blocks where adaLN **gate values** (not tangents) scale the JVP tangent at each block. As gates grow from zero-init during training, the Jacobian grows multiplicatively through depth. Since dudt is detached in `V = u + h * sg(dudt)`, it receives no direct gradient pressure to stay small.

**Fix**: Clip per-sample L2 norm of dudt before use in V (analogous to gradient clipping). `max_dudt_norm=50.0` gives ~3x headroom above expected converged value (~17 for 128-dim actions). Clipping preserves direction. Raw (pre-clip) `dudt_norm` still logged for diagnosis, plus new `dudt_clip_frac` metric to monitor how often clipping activates.

**Expected behavior**: `dudt_clip_frac` starts high and decreases as model converges to correct dudt. If it stays at 1.0, increase `max_dudt_norm`.

**Files changed**: `flower_vla/agents/meanflower.py`, `conf/trainer/agent/meanflower_vla.yaml`

## [2026-03-03] Fix dudt explosion: detach time conditioning from JVP

**Problem**: Despite zero-init (dudt=0 at step 1), `dudt_norm` still explodes to 2199 by step 1k, and `v_loss` increases from 1.18 to 9.64 (v-head getting worse).

**Root cause**: Architectural mismatch with official iMF. Our DiT uses **adaLN modulation** conditioned on `t` through every block: `t → t_embedder → global_cond → adaLN_modulation → shift/scale/gate`. This creates a deep `∂u/∂t` chain through ALL blocks that amplifies `dudt` via the JVP. The sinusoidal embedding has frequencies up to 1000, making `∂t_emb/∂t` huge.

The official iMF (`imfDiT.py`) uses a fundamentally different architecture:
- **No adaLN** — uses simple zero-init vector gates (`attn_scale`, `mlp_scale`)
- **h enters as tokens** in the sequence, not as global conditioning
- Transformer blocks have NO time-dependent modulation
- `∂u/∂h` only propagates through attention (bounded by softmax) and the decoder (shallow)

**Fix**: Detach `t` before `t_embedder` during JVP (`detach_time_cond=True`). This zeroes the JVP tangent through the adaLN chain while keeping the primal computation correct. After this fix:
- `dudt = ∂u/∂z · v_c + ∂u/∂h_decoder` (shallow, bounded)
- The deep `∂u/∂t_emb` through all blocks is eliminated
- Blocks still receive correct time conditioning (only the tangent is zeroed, not the value)
- `t_embedder` weights still get gradients from `loss_v` (v_pred is computed without detach)

**Files changed**: `flower_vla/agents/meanflower.py`

## [2026-03-02] Fix v-head training instability: zero-init output layers

**Problem**: `dudt_norm` explodes from 450 to 6242 in 1k steps after adding v-head architecture. Compared against official iMF JAX implementation (`imfDiT.py`).

**Root cause**: Missing zero-initialization from official iMF. The official code zero-inits both `FinalLayer` (u and v decoders output exactly zero at init) and residual gates in `TransformerBlock` (each block is identity at init). Our code used default PyTorch init (Kaiming/Xavier), producing large random outputs and a large random Jacobian `∂u/∂z` at step 0.

**Fix**: Two zero-init changes matching the official iMF implementation:
1. **`MeanFlowDecoder.decoder`**: Zero-init final linear layer weight and bias → `u=0`, `v=0`, `dudt=0` at initialization
2. **`FlowBlock.adaLN_modulation`**: Zero-init output linear weight and bias → `gate_msa=0`, `gate_mlp=0`, `shift=0`, `scale=0` at init → each block is identity (since `modulate` uses `(1+scale)*x + shift`)

**Expected effect**: `dudt_norm` should start near 0 at step 1 and grow gradually, instead of starting at 450 and exploding.

**Files changed**: `flower_vla/agents/networks/meanflower_transformers.py`

## [2026-03-02] Fix OOM from v-head blocks inside JVP

**Problem**: CUDA OOM on 4x A100 64GB after adding the v-head architecture. The v-head added 8 extra DiT blocks that were unnecessarily propagated through `torch.func.jvp` as dual tensors, roughly doubling per-block memory. Additionally, `v_cond_fn` ran through all 20 blocks (including 8 unused u-head blocks) while retaining the full autograd graph despite the result being `.detach()`'d.

**Root cause**: Three sources of wasted memory in the JVP + v-head computation:
1. `u_func` used `return_v=True`, propagating dual tensors through v_head_blocks (8 blocks) even though v_pred was only auxiliary (`has_aux=True` doesn't skip computation, only discards tangents at return)
2. `v_cond_fn` ran through u_head_blocks (8 blocks) it didn't need, since it only used the v-head output
3. `v_cond_fn` retained the autograd computation graph (20 blocks of activations) despite the result being detached

**Fix**: Restructure the forward passes to minimize memory during JVP:
1. Added `v_only` mode to `dit_forward_meanflow` — skips u-head blocks entirely, only runs shared + v-head
2. JVP now only runs through shared + u-head (12 blocks with dual tensors, down from 20)
3. v_pred computed separately outside JVP via `v_only=True` (12 blocks, normal autograd)
4. `v_cond_fn` wrapped in `torch.no_grad()` and uses `v_only=True` (12 blocks, no activations retained)

**Memory reduction**: ~40% reduction in peak GPU memory during training step. Before: ~60 block-equivalents (where dual = 2x), after: ~36 block-equivalents.

**Files changed**: `flower_vla/agents/meanflower.py`

## [2026-03-02] Implement Improved Mean Flow (iMF) with v-head architecture

**Problem**: Training loss decreases but validation MSE degrades after ~10k steps (0.137 at 10k → 0.291 at 30k). The original MeanFlow's self-referential target `u_tgt = v - h * du/dt` causes bootstrap instability. Initial boundary-condition approach (using u-head at h=0 as v) failed — `dudt_norm` exploded from 438 to 5982 in 1k steps because the shared u-head parameters received conflicting gradients from `loss_u` and `loss_v`.

**Solution**: Full iMF architecture from Geng, Lu et al. (2025, arXiv:2512.02012) with a **separate v-head** (own transformer blocks + decoder), matching the official JAX implementation.

**Architecture changes:**
- DiT blocks split into: `shared_blocks` (4 layers) + `u_head_blocks` (8 layers) + `v_head_blocks` (8 layers, new)
- New `v_action_decoders` (MeanFlowDecoder per action space) for v-head output
- New `aux_head_depth` config parameter (default: 8, matching reference)
- Network returns `(u, v)` during training, just `u` during inference (v-head skipped)

**Algorithm (matching official iMF implementation):**
1. **v-head prediction**: `v_c = v_head(z, t, h=0)` — separate network head, own parameters
2. **JVP with `has_aux=True`**: differentiates only u-head, v-head output is auxiliary
3. **JVP tangent**: `v_c` (v-head prediction at h=0), not `e - x`
4. **Compound function**: `V = u_pred + h * sg(du/dt)`
5. **Loss**: `loss = loss_u + loss_v` where `loss_u = ||V - (e-x)||^2` and `loss_v = ||v_head - (e-x)||^2`
6. **Key**: v-head has separate blocks → clean gradients from `loss_v`, no interference from noisy `loss_u`

**Config changes:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| aux_head_depth | 8 | iMF default. 12 total = 4 shared + 8 per head |

**Files changed**: `flower_vla/agents/meanflower.py`, `conf/trainer/agent/meanflower_vla.yaml`

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

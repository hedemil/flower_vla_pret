# MeanFlower VLA - Changelog

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

# MeanFlower VLA - Run Comparison

## Run A: Pre-Ratio-Fix Baseline (2026-03-09)

### Parameters

| Parameter | Value |
|-----------|-------|
| `data_proportion` | 0.75 (75% velocity h=0, 25% integral h>0) — **inverted from reference** |
| `norm_p` | 1.0 |
| `norm_eps` | 0.01 |
| `ema_decay` | 0.999 |
| `P_mean` | -0.4 |
| `P_std` | 1.0 |
| `noise_dist` | logit_normal |
| `n_layers` | 12 | 
| `dit_dim` | 1024 |
| `n_heads` | 8 |
| `transformer_weight_decay` | 0.1 |
| `dropout (attn/resid/mlp)` | 0.0 / 0.0 / 0.0 |
| `detach_time_cond` | True |
| Datasets | fractal + bridge + libero (spatial/object/goal/10) |

### Training Metrics

| Step | raw_mse | v_loss | dudt_norm | cos_u_utgt | cos_u_v |
|------|---------|--------|-----------|------------|---------|
| 1 | 136.22 | 1.216 | 0.00 | 0.000 | 0.000 |
| 1k | 45.55 | 0.425 | 190.78 | 0.787 | 0.800 |
| 2k | 35.81 | 0.357 | 209.69 | 0.838 | 0.838 |
| 3k | 29.98 | 0.296 | 250.75 | 0.865 | 0.864 |
| 5k | 26.22 | 0.275 | 246.41 | 0.881 | 0.876 |
| 7k | 25.01 | 0.247 | 282.68 | 0.891 | 0.889 |
| 10k | 19.75 | 0.212 | 253.85 | 0.911 | 0.905 |
| 15k | 19.55 | 0.206 | 293.02 | 0.914 | 0.909 |
| 20k | 17.57 | 0.194 | 308.80 | 0.918 | 0.911 |
| 25k | 19.92 | 0.216 | 289.41 | 0.912 | 0.905 |
| 30k | 17.77 | 0.184 | 277.90 | 0.922 | 0.917 |
| 34k | 14.61 | 0.169 | 264.37 | 0.935 | 0.927 |

### Val Loss (per-sample MSE of single-step generation)

| Step | overall | fractal | bridge | libero_spatial | libero_object | libero_goal | libero_10 |
|------|---------|---------|--------|----------------|---------------|-------------|-----------|
| 10k | **0.175** | 0.165 | 0.211 | 0.167 | 0.154 | 0.158 | 0.149 |
| 20k | **0.271** | 0.259 | 0.311 | 0.264 | 0.250 | 0.252 | 0.244 |
| 30k | **0.313** | 0.301 | 0.351 | 0.312 | 0.298 | 0.298 | 0.289 |

### Key Observation

Training metrics keep improving (raw_mse 136→14.6, cos_u_utgt 0→0.94) but **val_loss increases monotonically**: 0.175 → 0.271 → 0.313 (+79% from 10k to 30k). All datasets degrade uniformly. Bridge consistently worst, libero_10 consistently best.

---

## Run B: Post-Ratio-Fix (2026-03-10)

### Parameter Changes from Run A

| Parameter | Run A | Run B | Rationale |
|-----------|-------|-------|-----------|
| `ratio` (was `data_proportion`) | 0.75 (75% velocity) | **0.75 (75% integral)** | Flipped semantics to match py-meanflow reference |
| `norm_p` | 1.0 | **0.75** | Match reference CIFAR10 config |
| `norm_eps` | 0.01 | **0.001** | Match reference CIFAR10 config |

All other parameters unchanged from Run A.

### Training Metrics

| Step | raw_mse | v_loss | dudt_norm | cos_u_utgt | cos_u_v |
|------|---------|--------|-----------|------------|---------|
| 1 | 129.59 | 1.157 | 0.00 | 0.000 | 0.000 |
| 1k | 43.48 | 0.478 | 174.08 | 0.761 | 0.785 |
| 2k | 31.86 | 0.398 | 198.24 | 0.820 | 0.828 |
| 3k | 27.31 | 0.342 | 228.55 | 0.846 | 0.851 |
| 5k | 24.01 | 0.325 | 223.74 | 0.863 | 0.863 |
| 7k | 17.87 | 0.261 | 246.16 | 0.894 | 0.892 |
| 10k | 19.35 | 0.251 | 268.04 | 0.895 | 0.894 |
| 12k | 15.00 | 0.225 | 238.14 | 0.911 | 0.908 |
| 15k | 17.28 | 0.244 | 261.00 | 0.901 | 0.903 |
| 20k | 15.12 | 0.231 | 226.97 | 0.913 | 0.911 |
| 22k | 12.98 | 0.203 | 265.43 | 0.927 | 0.920 |
| 25k | 16.96 | 0.231 | 259.66 | 0.904 | 0.909 |
| 30k | 13.69 | 0.203 | 254.08 | 0.925 | 0.916 |
| 34k | 12.57 | 0.208 | 257.72 | 0.925 | 0.918 |
| 36k | 11.66 | 0.184 | 276.94 | 0.936 | 0.929 |
| 40k | 10.66 | 0.198 | 247.31 | 0.936 | 0.926 |
| 42k | 11.32 | 0.211 | 233.59 | 0.931 | 0.923 |

### Val Loss (EMA model, per-sample MSE of single-step generation)

| Step | overall | fractal | bridge | libero_spatial | libero_object | libero_goal | libero_10 |
|------|---------|---------|--------|----------------|---------------|-------------|-----------|
| 10k | **0.173** | 0.163 | 0.211 | 0.162 | 0.151 | 0.151 | 0.145 |
| 20k | **0.245** | 0.234 | 0.288 | 0.231 | 0.222 | 0.218 | 0.217 |
| 30k | **0.265** | 0.254 | 0.307 | 0.254 | 0.244 | 0.240 | 0.242 |
| 40k | **0.272** | 0.259 | 0.314 | 0.263 | 0.251 | 0.247 | 0.244 |

### Analysis

Val_loss still increases but **rate of increase is slowing** (approaching plateau ~0.27):
- 10k→20k: +0.072
- 20k→30k: +0.020
- 30k→40k: +0.007

Ratio fix helped vs Run A (0.272 vs 0.313 at 30k, -15%) but did not solve the fundamental divergence. Training metrics keep improving (raw_mse 130→10.7, cos_u_utgt 0→0.94) while val_loss degrades. Best checkpoint remains at 10k.

**Next diagnostic**: Add online-model eval (no EMA) alongside EMA eval to determine if EMA lag is the cause.

---

## Reference: py-meanflow CIFAR10 Config

| Parameter | CIFAR10 v0 | CIFAR10 v1 | Our Run B |
|-----------|-----------|-----------|-----------|
| ratio (integral %) | 0.75 (75%) | 0.75 (75%) | 0.75 (75%) |
| P_mean_t | -2.0 | -0.6 | -0.4 |
| P_std_t | 2.0 | 1.6 | 1.0 |
| P_mean_r | -2.0 | -4.0 (r→0, larger h) | -0.4 (same as t) |
| P_std_r | 2.0 | 1.6 | 1.0 |
| norm_p | 0.75 | 0.75 | 0.75 |
| norm_eps | 0.001 | 0.001 | 0.001 |
| ema_decay | 0.9999 | 0.9999 | 0.999 |
| dropout | 0.2 | 0.2 | 0.0 |

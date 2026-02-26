# MeanFlow Training-Time CFG: Implementation Guide

Reference: [Gsunshine/meanflow](https://github.com/Gsunshine/meanflow/blob/main/meanflow.py)

---

## 1. Background: Standard CFG vs MeanFlow CFG

### Standard CFG (what we have now)

During **training**, randomly drop conditioning features so the model learns both conditional and unconditional behavior:
- `encode_observations` (line 423): zeros text features with probability `cfg_dropout`
- `dit_forward_meanflow` (line 893): zeros freq/proprio with probability `cfg_dropout`

During **inference**, run the network twice and interpolate:
```
u_cond   = model(z, t, h, cond)
u_uncond = model(z, t, h, null_cond)
u = u_uncond + cfg_lambda * (u_cond - u_uncond)
```
**Cost: 2 forward passes per inference step.**

### MeanFlow Training-Time CFG

During **training**, compute a *guided velocity target* and train the model to directly predict it. During **inference**, one forward pass already outputs the guided result.

**Cost: 1 forward pass at inference. 1 extra `v_fn` call during training (cheap relative to JVP).**

---

## 2. The Official MeanFlow Training-Time CFG

All references below are to `meanflow.py` in the official repo.

### 2.1 Config parameters (lines 73-83)

```python
guidance_eq:            str   = 'cfg'      # guidance equation type
omega:                  float = 1.0        # guidance strength (like cfg_lambda but at train time)
kappa:                  float = 0.5        # blend coefficient (0 = simpler variant)
class_dropout_prob:     float = 0.1        # conditioning dropout probability
t_start:                float = 0.0        # time range for guidance
t_end:                  float = 1.0        # time range for guidance
```

### 2.2 `v_fn` — instantaneous velocity (line 128)

Calls the network with `h=0` (zero time-offset), which means "predict instantaneous velocity at time t" rather than "predict average velocity over interval [r, t]":

```python
def v_fn(self, x, t, y, train=False):
    h = jnp.zeros_like(t)
    return self.u_fn(x, t, h, y=y, train=train)
```

Note: `train=False` — no dropout or training artifacts during the guidance computation.

### 2.3 `guidance_fn` — compute guided velocity (lines 132-151)

Takes the **ground-truth** velocity `v = e - x` (known from data, free) and amplifies the conditional signal:

```python
def guidance_fn(self, v_t, z_t, t, y, train=False):
    if self.guidance_eq == 'cfg' and self.kappa == 0:
        # Simple variant: only needs one extra v_fn call
        y_null = jnp.array([self.num_classes] * z_t.shape[0])
        v_uncond = self.v_fn(z_t, t, y=y_null, train=train)

        omega = jnp.where((t >= self.t_start) & (t <= self.t_end), self.omega, 1.0)
        v_g = v_uncond + omega * (v_t - v_uncond)

    elif self.guidance_eq == 'cfg' and self.kappa > 0:
        # Full variant: needs two extra v_fn calls
        y_null   = jnp.array([self.num_classes] * z_t.shape[0])
        v_uncond = self.v_fn(z_t, t, y=y_null, train=train)
        v_cond   = self.v_fn(z_t, t, y=y, train=train)

        omega = jnp.where((t >= self.t_start) & (t <= self.t_end), self.omega, 1.0)
        kappa = jnp.where((t >= self.t_start) & (t <= self.t_end), self.kappa, 0.0)
        v_g = omega * v_t + (1 - omega - kappa) * v_uncond + kappa * v_cond

    else:
        v_g = v_t

    return v_g
```

**Key insight:** `v_t` is the *ground-truth* velocity from data (not a network prediction). The network is only called for the unconditional estimate `v_uncond`. With `omega > 1.0`:
```
v_g = v_uncond + omega * (v_true - v_uncond)
```
This extrapolates *beyond* the true velocity in the direction away from unconditional — baking guidance into the training target.

With `omega = 1.0`: `v_g = v_uncond + 1*(v_true - v_uncond) = v_true` — no guidance effect (safe default).

### 2.4 `cond_drop` — conditioning dropout (lines 153-162)

For a fraction of samples, revert to unguided velocity and null conditioning:

```python
def cond_drop(self, v_t, v_g, labels):
    bz = v_t.shape[0]

    rand_mask = jax.random.uniform(self.make_rng('gen'), shape=(bz,)) < self.class_dropout_prob
    num_drop = jnp.sum(rand_mask).astype(jnp.int32)
    drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

    y_inp = jnp.where(drop_mask.reshape(bz,), self.num_classes, labels)
    v_g   = jnp.where(drop_mask, v_t, v_g)
    return y_inp, v_g
```

**Purpose:** The model needs to also learn what "unconditional" means so that future `v_fn` calls produce useful unconditional predictions. This is a bootstrapping loop:
- Dropped samples: `v_g = v_true` (unguided), conditioning = null → trains unconditional path
- Non-dropped samples: `v_g = guided velocity`, conditioning = real → trains guided predictions

### 2.5 The training forward pass (lines 168-211)

Putting it all together:

```python
def forward(self, imgs, labels, train=True):
    x  = imgs.astype(self.dtype)
    bz = imgs.shape[0]

    # --- Step 1: Sample timepoints ---
    t, r = self.sample_tr(bz)

    # --- Step 2: Create noisy sample and ground-truth velocity ---
    e   = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype)
    z_t = (1 - t) * x + t * e
    v   = e - x                          # <-- ground-truth velocity (FREE, from data)

    # --- Step 3: Compute guided velocity ---
    v_g = self.guidance_fn(v, z_t, t, labels, train=False)   # <-- 1 extra v_fn call

    # --- Step 4: Apply conditioning dropout ---
    y_inp, v_g = self.cond_drop(v, v_g, labels)              # <-- some samples revert to unguided

    # --- Step 5: JVP with v_g as tangent ---
    def u_fn(z_t, t, r):
        return self.u_fn(z_t, t, t - r, y=y_inp, train=train)  # <-- uses y_inp (post-dropout)

    dt_dt = jnp.ones_like(t)
    dr_dt = jnp.zeros_like(t)
    u, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, dt_dt, dr_dt))   # <-- tangent is v_g, not v!

    # --- Step 6: Compute target and loss ---
    u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt  # <-- target uses v_g
    u_tgt = jax.lax.stop_gradient(u_tgt)

    loss = (u - u_tgt) ** 2
    # ... adaptive weighting ...
```

---

## 3. Mapping to Our Codebase

### 3.1 What exists (in `flower_vla/agents/meanflower.py`)

| Official concept | Our equivalent | Status |
|---|---|---|
| `omega` | — | Does not exist. Add as `__init__` param (~line 90) |
| `v_fn(x, t, y)` | — | Does not exist. Implement as `dit_forward_meanflow(z, t, h=0, cond)` |
| `guidance_fn(v, z, t, y)` | `apply_guidance_fn` (line 763) | Stub (`pass`). Implement |
| `cond_drop(v, v_g, labels)` | — | Does not exist. Implement |
| `class_dropout_prob` | `cfg_dropout` (line 54) | Exists (0.1). Reuse |
| Random feature dropout | `encode_observations:423` and `dit_forward_meanflow:893` | Exists. **Remove** (replaced by `cond_drop`) |
| JVP tangent = `v_g` | `meanflow_loss:574` uses `v` | Change to `v_g` |
| Target `u_tgt = v_g - h*du/dt` | `meanflow_loss:582` uses `v` | Change to `v_g` |
| `apply_guidance` bool | Line 90, 159 | Exists but is a bool not a method. Line 529 would crash if True |

### 3.2 What to change

#### A. Add `omega` parameter

In `__init__` signature (~line 90), add:
```python
omega: float = 1.0,
```

Store it (~line 159):
```python
self.omega = omega
```

`omega = 1.0` is a no-op (backward compatible). `omega > 1.0` activates training-time CFG.

#### B. Implement `v_fn`

New method (near `dit_forward_meanflow`, ~line 868):
```python
def v_fn(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
    """Instantaneous velocity prediction: network with h=0."""
    h = torch.zeros_like(t)
    return self.dit_forward_meanflow(z, t, h, cond_dict)
```

#### C. Implement `_make_null_cond`

Creates a null conditioning dict. In the official code, null conditioning is just `y_null = num_classes` (a single null label). For our VLA, null conditioning means zeroing text features, frequency embeddings, and proprioception:

```python
def _make_null_cond(self, cond: dict) -> dict:
    """Create null conditioning by zeroing text, freq, and proprio."""
    null_cond = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cond.items()}

    # Zero text portion of features
    prompt_length = self.prompt_embeds.shape[1]
    image_length = 50 if self.use_second_view is False else 100
    text_start = prompt_length + image_length
    null_cond['features'] = null_cond['features'].clone()
    null_cond['features'][:, text_start:, :] = 0.0

    # Zero frequency and proprio embeddings
    null_cond['frequency_embeds'] = torch.zeros_like(null_cond['frequency_embeds'])
    if null_cond.get('proprio') is not None:
        null_cond['proprio'] = torch.zeros_like(null_cond['proprio'])

    return null_cond
```

#### D. Implement `guidance_fn`

Replace the `apply_guidance_fn` stub (line 763). Reference: official `guidance_fn` (lines 132-151).

```python
def guidance_fn(self, v: torch.Tensor, z: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
    """
    Compute guided velocity target.
    Reference: meanflow.py:132-151

    v_g = v_uncond + omega * (v - v_uncond)

    With omega=1.0: v_g = v (no guidance, safe default)
    With omega>1.0: amplifies conditional signal
    """
    if self.omega == 1.0:
        return v  # No guidance effect

    null_cond = self._make_null_cond(cond)
    t_flat = t.view(-1)

    # Official code uses train=False for guidance computation
    with torch.no_grad():
        was_training = self.training
        self.eval()
        v_uncond = self.v_fn(z, t_flat, null_cond)
        if was_training:
            self.train()

    return v_uncond + self.omega * (v - v_uncond)
```

#### E. Implement `cond_drop`

Reference: official `cond_drop` (lines 153-162). The official code modifies labels; we modify the conditioning dict.

```python
def cond_drop(self, v: torch.Tensor, v_g: torch.Tensor, cond: dict) -> Tuple[torch.Tensor, dict]:
    """
    Conditioning dropout: for cfg_dropout fraction of samples,
    revert v_g to unguided v and null out their conditioning.
    Reference: meanflow.py:153-162
    """
    if self.cfg_dropout <= 0:
        return v_g, cond

    b = v.shape[0]
    drop_mask = torch.rand(b, device=v.device) < self.cfg_dropout

    if not drop_mask.any():
        return v_g, cond

    # Revert v_g -> v for dropped samples
    expand_shape = [b] + [1] * (v.dim() - 1)
    drop_expanded = drop_mask.view(expand_shape).to(v.dtype)
    v_g = v * drop_expanded + v_g * (1 - drop_expanded)

    # Null out conditioning for dropped samples
    cond = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cond.items()}

    prompt_length = self.prompt_embeds.shape[1]
    image_length = 50 if self.use_second_view is False else 100
    text_start = prompt_length + image_length

    cond['features'] = cond['features'].clone()
    cond['features'][drop_mask, text_start:, :] = 0.0

    cond['frequency_embeds'] = cond['frequency_embeds'].clone()
    cond['frequency_embeds'][drop_mask] = 0.0

    if cond.get('proprio') is not None:
        cond['proprio'] = cond['proprio'].clone()
        cond['proprio'][drop_mask] = 0.0

    return v_g, cond
```

#### F. Modify `meanflow_loss` (line 494)

Four changes in the loss function:

**1.** Replace line 529 (`v_g = self.apply_guidance(...)` placeholder):
```python
# BEFORE (line 529):
v_g = self.apply_guidance(v, z, texp, cond) if self.apply_guidance else v

# AFTER:
v_g = self.guidance_fn(v, z, texp, cond)       # guided velocity (official: line 188)
v_g, cond_dropped = self.cond_drop(v, v_g, cond)  # conditioning dropout (official: line 191)
```

**2.** Change `u_func` to use `cond_dropped` (line 540-544):
```python
# BEFORE (line 544):
return self.dit_forward_meanflow(z_input, t_flat, h_flat, cond)

# AFTER:
return self.dit_forward_meanflow(z_input, t_flat, h_flat, cond_dropped)
```

**3.** JVP tangent: use `v_g` instead of `v` (line 574):
```python
# BEFORE (line 574):
(v, dtdt, drdt)

# AFTER:
(v_g, dtdt, drdt)
```

Note: `v_g` must also be cast to float32 at the same point as `v` (line 535):
```python
v_g = v_g.float()
```

**4.** Target: use `v_g` instead of `v` (line 582):
```python
# BEFORE (line 582):
u_tgt = (v - h * dudt).detach()

# AFTER:
u_tgt = (v_g - h * dudt).detach()
```

#### G. Remove existing random dropout

These are replaced by the explicit `cond_drop` mechanism:

**1.** `encode_observations` lines 422-432: remove the `if self.cfg_dropout > 0 and self.training:` block that zeros text features.

**2.** `dit_forward_meanflow` lines 892-896: remove the `if self.training and self.cfg_dropout > 0:` block that zeros freq/proprio.

#### H. Inference: no changes strictly needed

With training-time CFG, the model's single-pass output at inference is already guided. The existing `cfg_lambda` / two-pass CFG in `_sample_with_fixed_steps` (lines 728-753) is skipped by default (`cfg_lambda = 1.0`), so it's harmless. You can keep it as a fallback for experimentation.

---

## 4. Summary of Data Flow

### Training (with `omega > 1.0`)

```
Ground-truth velocity:   v = e - actions          (free, from data)
                              |
                              v
Guided velocity:         v_g = guidance_fn(v, z, t, cond)
                              |                |
                              |           v_fn(z, t, null_cond)  [1 extra forward pass, no_grad]
                              |                |
                              |           v_uncond + omega * (v - v_uncond)
                              v
Cond dropout:            v_g, cond_dropped = cond_drop(v, v_g, cond)
                              |                |
                              |          ~10% samples: v_g -> v, cond -> null
                              v
JVP:                     u, du/dt = jvp(u_fn, (z, t, r), (v_g, 1, 0))
                              |
                              v
Target:                  u_tgt = v_g - h * du/dt
Loss:                    MSE(u, u_tgt)
```

### Inference (single step)

```
z_1 ~ N(0, 1)
z_0 = z_1 - u(z_1, t=1, h=1)      <-- single forward pass, already guided
```

---

## 5. Config

```yaml
# Training-time CFG
omega: 1.5            # guidance strength (1.0 = disabled, >1.0 = active)
cfg_dropout: 0.1      # fraction of samples with null conditioning (existing param)

# Inference-time CFG (optional fallback, default is disabled)
# cfg_lambda: 1.0     # only needed if you want additional inference-time guidance
```

---

## 6. Verification Checklist

- [ ] `omega=1.0` (default): `guidance_fn` returns `v` unchanged, `cond_drop` still applies dropout. Should behave identically to current code.
- [ ] `omega=1.5`: guided targets used in training. Monitor that `loss` converges.
- [ ] `v_loss` metric (existing): should be higher than `loss` when `omega > 1.0`, confirming the model learns guided (not raw) velocity predictions.
- [ ] Inference: single-step sampling without `cfg_lambda > 1.0` should produce language-conditioned actions.
- [ ] Compare quality of `omega=1.5` single-step inference vs standard two-pass CFG with `cfg_lambda=1.5`.

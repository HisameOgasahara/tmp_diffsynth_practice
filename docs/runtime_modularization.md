# Runtime modularization plan

## Comparison baseline

This document compares the current minimal Anima practice repository with DiffSynth-Studio using the following commits.

- `HisameOgasahara/tmp_diffsynth_practive`: `a3b2350fe7a687ba54a74a1c322d1e2f3de0a12d`
- `modelscope/DiffSynth-Studio`: `7539a33b16844e2ce7e06306a2576346aa00de2b`

The comparison is limited to the Anima inference core. Generic framework features such as model registries, VRAM management, hot-loading, and broad multi-model infrastructure are not the focus.

## Current structure

The current repository keeps the model definitions separated, while most inference orchestration is concentrated in `src/anima_core/runtime.py`.

```text
src/anima_core/
├── anima_dit.py
├── text_encoder.py
├── vae.py
├── ops.py
└── runtime.py
```

The files currently have the following roles.

- `anima_dit.py`: Anima DiT, Transformer blocks, attention, positional embeddings, AdaLN-related logic, `LLMAdapter`, and text-conditioning preprocessing.
- `text_encoder.py`: Qwen3-based text encoder wrapper.
- `vae.py`: latent/image encode-decode model.
- `ops.py`: low-level shared operations such as attention and gradient-checkpoint wrappers.
- `runtime.py`: inference orchestration, including weight loading, tokenizer loading, prompt encoding, conditioning adaptation, noise initialization, Z-Image sigma/timestep schedule, CFG, Euler sampling, and VAE decoding.

The current runtime is therefore not only an Euler sampler. It is the execution layer that connects the complete minimal T2I path.

## Current runtime flow

```text
weights/tokenizers
      ↓
prompt encoding
      ↓
Anima conditioning
      ↓
initial noise
      ↓
Z-Image sigma/timestep schedule
      ↓
Anima DiT velocity prediction
      ↓
CFG
      ↓
Euler latent update
      ↓
VAE decode
      ↓
image
```

At present, only one sampling path is implemented: the Z-Image FlowMatch schedule with a deterministic Euler update.

## DiffSynth-Studio structure

At the comparison commit, DiffSynth separates the same responsibilities across model files, pipeline units, a scheduler, and a base pipeline.

Relevant files include:

```text
diffsynth/
├── models/
│   ├── anima_dit.py
│   ├── z_image_text_encoder.py
│   └── wan_video_vae.py
│
├── pipelines/
│   └── anima_image.py
│       ├── AnimaUnit_ShapeChecker
│       ├── AnimaUnit_NoiseInitializer
│       ├── AnimaUnit_InputImageEmbedder
│       ├── AnimaUnit_PromptEmbedder
│       └── model_fn_anima
│
└── diffusion/
    ├── flow_match.py
    │   └── FlowMatchScheduler
    └── base_pipeline.py
        ├── CFG handling
        └── scheduler.step orchestration
```

The model definitions are already split similarly to this repository. The main structural difference is that DiffSynth also separates the runtime responsibilities.

## Current runtime vs DiffSynth

| Current repository | DiffSynth-Studio counterpart |
|---|---|
| `download_weights()` | `ModelConfig` / model loading infrastructure |
| `load_text_encoder()` | model loader |
| `load_dit()` | model loader |
| `load_vae()` | model loader |
| `load_tokenizers()` | `AnimaImagePipeline.from_pretrained()` |
| `encode_prompt()` | `AnimaUnit_PromptEmbedder` |
| `adapt_conditioning()` | `AnimaDiT.preprocess_text_embeds()` |
| initial latent noise | `AnimaUnit_NoiseInitializer` |
| `z_image_schedule()` | `FlowMatchScheduler.set_timesteps_z_image()` |
| CFG inside `sample_euler()` | `BasePipeline.cfg_guided_model_fn()` |
| Euler latent update | `FlowMatchScheduler.step()` |
| `decode_image()` | pipeline-side VAE decode and image conversion |

## How DiffSynth handles the sampler

For Anima, DiffSynth constructs:

```python
self.scheduler = FlowMatchScheduler("Z-Image")
```

The scheduler owns both the sigma/timestep schedule and the default update rule.

The default FlowMatch update is Euler:

```python
prev_sample = sample + model_output * (sigma_next - sigma)
```

Therefore, the current repository's `z_image_schedule()` plus `sample_euler()` is essentially a flattened Anima-specific version of the DiffSynth Z-Image FlowMatch path.

DiffSynth also defines specialized scheduler subclasses for sampling rules that differ from the default deterministic Euler step, such as ancestral or stochastic variants.

## Planned modularization

If this repository later supports multiple samplers, `runtime.py` should be reduced to orchestration instead of accumulating sampler-specific branches.

A suitable future layout is:

```text
src/anima_core/
├── models/
│   ├── anima_dit.py
│   ├── text_encoder.py
│   └── vae.py
│
├── schedules/
│   └── z_image.py
│
├── samplers/
│   ├── euler.py
│   ├── heun.py
│   ├── midpoint.py
│   ├── ancestral.py
│   └── er_sde.py
│
├── conditioning.py
├── model_loader.py
├── ops.py
└── runtime.py
```

The exact filenames can change later; the important boundary is the responsibility split.

### 1. Model definitions

Keep model mathematics and neural-network structure separate from inference orchestration.

```text
anima_dit.py
text_encoder.py
vae.py
```

These should answer only questions such as:

- what does the network compute?
- how is conditioning injected?
- how are latent and pixel spaces transformed?

### 2. Conditioning

Move prompt tokenization, Qwen encoding, T5 token-ID preparation, and Anima conditioning adaptation out of the general runtime.

Conceptually:

```text
prompt
  ↓
tokenizers
  ↓
Qwen hidden states + T5 token IDs
  ↓
Anima conditioning
```

### 3. Schedule

The schedule should decide only which noise/time points are visited.

For example:

```text
sigma_0 > sigma_1 > ... > sigma_N
```

The current `z_image_schedule()` belongs here.

### 4. Sampler / solver

The sampler should decide how to move between schedule points.

For the same schedule, different numerical methods may be compared:

```text
Euler
Heun
Midpoint
RK-family methods
Ancestral variants
ER-SDE
```

This separation becomes important for methods such as Heun, because a second-order method may require additional DiT evaluations within one sampling step.

### 5. Model function

The DiT evaluation should be separated from the numerical solver.

Conceptually:

```text
model_fn(x, sigma, conditioning)
    ↓
velocity

sampler.step(model_fn, x, sigma_i, sigma_next)
    ↓
next latent
```

This avoids embedding `dit(...)` calls directly inside an Euler-specific loop and makes higher-order or stochastic samplers easier to add.

### 6. Runtime

The final `runtime.py` should mainly coordinate the pieces.

```text
load models
    ↓
build conditioning
    ↓
create initial latent
    ↓
choose schedule
    ↓
choose sampler
    ↓
run model_fn + sampler loop
    ↓
VAE decode
```

The runtime should know the order of execution, but not contain the mathematics of every sampler.

## DiffSynth-style split vs planned split

DiffSynth effectively groups the schedule and default sampling step inside `FlowMatchScheduler`.

```text
DiffSynth
FlowMatchScheduler
├── set_timesteps()
└── step()
```

For this repository, a slightly stricter separation is preferable if the purpose is sampler experimentation.

```text
planned
schedule
├── sigma/timestep construction

sampler
├── Euler
├── Heun
├── ER-SDE
└── other solvers

model_fn
└── Anima DiT evaluation
```

This keeps the current minimal-learning purpose while making sampler comparisons easier than a direct copy of the full DiffSynth abstraction.

## Summary

The current repository already follows DiffSynth closely at the model-definition level. The main simplification is that DiffSynth's pipeline units, scheduler, CFG handling, and model-loading responsibilities have been flattened into `runtime.py`.

The next modularization step should therefore focus on splitting the runtime rather than restructuring the model code:

```text
current
runtime.py
├── loading
├── conditioning
├── schedule
├── CFG
├── Euler sampling
└── decode

future
runtime.py       → orchestration only
conditioning.py  → prompt/conditioning path
schedules/       → sigma/timestep definitions
samplers/        → numerical sampling methods
model_loader.py  → weight/model loading
models/          → model definitions
```

This preserves the minimal Anima core while allowing multiple samplers to be added without coupling them to `anima_dit.py` or growing one large runtime function.

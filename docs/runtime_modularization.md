# 런타임 구조와 분리 계획

## 비교 기준

기존 비교는 실습 저장소 `a3b2350fe7a687ba54a74a1c322d1e2f3de0a12d`와 DiffSynth-Studio `7539a33b16844e2ce7e06306a2576346aa00de2b`를 기준으로 합니다. Anima 추론 코어와 sampler의 책임 분리를 다루며, 범용 registry나 다중 모델 프레임워크 설계는 범위에 포함하지 않습니다.

## 현재 책임

| 소스 | 책임 |
|---|---|
| `anima_dit.py` | DiT, attention, positional embedding, AdaLN, LLMAdapter |
| `text_encoder.py` | Qwen 기반 text encoder |
| `vae.py` | latent와 이미지 변환 |
| `ops.py` | attention 등 공통 연산 |
| `runtime.py` | 모델 로딩, conditioning, schedule, CFG, Euler, decode 연결 |
| `lora.py` | [LoRA 가중치 합산](lora.md) |
| `prompt_weights.py` | [토큰 가중치 해석](prompt_weights.md) |
| `model_cache.py` | [모델 보관과 장치 이동](model_cache.md) |
| `logging_utils.py`, `profiling.py` | [로그와 병목 진단](diagnostics.md) |

모델 정의는 분리되어 있으나, 기본 T2I 실행 흐름은 아직 `runtime.py`에 모여 있습니다.

```text
모델·tokenizer 준비 → prompt encoding → Anima conditioning
→ 초기 noise → schedule → DiT 예측·CFG → Euler 갱신 → VAE decode
```

## DiffSynth와의 대응

| 실습 함수 | DiffSynth의 담당 영역 |
|---|---|
| `download_weights`, `load_*` | ModelConfig와 모델 로딩 기반 구조 |
| `load_tokenizers`, `encode_prompt`, `adapt_conditioning` | Anima pipeline과 prompt embedder |
| noise 준비 | AnimaUnit_NoiseInitializer |
| `z_image_schedule` | FlowMatchScheduler의 Z-Image schedule |
| `sample_euler` 내부 CFG | BasePipeline의 CFG model function |
| Euler latent 갱신 | FlowMatchScheduler.step |
| `decode_image` | VAE와 pipeline의 이미지 변환 |

현재 sampler는 DiffSynth native의 Z-Image FlowMatch Euler입니다. schedule은 sigma/timestep을 만들고, 갱신은 `x_next = x + velocity * (sigma_next - sigma)`를 계산합니다. ComfyUI baseline의 ER-SDE는 구현하지 않았습니다.

DiffSynth는 schedule과 기본 갱신을 `FlowMatchScheduler`에 함께 둡니다. 여러 solver를 실험하려면 두 책임을 분리하는 편이 적합합니다.

## 향후 분리

| 영역 | 담당할 내용 |
|---|---|
| `models/` | 기존 모델 정의 |
| `model_loader.py` | 가중치 다운로드와 모델 로딩 |
| `conditioning.py` | tokenization, Qwen encoding, Anima adapter |
| `schedules/` | 방문할 sigma/timestep 결정 |
| `model_fn` | DiT 평가와 CFG로 velocity 반환 |
| `samplers/` | Euler, Heun, midpoint, ancestral, ER-SDE 등 갱신 규칙 |
| `runtime.py` | 위 영역의 실행 순서 조율 |

solver는 `model_fn(x, sigma, conditioning)`을 호출해 예측을 얻고 다음 latent를 계산하도록 분리합니다. Heun처럼 한 스텝에서 DiT 평가가 여러 번 필요한 방법이나 확률적 sampler를 추가할 때, 모델 정의와 solver를 서로 수정하지 않도록 하는 것이 목적입니다. 이 표는 계획이며 추가 sampler와 디렉터리 재배치는 아직 구현하지 않았습니다.

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
| `runtime.py` | 모델 로딩, conditioning, 생성 sampler 선택, decode 연결 |
| `sampling/model_prediction.py` | DiT 호출, CFG, velocity·denoised 변환 |
| `sampling/schedules.py` | Z-Image schedule과 RF 시작 sigma 보정 |
| `sampling/noise.py` | 추가 정규 노이즈와 Brownian noise |
| `sampling/euler.py`, `heun.py`, `dpmpp.py`, `er_sde.py` | sampler별 갱신식과 예측 이력 |
| `lora.py` | [LoRA 가중치 합산](lora.md) |
| `prompt_weights.py` | [토큰 가중치 해석](prompt_weights.md) |
| `model_cache.py` | [모델 보관과 장치 이동](model_cache.md) |
| `logging_utils.py`, `profiling.py` | [로그와 병목 진단](diagnostics.md) |

모델 정의와 생성 샘플링은 분리되어 있습니다. `runtime.py`는 T2I 실행 순서를 연결하며, 모델 로딩과 conditioning 함수는 이 파일에 남아 있습니다.

```text
모델·tokenizer 준비 → prompt encoding → Anima conditioning
→ 초기 noise → schedule → 선택 sampler·DiT 예측·CFG → VAE decode
```

## DiffSynth와의 대응

| 실습 함수 | DiffSynth의 담당 영역 |
|---|---|
| `download_weights`, `load_*` | ModelConfig와 모델 로딩 기반 구조 |
| `load_tokenizers`, `encode_prompt`, `adapt_conditioning` | Anima pipeline과 prompt embedder |
| noise 준비 | AnimaUnit_NoiseInitializer |
| `z_image_schedule` | FlowMatchScheduler의 Z-Image schedule |
| `ModelPrediction.velocity` | BasePipeline의 CFG model function |
| `sampling/euler.py` latent 갱신 | FlowMatchScheduler.step |
| `decode_image` | VAE와 pipeline의 이미지 변환 |

공통 schedule은 DiffSynth의 Z-Image 방식입니다. 기본 Euler는 `x_next = x + velocity * (sigma_next - sigma)`를 계산하며, 기존 연산 순서를 유지합니다. Heun, RF Euler ancestral, DPM++ 2M, RF DPM++ 2M SDE, RF ER-SDE도 선택할 수 있습니다. 세부 설정과 구현 출처는 [생성 샘플러](sampling.md)를 참고하세요.

DiffSynth는 schedule과 기본 갱신을 `FlowMatchScheduler`에 함께 둡니다. 이 저장소는 여러 solver를 비교할 수 있도록 두 책임을 분리했습니다.

## 향후 분리

| 영역 | 담당할 내용 |
|---|---|
| `models/` | 기존 모델 정의 |
| `model_loader.py` | 가중치 다운로드와 모델 로딩 |
| `conditioning.py` | tokenization, Qwen encoding, Anima adapter |
| `runtime.py` | 분리한 모델 로딩·conditioning의 실행 순서 조율 |

생성 solver는 `ModelPrediction`을 통해 velocity 또는 denoised를 얻고 다음 latent를 계산합니다. Heun의 추가 평가와 확률적 sampler의 노이즈·이력은 각 sampler 안에서 처리합니다. 위 표의 모델·로더·conditioning 추가 분리는 향후 계획이며 이번 샘플러 도입에 포함하지 않습니다.

## 학습 기능 계획

| 기능 | 참고 방식 |
|---|---|
| 캡션 셔플 | [anima-lora](https://github.com/sorryhyun/anima_lora)의 캡션 변형 캐시·학습 중 선택 방식 |

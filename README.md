# DiT · Flow Matching 학습 실습

**T2I 생성**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_minimal_colab.ipynb)

**LoRA 훈련**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_lora_train_colab.ipynb?forceEdit=true&sandboxMode=true)

**Muon LoRA 훈련 · GPU Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/feat/muon-optimizer/anima_lora_muon_train_colab.ipynb?forceEdit=true&sandboxMode=true)

Anima를 대상으로 DiT(Diffusion Transformer)·플로우 매칭(FM)을 학습하고, 생성·학습 코드를 이 저장소 안에서 완결하는 프로젝트입니다.

| 항목 | 방향 |
|---|---|
| 교재 | `reference`의 DiffSynth-Studio, [anima-lora](https://github.com/sorryhyun/anima_lora) |
| 구현 방식 | 기존 코어 로직 활용, 프레임워크 주변 구조 제거 |
| 코드 구성 | 필요한 구현을 이 리포에 포함하여 DiffSynth·Diffusers·ComfyUI 설치 없이 생성·학습 실행 |
| 실행 의존성 | PyTorch 등 일반 라이브러리, 모델 가중치 |
| 실행 환경 | 모델·생성·학습 모듈을 Colab 전용 코드와 분리 |

| 영역 | Anima 핵심 구조·현재 실습 방식 |
|---|---|
| 이미지 표현 | Qwen-Image VAE의 16채널 latent 사용, 이미지 크기를 가로·세로 각각 1/8로 압축 |
| DiT | latent를 2×2 패치로 나눠 28개 블록에서 처리. RoPE 위치 표현, 텍스트 cross-attention, 시간 조건 AdaLN 사용 |
| 텍스트 조건 | Qwen3-0.6B 출력과 T5 tokenizer의 토큰 ID를 6층 LLMAdapter로 결합해 DiT에 전달 |
| 생성 | 노이즈 latent에서 RF 속도를 예측해 CFG·선택 sampler로 갱신한 뒤 VAE로 디코딩. 공통 Z-Image schedule 사용 |
| LoRA 학습 | Muon / AdamW 선택. 데이터·노이즈의 직선 보간과 속도 예측 MSE로 학습. DiT에 LoRA를 적용하고 TE·text adapter는 고정하며, VAE latent·adapter 출력은 사전 캐시 |

## 현재 구현 상태

| 상태 | 기능 | 계획 |
|---|---|---|
| 구현됨 | Anima 모델 정의 분리, 텍스트 조건 처리, CFG, 생성 sampler 11종, VAE 디코딩 | [샘플러](docs/sampling.md) · [런타임 구조](docs/runtime_modularization.md) |
| 구현됨 | 기존 LoRA 가중치 적용, 토큰 가중치, 모델 캐시, 로그·프로파일러 | — |
| 구현됨 | LoRA 학습: Muon / AdamW 선택, TE·text adapter 학습 제외, text adapter 출력 사전 캐시 | [학습 기능](docs/runtime_modularization.md#학습-기능-계획) |

## 학습 확인 환경

| 항목 | 값 (2026-10-01 Colab) |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA / cuDNN | 12.8 / 9.19.0 |
| NVIDIA 드라이버 | 580.82.07 |
| GPU | Tesla T4, VRAM 15.0 GB |
| 학습 설정 | Muon, 해상도 512, batch 4, repeat 6, bf16 · [실행 TOML](assets/examples/변화과정/v2_muon/training.toml) |
| 시스템 RAM (약 296스텝) | 3.2 / 12.7 GB |
| GPU RAM (약 296스텝) | 7.3 / 15.0 GB |

전체 Python 패키지 버전: [requirements-freeze.txt](docs/requirements-freeze.txt)

## 생성 예시

![Anima Muon LoRA 500스텝 생성 예시](assets/examples/변화과정/v2_muon/다운로드_500_fin_wfs08.png)

| 항목 | 값 |
|---|---|
| 학습 캐릭터 | [시라카와 유이나](https://heaven-burns-red.com/character/30g/shirakawa-yuina/) |
| LoRA | Muon 500스텝, 적용 강도 1.0 |
| 생성 설정 | 1216×832, 30스텝, CFG 4.0, shift 3.0, denoise 1.0, fp16 |
| 시드 | 무작위 (`random_seed = true`) |
| 프롬프트·생성 설정 | [다운로드_500_fin_wfs08.json](assets/examples/변화과정/v2_muon/다운로드_500_fin_wfs08.json) |
| 학습 설정 원본 | [training.toml](assets/examples/변화과정/v2_muon/training.toml) |
| 생성 변화과정 | [v2_muon](assets/examples/변화과정/v2_muon) |

## 사용 안내

- [노트북 사용](docs/notebook_usage.md): 최초 실행, 설정 변경, 반복 생성
- [생성 샘플러](docs/sampling.md): Euler·Heun·ancestral, DPM++ 2M·2M SDE, ER-SDE, exp-Heun x0·SDE, SA Solver, RES multistep, gradient estimation
- [LoRA](docs/lora.md): 파일 준비, 지원 형식, 가중치 합산
- [토큰 가중치](docs/prompt_weights.md): 괄호 문법과 conditioning 처리
- [모델 캐시](docs/model_cache.md): CPU 보관, GPU 이동, 교체와 초기화
- [로그와 프로파일러](docs/diagnostics.md): 진행 출력과 병목 진단
- [런타임 구조와 분리 계획](docs/runtime_modularization.md): DiffSynth 비교와 향후 모듈 경계

## 크레딧

| 프로젝트 | 활용 |
|---|---|
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | 교재 및 Anima 모델 구현 출처 · Apache-2.0 |
| [anima-lora](https://github.com/sorryhyun/anima_lora) | LoRA 학습 구현 참고 |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Anima 이미지 생성 참고, 샘플러 갱신식·Brownian noise 구현 출처 · [GPL-3.0](licenses/ComfyUI-LICENSE) |

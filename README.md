# DiT · Flow Matching 학습 프로젝트

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_minimal_colab.ipynb)

Anima를 대상으로 DiT(Diffusion Transformer)·플로우 매칭(FM)을 학습하고, 생성·학습 코드를 이 저장소 안에서 완결하는 프로젝트입니다.

| 항목 | 방향 |
|---|---|
| 교재 | `reference`의 DiffSynth-Studio |
| 구현 방식 | 기존 코어 로직 활용, 프레임워크 주변 구조 제거 |
| 독립성 | DiffSynth·Diffusers·ComfyUI 설치 및 외부 리포 코드 의존성 제거 |
| 실행 의존성 | PyTorch 등 일반 라이브러리, 모델 가중치 |
| 모듈 구성 | 실행 환경과 독립적인 모델·생성·학습 로직 |

## 학습 및 구현 목표

| 영역 | 학습·구현 내용 |
|---|---|
| DiT | attention, 위치 임베딩, 시간·텍스트 조건 처리 |
| FM | 데이터·노이즈 보간, 시간 샘플링, velocity 예측, 학습 손실 |
| 생성 | 모델 예측으로 노이즈에서 이미지 생성 |
| 학습 | 데이터 입력, 학습, 체크포인트 저장·불러오기 |

## 현재 구현 상태

| 상태 | 기능 |
|---|---|
| 구현됨 | Anima 모델 정의 분리, 텍스트 조건 처리, CFG, FM Euler 샘플링, VAE 디코딩 |
| 구현됨 | 기존 LoRA 가중치 적용, 토큰 가중치, 모델 캐시, 로그·프로파일러 |
| 예정 | 모델 학습 |

## 생성 예시

![Anima LoRA 생성 예시](assets/examples/anima_lora.png)

## 사용 안내

- [노트북 사용](docs/notebook_usage.md): 최초 실행, 설정 변경, 반복 생성
- [LoRA](docs/lora.md): 파일 준비, 지원 형식, 가중치 합산
- [토큰 가중치](docs/prompt_weights.md): 괄호 문법과 conditioning 처리
- [모델 캐시](docs/model_cache.md): CPU 보관, GPU 이동, 교체와 초기화
- [로그와 프로파일러](docs/diagnostics.md): 진행 출력과 병목 진단
- [런타임 구조와 분리 계획](docs/runtime_modularization.md): DiffSynth 비교와 향후 모듈 경계

## 크레딧

| 프로젝트 | 활용 |
|---|---|
| DiffSynth-Studio | 교재 및 Anima 모델 구현 출처 · Apache-2.0 |
| ComfyUI | Anima 이미지 생성 참고 |

`src/anima_core/`의 `anima_dit.py`, `text_encoder.py`, `vae.py`는 `HisameOgasahara/DiffFlowDiT_test`에 포함된 DiffSynth-Studio 구현을 축약해 가져왔습니다.

향후 참고: Diffusers, [anima_lora](https://github.com/sorryhyun/anima_lora). 필요한 코드는 출처와 라이선스를 유지해 가져올 수 있습니다.

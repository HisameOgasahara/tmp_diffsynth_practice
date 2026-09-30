# DiT · Flow Matching 학습 프로젝트

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_minimal_colab.ipynb)

DiffSynth-Studio를 교재로 삼아 DiT(Diffusion Transformer)와 플로우 매칭(FM)을 학습하는 프로젝트입니다. 첫 대상은 Anima이며, 기존 구현에서 필요한 코어 로직을 가져오고 프레임워크에 결합된 주변 구조를 덜어내어, 이미지 생성과 모델 학습에 필요한 구현을 이 저장소 안에서 완결하는 것이 목표입니다.

## 학습 및 구현 목표

- Anima의 DiT 구조와 attention, 위치 임베딩, 시간·텍스트 조건 처리를 이해하고 필요한 코어 로직을 구성합니다.
- 플로우 매칭의 데이터·노이즈 보간, 시간 샘플링, velocity 예측과 학습 손실을 이해하고 생성·학습에 연결합니다.
- 학습한 모델의 예측을 이용해 노이즈에서 이미지를 생성하는 과정을 연결합니다.
- 데이터 입력부터 학습, 체크포인트 저장·불러오기, 이미지 생성까지 이 저장소에서 실행할 수 있도록 구현합니다.

## 참고 코드와 실행 의존성

`reference`의 DiffSynth-Studio 소스를 모델 구조, 생성 흐름, 학습 방식의 교재로 활용합니다. DiffSynth·Diffusers·ComfyUI 및 [anima_lora](https://github.com/sorryhyun/anima_lora)의 코드를 참고하거나 필요한 구현을 가져올 수 있습니다.

실행에는 PyTorch 등 일반 라이브러리와 모델 가중치를 사용합니다. 생성·학습에 필요한 코드는 이 저장소에 포함하여, DiffSynth·Diffusers·ComfyUI의 패키지 설치나 외부 리포의 코드 없이 실행하는 것을 목표로 합니다.

## 실행 환경

모델과 생성·학습 로직은 Colab과 독립적인 Python 모듈로 구성합니다. 초기 실행 환경은 Colab이며, 노트북은 실행 환경 준비와 설정, 저장소 모듈 호출을 담당합니다.

## 현재 구현 상태

현재 실습의 출발점은 Anima 이미지 생성입니다. DiffSynth에서 필요한 모델 정의를 분리했으며, Colab 노트북에서 텍스트 조건 처리, CFG, 플로우 매칭 기반 Euler 샘플링, VAE 디코딩을 실행합니다. LoRA 가중치 적용, 토큰 가중치, 모델 캐시와 실행 진단 기능도 제공합니다.

모델 학습 기능은 앞으로 구현할 목표입니다. 현재 LoRA 기능은 기존 가중치를 이미지 생성에 적용하는 기능입니다.

## 생성 예시

LoRA를 적용해 생성한 이미지입니다.

![Anima LoRA 생성 예시](assets/examples/anima_lora.png)

## 사용 안내

- [노트북 사용](docs/notebook_usage.md): 최초 실행, 설정 변경, 반복 생성
- [LoRA](docs/lora.md): 파일 준비, 지원 형식, 가중치 합산
- [토큰 가중치](docs/prompt_weights.md): 괄호 문법과 conditioning 처리
- [모델 캐시](docs/model_cache.md): CPU 보관, GPU 이동, 교체와 초기화
- [로그와 프로파일러](docs/diagnostics.md): 진행 출력과 병목 진단
- [런타임 구조와 분리 계획](docs/runtime_modularization.md): DiffSynth 비교와 향후 모듈 경계

## 크레딧

- **DiffSynth-Studio**: DiT·플로우 매칭 학습의 교재이자 Anima 모델 구현의 출처입니다. `src/anima_core/anima_dit.py`, `text_encoder.py`, `vae.py`는 `HisameOgasahara/DiffFlowDiT_test`에 포함된 DiffSynth-Studio의 Anima 관련 구현을 축약해 가져왔습니다. 원본 소스의 라이선스는 Apache-2.0입니다.
- **ComfyUI**: Anima 이미지 생성 실습의 참고 프로젝트입니다.

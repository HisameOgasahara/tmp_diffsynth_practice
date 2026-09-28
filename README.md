# Minimal Anima T2I practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_minimal_colab.ipynb)

DiffSynth의 Anima T2I 계산 경로에서 필요한 모델 정의만 떼어낸 Colab 실습입니다.

- **ComfyUI / Diffusers / DiffSynth 패키지를 설치하지 않습니다.**
- **그 세 저장소를 별도로 git clone하지 않습니다.**
- Colab에서는 이 저장소 하나만 clone합니다.
- 모델 가중치는 공개 Hugging Face `circlestone-labs/Anima`의 고정 revision에서 인증 없이 다운로드합니다.
- Qwen/T5 tokenizer는 공개 Hugging Face tokenizer를 사용합니다.
- 생성 HP와 prompt 기본값은 `DiffFlowDiT_test/config/generation.json`의 baseline 값입니다.
- sampler는 현재 **DiffSynth native의 Z-Image FlowMatch Euler**입니다. ComfyUI baseline의 ER-SDE를 재현하는 단계는 아직 넣지 않았습니다.

## 구성

`anima_minimal_colab.ipynb`은 다음처럼 단계별 독립 셀로 나뉩니다.

1. 일반 Python 라이브러리 설치
2. 이 저장소 clone
3. prompt / seed / steps / CFG / 크기 `@param`
4. Anima weights 다운로드
5. Qwen text encoding
6. Anima text adapter + DiT + CFG + Euler
7. VAE decode / PNG 저장

핵심 소스는 `src/anima_core/`에만 있습니다. DiffSynth의 범용 pipeline, model pool, VRAM manager, LoRA/ControlNet, registry 등은 포함하지 않습니다.

## 현재 범위

목적은 먼저 **프레임워크 없이 Anima T2I 계산 그래프를 직접 만지는 최소 기준점**을 만드는 것입니다. 이후 ComfyUI에서 확인한 tokenizer/conditioning/numerics 차이나 새 논문 구현을 이 core에 직접 바꾸어 붙이는 방향을 전제로 합니다.

Third-party source notice는 `THIRD_PARTY.md`를 확인하세요.

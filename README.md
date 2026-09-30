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
   - 선택: 로컬 / Google Drive / Hugging Face에서 Anima LoRA 파일 준비
5. Qwen text encoding
6. Anima text adapter + DiT + CFG + Euler
7. VAE decode / PNG 저장

핵심 소스는 `src/anima_core/`에만 있습니다. DiffSynth의 범용 pipeline, model pool, VRAM manager, ControlNet, registry 등은 포함하지 않습니다. LoRA 파일 처리와 가중치 합산은 자체 `lora.py`에서 수행합니다.

## LoRA 이미지 생성

노트북의 **4-1번 선택 셀**에서 `LORA_SOURCE`, `LORA_PATH`, `LORA_SCALE`을 설정합니다.

- 로컬 파일: `local`을 선택하고 `/content/my_anima_lora.safetensors`처럼 경로를 입력합니다.
- Google Drive: `google_drive`를 선택하고 `/content/drive/MyDrive/.../my_anima_lora.safetensors`처럼 경로를 입력합니다. 셀이 Drive를 마운트합니다.
- Hugging Face: `hugging_face`를 선택하고 `HF_LORA_REPO`, `HF_LORA_FILENAME`, `HF_LORA_REVISION`을 입력합니다. 다운로드 결과 경로를 설정에 저장합니다. 비공개 또는 gated 저장소는 HF 인증이 필요합니다.
- LoRA를 끄려면 `local`에서 `LORA_PATH`를 비우거나 선택 셀을 건너뜁니다. 3번 셀을 다시 실행하면 LoRA 설정도 초기화되므로 선택 셀을 다시 실행하세요.

6번 셀에서 `load_dit()`가 기본 가중치를 로드한 뒤 LoRA를 합칩니다. 그 다음 Anima text adapter와 Euler sampling을 실행하므로 `llm_adapter` 대상 LoRA도 conditioning에 반영됩니다. 경로나 강도를 변경했다면 6번과 7번 셀을 다시 실행합니다. prompt를 변경했다면 5번 셀부터 다시 실행합니다. 필요한 트리거 단어는 LoRA 배포 설명을 따라 prompt에 넣습니다.

Python에서는 같은 함수를 호출합니다.

```python
from anima_core.runtime import load_dit

dit = load_dit(base_path, lora_path=lora_path, lora_scale=0.8)
# 이어서 adapt_conditioning -> sample_euler -> decode_image
```

`src/anima_core/lora.py`는 DiffSynth의 `GeneralLoRALoader`와 `BasePipeline.load_lora()`의 fusion 경로를 바탕으로 `W += scale * (alpha / rank) * B @ A`를 계산합니다. 파일에 레이어별 `alpha`가 없으면 해당 배율은 1입니다. 행렬 곱셈과 덧셈은 레이어별 FP32로 계산한 뒤 기존 모델 dtype으로 저장합니다. 합산은 메모리에 로드한 모델에만 적용하며 원본 가중치 파일은 수정하지 않습니다. 같은 모델에 반복 적용하면 누적되므로 강도를 바꾸거나 기본 모델로 돌아갈 때는 `load_dit()`로 다시 로드합니다.

지원 범위는 Anima의 **Linear 레이어용 표준 `.safetensors` LoRA**입니다.

- `레이어.lora_A.weight` / `레이어.lora_B.weight` 및 `.default.weight` 형식
- `레이어.lora_down.weight` / `레이어.lora_up.weight` 및 선택적인 `레이어.alpha`
- `diffusion_model.`, `model.diffusion_model.`, `net.`, `transformer.`, `base_model.model.` 접두사
- 실제 Anima 레이어 이름의 점을 밑줄로 바꾼 `lora_unet_...` 형식

대응하는 레이어와 A/B 크기를 확인한 뒤 합산합니다. 지원하지 않는 텐서, 누락된 쌍, 다른 모델의 레이어는 오류로 표시합니다. Qwen text encoder용 LoRA, Conv LoRA, fused QKV 전용 LoRA, DoRA, LyCORIS, 라우팅을 사용하는 변형은 현재 지원하지 않습니다.

## 현재 범위

목적은 먼저 **프레임워크 없이 Anima T2I 계산 그래프를 직접 만지는 최소 기준점**을 만드는 것입니다. 이후 ComfyUI에서 확인한 tokenizer/conditioning/numerics 차이나 새 논문 구현을 이 core에 직접 바꾸어 붙이는 방향을 전제로 합니다.

Third-party source notice는 `THIRD_PARTY.md`를 확인하세요.

# LoRA 이미지 생성

## 파일 준비

노트북의 **4-1번 선택 셀**에서 `LORA_SOURCE`, `LORA_PATH`, `LORA_SCALE`을 설정합니다.

- 로컬 파일: `local`을 선택하고 `/content/my_anima_lora.safetensors`처럼 경로를 입력합니다.
- Google Drive: `google_drive`를 선택하고 `/content/drive/MyDrive/.../my_anima_lora.safetensors`처럼 경로를 입력합니다. 셀이 Drive를 마운트합니다.
- Hugging Face: `hugging_face`를 선택하고 `HF_LORA_REPO`, `HF_LORA_FILENAME`, `HF_LORA_REVISION`을 입력합니다. 다운로드 결과 경로를 설정에 저장합니다. 비공개 또는 gated 저장소는 HF 인증이 필요합니다.
- LoRA를 끄려면 `local`에서 `LORA_PATH`를 비운 뒤 4-1번 셀을 실행합니다. 설정 변경 후 실행 순서는 [노트북 사용](notebook_usage.md#반복-생성)을 참고하세요.

6번 셀에서 `load_dit()`가 기본 가중치를 로드한 뒤 LoRA를 합칩니다. 그 다음 Anima text adapter와 선택 sampler을 실행하므로 `llm_adapter` 대상 LoRA도 conditioning에 반영됩니다. 필요한 트리거 단어는 LoRA 배포 설명을 따라 prompt에 넣습니다.

## Python 호출

```python
from anima_core.runtime import load_dit

dit = load_dit(base_path, lora_path=lora_path, lora_scale=0.8)
# 이어서 adapt_conditioning -> sample_latents -> decode_image
```

`src/anima_core/lora.py`는 DiffSynth의 `GeneralLoRALoader`와 `BasePipeline.load_lora()`의 fusion 경로를 바탕으로 `W += scale * (alpha / rank) * B @ A`를 계산합니다. 파일에 레이어별 `alpha`가 없으면 해당 배율은 1입니다. 행렬 곱셈과 덧셈은 레이어별 FP32로 계산한 뒤 기존 모델 dtype으로 저장합니다. 합산은 메모리에 로드한 모델에만 적용하며 원본 가중치 파일은 수정하지 않습니다. 같은 모델에 반복 적용하면 누적되므로 강도를 바꾸거나 기본 모델로 돌아갈 때는 `load_dit()`로 다시 로드합니다.

## 지원 형식

지원 범위는 Anima의 **Linear 레이어용 표준 `.safetensors` LoRA**입니다.

- `레이어.lora_A.weight` / `레이어.lora_B.weight` 및 `.default.weight` 형식
- `레이어.lora_down.weight` / `레이어.lora_up.weight` 및 선택적인 `레이어.alpha`
- `diffusion_model.`, `model.diffusion_model.`, `net.`, `transformer.`, `base_model.model.` 접두사
- 실제 Anima 레이어 이름의 점을 밑줄로 바꾼 `lora_unet_...` 형식

대응하는 레이어와 A/B 크기를 확인한 뒤 합산합니다. 지원하지 않는 텐서, 누락된 쌍, 다른 모델의 레이어는 오류로 표시합니다. Qwen text encoder용 LoRA, Conv LoRA, fused QKV 전용 LoRA, DoRA, LyCORIS, 라우팅을 사용하는 변형은 현재 지원하지 않습니다.

관련 문서: [노트북 사용](notebook_usage.md) · [모델 캐시](model_cache.md)

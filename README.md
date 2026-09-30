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

## 진행 로그와 병목 진단

노트북 **3번 셀**에서 로그와 프로파일러 옵션을 설정합니다. 각 실행 셀은 설정 JSON을 다시 읽어 적용합니다.

| 옵션 | 동작 |
|---|---|
| `LOG_LEVEL` | `INFO`는 단계 시작/완료·경과 시간, `DEBUG`는 생성 스텝별 메시지도 표시 |
| `SHOW_PROGRESS` | 가중치 텐서 로딩과 Euler 생성의 현재/전체 진행률 표시 |
| `ENABLE_PROFILER` | 선택한 단계에서 PyTorch profiler 활성화; 기본값은 꺼짐 |
| `PROFILE_STAGES` | 기록할 함수 이름을 쉼표로 구분; 기본값은 `sample_euler` |
| `PROFILE_OUTPUT_DIR` | trace JSON과 연산별 요약 TXT를 저장할 폴더 |
| `PROFILE_WAIT`, `PROFILE_WARMUP`, `PROFILE_ACTIVE` | 생성 시 건너뛸 스텝, 준비 스텝, 실제 기록 스텝 수 |
| `PROFILE_RECORD_SHAPES` | 연산 입력 크기 기록 및 크기별 요약 |
| `PROFILE_MEMORY` | 메모리 할당/해제 기록 |
| `PROFILE_WITH_STACK` | 호출 스택 기록 |
| `PROFILE_ROW_LIMIT` | 요약 표에 표시할 연산 수 |

로그에는 가중치 다운로드/캐시 확인, tokenizer·Qwen·DiT·VAE 로딩, LoRA 적용, prompt encoding, text adapter, Euler 생성, VAE 디코딩이 표시됩니다. 단계별 경과 시간과 진행률의 속도/남은 시간은 CPU에서 관측한 값입니다. CUDA는 비동기로 실행하므로 GPU 연산의 정확한 시간은 profiler의 CUDA 표를 확인합니다.

`PROFILE_STAGES`에는 `load_text_encoder`, `encode_prompt`, `load_dit`, `adapt_conditioning`, `sample_euler`, `load_vae`, `decode_image`를 지정할 수 있습니다. 로딩까지 비교하려면 예를 들어 `load_dit,sample_euler,decode_image`를 입력합니다. 생성은 지정한 스텝 구간을 한 번 기록하며, 그 외 선택한 함수는 호출 전체를 기록합니다. 생성 스텝이 wait + warmup보다 작거나 같으면 프로파일을 생략한 이유를 출력하고 생성은 계속합니다.

프로파일 요약은 CPU/CUDA의 자식 연산을 제외한 시간 순으로 표시되고, 동일한 내용을 TXT로 저장합니다. trace에는 `weights/read_and_copy`, `model/initialize`, `model/lora_fusion`, `text/qwen_forward`, `text/anima_adapter`, `sampling/positive_dit`, `sampling/negative_dit`, `sampling/cfg`, `sampling/euler_update`, `vae/decode` 구간이 표시됩니다. 이를 통해 가중치 복사, positive/negative DiT, CFG 계산, 디코딩 중 어느 부분에 시간이 걸리는지 구분할 수 있습니다. 파일마다 고유 이름을 사용하므로 반복 실행 결과가 함께 남습니다.

trace JSON은 Perfetto 또는 Chrome trace viewer에서 열 수 있습니다. CUDA 커널 타임라인은 실행 환경의 CUPTI 지원에 따라 제공 여부가 달라집니다. profiler를 켜면 기록·동기화·파일 저장 비용이 생기므로, 일반 생성 속도와 profiler 실행 속도는 별도로 비교하세요. shape/memory/stack 옵션은 필요한 경우에만 켜는 편이 좋습니다. prompt encoding과 conditioning을 다시 실행하면 호출별 별도 파일이 만들어집니다.

Python에서도 동일한 설정을 사용합니다.

```python
from anima_core.runtime import configure_diagnostics

configure_diagnostics({
    "logging": {"level": "INFO", "progress": True},
    "profiler": {
        "enabled": True,
        "stages": ["load_dit", "sample_euler", "decode_image"],
        "output_dir": profile_output_dir,
        "warmup": 1,
        "active": 3,
    },
})
# 이후 기존 load_dit / sample_euler / decode_image 호출
```

`logging_utils.py`가 로그와 진행률 표시를, `profiling.py`가 profiler 설정·기록 구간·결과 저장을 담당합니다. `runtime.py`에는 단계 표시와 생성 스텝 경계만 연결되어 있습니다. profiler를 끄면 trace와 연산 요약을 생성하지 않습니다.

## 현재 범위

목적은 먼저 **프레임워크 없이 Anima T2I 계산 그래프를 직접 만지는 최소 기준점**을 만드는 것입니다. 이후 ComfyUI에서 확인한 tokenizer/conditioning/numerics 차이나 새 논문 구현을 이 core에 직접 바꾸어 붙이는 방향을 전제로 합니다.

Third-party source notice는 `THIRD_PARTY.md`를 확인하세요.

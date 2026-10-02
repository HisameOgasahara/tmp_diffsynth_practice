# 진행 로그와 병목 진단

노트북 **3번 셀**에서 로그와 프로파일러 옵션을 설정합니다.

| 옵션 | 동작 |
|---|---|
| `LOG_LEVEL` | `INFO`는 단계 시작/완료·경과 시간, `DEBUG`는 생성 스텝별 메시지도 표시 |
| `SHOW_PROGRESS` | 가중치 텐서 로딩과 선택 sampler 생성의 현재/전체 진행률 표시 |
| `ENABLE_PROFILER` | 선택한 단계에서 PyTorch profiler 활성화; 기본값은 꺼짐 |
| `PROFILE_STAGES` | 기록할 함수 이름을 쉼표로 구분; 기본값은 `sample_latents` |
| `PROFILE_OUTPUT_DIR` | trace JSON과 연산별 요약 TXT를 저장할 폴더 |
| `PROFILE_WAIT`, `PROFILE_WARMUP`, `PROFILE_ACTIVE` | 생성 시 건너뛸 스텝, 준비 스텝, 실제 기록 스텝 수 |
| `PROFILE_RECORD_SHAPES` | 연산 입력 크기 기록 및 크기별 요약 |
| `PROFILE_MEMORY` | 메모리 할당/해제 기록 |
| `PROFILE_WITH_STACK` | 호출 스택 기록 |
| `PROFILE_ROW_LIMIT` | 요약 표에 표시할 연산 수 |

로그에는 가중치 다운로드/캐시 확인, tokenizer·Qwen·DiT·VAE 로딩, LoRA 적용, prompt encoding, text adapter, 선택 sampler 생성, VAE 디코딩이 표시됩니다. 단계별 경과 시간과 진행률의 속도/남은 시간은 CPU에서 관측한 값입니다. CUDA는 비동기로 실행하므로 GPU 연산의 정확한 시간은 profiler의 CUDA 표를 확인합니다.

`PROFILE_STAGES`에는 `load_text_encoder`, `encode_prompt`, `load_dit`, `adapt_conditioning`, `sample_latents`, `sample_euler`, `load_vae`, `decode_image`, `cache_transfer`를 지정할 수 있습니다. 로딩까지 비교하려면 예를 들어 `load_dit,sample_latents,decode_image`를 입력합니다. 생성은 지정한 스텝 구간을 한 번 기록하며, 그 외 선택한 함수는 호출 전체를 기록합니다. 생성 스텝이 wait + warmup보다 작거나 같으면 프로파일을 생략한 이유를 출력하고 생성은 계속합니다.

프로파일 요약은 CPU/CUDA의 자식 연산을 제외한 시간 순으로 표시되고, 동일한 내용을 TXT로 저장합니다. trace에는 `weights/read_and_copy`, `model/initialize`, `model/lora_fusion`, `text/qwen_forward`, `text/anima_adapter`, `sampling/positive_dit`, `sampling/negative_dit`, `sampling/cfg`, `sampling/euler_update`, `vae/decode` 구간이 표시됩니다. 이를 통해 가중치 복사, positive/negative DiT, CFG 계산, 디코딩 중 어느 부분에 시간이 걸리는지 구분할 수 있습니다. 파일마다 고유 이름을 사용하므로 반복 실행 결과가 함께 남습니다.

trace JSON은 Perfetto 또는 Chrome trace viewer에서 열 수 있습니다. CUDA 커널 타임라인은 실행 환경의 CUPTI 지원에 따라 제공 여부가 달라집니다. profiler를 켜면 기록·동기화·파일 저장 비용이 생기므로, 일반 생성 속도와 profiler 실행 속도는 별도로 비교하세요. shape/memory/stack 옵션은 필요한 경우에만 켜는 편이 좋습니다. prompt encoding과 conditioning을 다시 실행하면 호출별 별도 파일이 만들어집니다.

Python에서도 동일한 설정을 사용합니다.

```python
from anima_core.runtime import configure_diagnostics

configure_diagnostics({
    "logging": {"level": "INFO", "progress": True},
    "profiler": {
        "enabled": True,
        "stages": ["load_dit", "sample_latents", "decode_image"],
        "output_dir": profile_output_dir,
        "warmup": 1,
        "active": 3,
    },
})
# 이후 load_dit / sample_latents / decode_image 호출
```

## 캐시 사용 시 진단

로그의 `cache hit`는 모델 재사용, `cache miss`는 최초 로딩 또는 교체를 뜻합니다. 장치 이동 메시지에서 CPU↔GPU 전환을 확인할 수 있습니다.

모델 캐시 사용 중에는 로딩 함수가 cache miss 때만 실행됩니다. 장치 이동 시간을 보려면 `PROFILE_STAGES`에 `cache_transfer`를 추가합니다. trace에서 `cache/model_transfer` 구간을 확인합니다. 캐시 보관·교체 정책은 [모델 캐시](model_cache.md)를 참고하세요.

관련 문서: [노트북 사용](notebook_usage.md)

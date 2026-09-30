# 모델 캐시

## 사용

노트북 3번 셀의 `USE_MODEL_CACHE`를 체크하면 CPU RAM에 로드한 모델을 재사용합니다. 체크하지 않으면 기존처럼 각 단계에서 파일을 읽고 모델을 준비합니다. 캐시를 끈 설정으로 실행 셀이 설정을 다시 적용하면 보관한 캐시도 비웁니다.

## 실행 흐름

DiffSynth의 단계별 onload/offload 방식을 참고한 최소 구현이며, DiffSynth를 설치하지 않습니다. `src/anima_core/model_cache.py`가 모델 보관과 장치 이동을 담당합니다.

| 셀 | 실행할 모델 | 단계 종료 후 |
|---|---|---|
| 5번 | Qwen text encoder를 GPU로 이동 | CPU RAM에 보관 |
| 6번 | LoRA가 적용된 DiT를 GPU로 이동 | CPU RAM에 보관 |
| 7번 | VAE를 GPU로 이동 | CPU RAM에 보관 |

처음 사용하는 모델은 CPU에서 가중치를 읽어 준비합니다. 이후에는 같은 객체를 GPU로 옮겨 사용하고 CPU로 돌려놓습니다. 단계 중 오류가 발생해도 CPU 복귀를 시도합니다. DiT 진단용 중간 텐서와 VAE 프레임 feature cache는 단계 종료 시 해제합니다.

## 재사용과 교체

모델 종류별로 하나씩 보관합니다. 재사용 조건은 가중치의 실제 경로·파일 크기·수정 시간과 dtype이며, DiT에는 LoRA 파일 정보와 강도도 포함합니다. 조건이 달라지면 해당 모델만 새로 준비합니다. LoRA를 바꾸거나 끄면 기본 가중치부터 다시 준비하므로 이전 LoRA 합산이 누적되지 않습니다.

prompt, seed, steps, CFG, 이미지 크기는 모델 캐시를 교체하지 않습니다. prompt를 바꾸면 5번부터, 생성 설정만 바꾸면 6번부터 다시 실행합니다. 3번 셀을 다시 실행했을 때 LoRA를 계속 쓰려면 4-1번 셀도 다시 실행합니다.

## 비용과 초기화

파일 재로딩은 줄지만 CPU↔GPU 전송은 매번 발생하며, Qwen·DiT·VAE 가중치가 CPU RAM을 사용합니다. HF의 다운로드 파일 캐시와 별개인 세션 메모리 캐시입니다. Colab 런타임 재시작 또는 2번 셀 재실행 시 초기화됩니다. 수동으로 비우려면 다음을 실행합니다.

```python
from anima_core.model_cache import clear_model_cache
clear_model_cache()
```

로그의 `cache hit`/`cache miss`와 장치 이동 메시지로 재사용 여부를 확인할 수 있습니다. 프로파일러의 `PROFILE_STAGES`에 `cache_transfer`를 추가하면 CPU↔GPU 이동을 기록합니다. `load_dit` 등 로딩 단계는 캐시 miss가 발생했을 때만 기록됩니다.

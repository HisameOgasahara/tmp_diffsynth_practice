# 프롬프트 토큰 가중치

## 사용

노트북 3번 셀에서 `USE_TOKEN_WEIGHTS`를 체크하고 프롬프트를 입력합니다. positive/negative 모두 적용하며, 변경 후 5 → 6 → 7번 셀을 실행합니다. 3번 셀 실행 후 LoRA를 계속 사용하려면 4-1번 셀도 다시 실행합니다. 체크하지 않으면 기존처럼 프롬프트 전체를 일반 문자열로 처리합니다.

| 입력 | 처리 |
|---|---|
| `(red hair)` | 해당 구간 가중치 1.1 |
| `(red hair:1.3)` | 해당 구간 가중치 1.3 |
| `((red hair))` | 기본 강조를 중첩하여 1.21 |
| `((red hair:1.3))` | 명시적 값이 현재 구간 가중치를 대체하므로 1.3 |
| `(red hair:0.7)` | 해당 구간 가중치 0.7 |
| `\(blue archive\)` | 괄호를 문자로 유지 |
| `[red hair]` | 대괄호는 일반 문자 |

괄호 안의 숫자는 외부 가중치와 곱하지 않고 현재 구간의 값을 대체합니다. 이는 확인한 ComfyUI의 `token_weights()` 규칙을 따릅니다. 캐릭터·작품 태그의 괄호를 문자로 남기려면 이스케이프합니다.

## 처리 경로

`src/anima_core/prompt_weights.py`가 프롬프트를 구간별로 해석하고 각 구간을 Qwen/T5 tokenizer로 따로 처리합니다. ComfyUI Anima처럼 Qwen 출력에는 가중치를 곱하지 않으며, T5 토큰에 대응하는 가중치를 Anima text adapter 출력에 곱합니다. T5의 추가 EOS 가중치는 1입니다. 토큰 가중치는 conditioning 파일에 함께 저장됩니다.

ComfyUI 코드를 복사하거나 패키지를 사용하지 않는 독립 구현입니다. 지원 범위는 위 괄호 문법이며, textual inversion과 ComfyUI의 전체 tokenizer/padding 경로는 포함하지 않습니다. 기존 실습의 최대 512 토큰과 padding 정책을 유지하므로 ComfyUI와 생성 이미지 전체가 동일함을 보장하지 않습니다. 모델 캐시는 재사용할 수 있지만 가중치를 바꾸면 conditioning을 다시 계산해야 합니다.

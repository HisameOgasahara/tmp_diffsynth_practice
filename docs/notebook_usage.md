# 노트북 사용

## 최초 실행

Colab에서 `anima_minimal_colab.ipynb`을 열고 아래 순서로 실행합니다.

| 셀 | 작업 |
|---|---|
| 1 | 일반 Python 라이브러리 설치 |
| 2 | 실습 저장소 clone 또는 최신 코드 반영 |
| 3 | prompt, seed, steps, CFG, 크기와 선택 기능 설정 |
| 4 | Anima 가중치 다운로드/캐시 확인; 선택 실행 |
| 4-1 | LoRA 파일 준비와 강도 설정; 선택 실행 |
| 5 | Qwen prompt encoding과 conditioning 파일 저장 |
| 6 | Anima text adapter, CFG, Euler 생성과 latent 저장 |
| 7 | VAE 디코딩과 PNG 저장 |

모델 가중치는 공개 HF `circlestone-labs/Anima`의 고정 revision에서 받으며, Qwen/T5 tokenizer도 HF에서 불러옵니다.

## 반복 생성

| 변경한 항목 | 다시 실행할 셀 |
|---|---|
| prompt, negative prompt, 토큰 가중치 | 3 → LoRA 사용 시 4-1 → 5 → 6 → 7 |
| seed, steps, CFG, 크기, shift, denoise | 3 → LoRA 사용 시 4-1 → 6 → 7 |
| LoRA 파일 또는 강도 | 4-1 → 6 → 7 |
| 로그·프로파일러·모델 캐시 옵션 | 3 → LoRA 사용 시 4-1 → 필요한 실행 단계 |

3번 셀은 설정 JSON을 새로 쓰며 LoRA 경로와 강도를 초기화합니다. LoRA를 계속 쓰려면 4-1번 셀을 다시 실행합니다. 각 실행 셀은 설정 JSON을 다시 읽습니다.

`RANDOM_SEED`를 체크하면 6번 셀 실행마다 새 시드를 선택하고 출력합니다. 체크하지 않으면 `SEED` 값을 사용합니다. 7번 셀의 PNG는 같은 출력 경로에 덮어씁니다.

같은 Colab 세션에서 1·2·4번 셀을 매번 실행할 필요는 없습니다. 저장소 업데이트를 받으려면 2번 셀을 실행합니다. 이 셀은 checkout을 원격 `main`으로 강제 갱신하고, 로드된 Python 모듈과 모델 캐시를 초기화합니다.

## 선택 기능

- [LoRA](lora.md)
- [토큰 가중치](prompt_weights.md)
- [모델 캐시](model_cache.md)
- [로그와 프로파일러](diagnostics.md)

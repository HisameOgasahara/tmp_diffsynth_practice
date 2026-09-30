# Minimal Anima T2I practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_diffsynth_practive/blob/main/anima_minimal_colab.ipynb)

PyTorch로 Anima의 diffusion / flow matching 이미지 생성 과정을 직접 다루는 실습 저장소입니다. DiffSynth에서 필요한 모델 정의를 분리했으며, ComfyUI·Diffusers·DiffSynth 설치 없이 실행합니다.

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

소스 출처와 라이선스 안내는 [THIRD_PARTY.md](THIRD_PARTY.md)를 참고하세요.

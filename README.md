# Minimal Anima T2I practice

DiffSynth의 Anima T2I 경로에서 필요한 부분만 떼어낸 실습용 저장소입니다.

- ComfyUI / Diffusers / DiffSynth 패키지 또는 별도 git clone을 사용하지 않습니다.
- 모델 가중치는 공개 Hugging Face `circlestone-labs/Anima`에서 인증 없이 다운로드합니다.
- 핵심 모델 정의는 DiffSynth 구현에서 Anima에 필요한 파일만 가져와 최소 의존성으로 연결합니다.
- 현재 샘플링은 DiffSynth native의 Z-Image FlowMatch Euler 경로입니다. 원본 ComfyUI baseline의 ER-SDE sampler를 재현하는 단계는 포함하지 않습니다.

Colab: `anima_minimal_colab.ipynb`

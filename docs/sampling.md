# 이미지 생성 샘플러

## 선택과 실행

생성 노트북의 3번 셀에서 `SAMPLER`를 선택합니다. 기본값은 기존 `euler`입니다. 샘플러·steps·CFG·seed만 변경했다면 3 → LoRA 사용 시 4-1 → 6 → 7번 셀을 실행합니다. 프롬프트도 변경했다면 5번 셀을 함께 실행합니다.

| `SAMPLER` | 방식 | N스텝의 예측 함수 호출 수 |
|---|---|---|
| `euler` | 기존 velocity 기반 Euler | N |
| `heun` | Euler 예측 후 다음 위치의 velocity로 보정 | 2N−1 |
| `euler_ancestral` | RF 전용 ancestral 갱신과 독립 정규 노이즈 | N |
| `dpmpp_2m` | ComfyUI 기본 2M 갱신식, 이전 denoised 예측 재사용 | N |
| `dpmpp_2m_sde` | RF logSNR 기반 2M SDE midpoint, Brownian noise | N |
| `er_sde` | RF ER-SDE, 최대 3단계 이력과 200점 수치 적분 | N |
| `exp_heun_2_x0` | RF logSNR·x0 기반 지수 Heun, phi2 보정 | 2N−1 |
| `exp_heun_2_x0_sde` | 지수 Heun의 확률적 갱신, 독립 정규 노이즈 | 2N−1 |
| `sa_solver` | RF Stochastic Adams, 기본 predictor 3·corrector 4 | N |
| `res_multistep` | 결정적 RES 2차 다단계 갱신 | N |
| `gradient_estimation` | 이전 속도의 차이로 Euler 보정, 기본 gamma 2 | N |

CFG가 1이면 예측 함수당 DiT를 한 번 호출하고, 그 외에는 positive·negative 조건으로 두 번 호출합니다. 모든 샘플러는 마지막 sigma=0 구간을 처리합니다. Heun은 마지막 구간에서 Euler를 사용합니다.

## 확률적 갱신 설정

| 설정 | 적용 대상 | 범위와 의미 |
|---|---|---|
| `ETA` | `euler_ancestral`, `dpmpp_2m_sde`, `exp_heun_2_x0_sde`, `sa_solver` | 0~1, 기본 1. 0이면 해당 방식의 결정적 갱신 사용 |
| `S_NOISE` | `euler_ancestral`, `dpmpp_2m_sde`, `er_sde`, `exp_heun_2_x0_sde`, `sa_solver` | 0 이상, 기본 1. 추가 노이즈 크기 |

`S_NOISE=0`은 노이즈 추가를 끕니다. `ETA`가 양수인 상태에서는 노이즈를 제거해도 결정적 Euler나 DPM++ 2M과 같은 결과가 되지는 않습니다. ER-SDE에는 `ETA`를 적용하지 않습니다.

SA Solver는 ComfyUI 기본값처럼 `percent_to_sigma(0.2)`부터 `percent_to_sigma(0.8)`까지 노이즈를 적용합니다. `ETA`는 이 구간의 tau 값입니다. PECE·CFG++ 변형은 포함하지 않습니다. exp-Heun 두 방식은 ComfyUI의 SEEDS-2 `r=1`, `solver_type="phi_2"` 경로를 사용합니다.

초기 노이즈와 추가 노이즈는 생성 seed에서 각각 독립적으로 준비합니다. 고정 seed로 비교하려면 `RANDOM_SEED`를 끕니다. 같은 실행 환경·설정에서는 반복 결과를 재현할 수 있으나, ComfyUI와 추가 노이즈의 생성 장치·정밀도가 달라 이미지가 완전히 같아지는 것은 보장하지 않습니다.

`dpmpp_2m_sde`의 Brownian noise에는 `torchsde==0.2.6`이 필요합니다. 생성 노트북의 1번 셀에서 설치합니다. Python 환경에서는 프로젝트 루트에서 다음 명령으로 설치합니다. 나머지 방식은 torchsde를 로드하지 않습니다.

```cmd
uv pip install --python .venv\Scripts\python.exe -r requirements-sampling.txt
```

## 공통 schedule과 RF 좌표

schedule은 기존 Z-Image 방식으로 고정합니다. `SHIFT`·`DENOISE`·steps로 sigma 목록을 만들고, 샘플러가 그 구간의 latent 갱신을 담당합니다. 현재 생성은 초기 노이즈에서 시작하는 T2I이며, `DENOISE`를 낮추는 동작도 기존 방식을 유지합니다. 입력 이미지 기반 img2img는 포함하지 않습니다.

Anima의 velocity에 CFG를 적용한 뒤 `denoised = x - sigma * velocity`로 변환합니다. Euler·Heun은 velocity를 직접 사용합니다. 신규 denoised 기반 샘플러는 FP32로 갱신식을 계산하고 다음 DiT 평가 전에 원래 latent dtype으로 복원합니다.

`dpmpp_2m`과 `res_multistep`은 ComfyUI 기본 구현의 `-log(sigma)` 시간축을 사용합니다. `dpmpp_2m_sde`, `er_sde`, exp-Heun 두 방식과 `sa_solver`는 RF용 logSNR 좌표를 사용하며, sigma=1에서 생기는 특이점을 피하도록 ComfyUI와 같은 `percent_to_sigma(1e-4)` 방식으로 시작 sigma를 보정합니다.

비교할 때 모델·LoRA·프롬프트·seed·schedule 설정을 고정합니다. 같은 steps에서의 결과와 같은 예측 호출 수 또는 생성 시간에서의 결과를 구분합니다. 수치 적분 차수가 높아도 이미지 품질이 항상 높아지는 것은 아닙니다.

## Python 호출과 모듈

```python
from anima_core.runtime import sample_latents

latents = sample_latents(
    dit, positive, negative, height, width, seed, steps, cfg_scale,
    sampler="dpmpp_2m_sde", eta=1.0, s_noise=1.0,
    shift=3.0, denoise=1.0, device="cuda", dtype=dtype,
)
```

기존 `runtime.sample_euler()` 호출과 `runtime.z_image_schedule()` 반환 형식도 유지합니다. 생성 노트북의 profiler 단계는 `sample_latents`입니다. 기존 Python Euler 호출을 기록할 때는 `sample_euler`를 사용할 수 있습니다.

| 모듈 | 책임 |
|---|---|
| `runtime.py` | 초기 latent·ModelPrediction 준비, 진행률·profiler 연결 |
| `sampling/runner.py` | sampler 선택·옵션 검사, schedule·추가 noise·RF 보정 준비, 실행과 예측 호출 수 계산 |
| `sampling/model_prediction.py` | DiT·CFG·denoised 변환 |
| `sampling/schedules.py` | sigma 목록과 RF 시작점 보정 |
| `sampling/noise.py` | seed 기반 정규·Brownian noise |
| `sampling/euler.py`, `heun.py`, `dpmpp.py`, `er_sde.py` | 방식별 갱신과 예측 이력 |
| `sampling/exp_heun.py`, `sa_solver.py`, `res_multistep.py`, `gradient_estimation.py` | 추가 5종의 갱신식·중간 예측·계수와 이력 |

기존 비교 노트북 `tmp_anima_sampler_compare_colab.ipynb`에서 11종을 선택할 수 있습니다. 같은 스텝 수와 비슷한 예측 호출 수의 실험을 지원하며, 호출 수는 `runner.expected_evaluations()`와 `runner.steps_for_evaluations()`로 계산합니다. 목표가 30회이면 Heun·exp-Heun은 15스텝·29회, 나머지는 30스텝·30회입니다.

LoRA 가중치는 생성 전에 기존 방식으로 적용합니다. 학습 경로는 생성 샘플러를 호출하지 않습니다.

## 구현 출처

갱신식과 Brownian noise 처리는 [ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/1b883beab11c04a2eb82cf6a3ee64294f3303897/comfy/k_diffusion/sampling.py)를 Anima 전용 함수에 맞춰 옮겼습니다. ComfyUI에서 가져와 수정한 샘플러 코드는 GPL-3.0 조건을 따르며, 원문은 [ComfyUI-LICENSE](../licenses/ComfyUI-LICENSE)에 포함합니다. DiffSynth에서 가져온 모델 코드의 출처와 Apache-2.0 고지는 유지합니다.

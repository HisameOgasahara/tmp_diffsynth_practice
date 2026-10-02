"""Anima 생성용 샘플러 선택. 갱신식 출처는 ComfyUI입니다."""

from .dpmpp import sample_dpmpp_2m, sample_dpmpp_2m_sde
from .er_sde import sample_er_sde
from .euler import sample_euler, sample_euler_ancestral
from .heun import sample_heun


SAMPLERS = {
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "er_sde": sample_er_sde,
}

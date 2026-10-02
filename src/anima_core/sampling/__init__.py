"""Anima 생성용 샘플러 선택. 갱신식 출처는 ComfyUI입니다."""

from .dpmpp import sample_dpmpp_2m, sample_dpmpp_2m_sde
from .er_sde import sample_er_sde
from .euler import sample_euler, sample_euler_ancestral
from .heun import sample_heun
from .exp_heun import sample_exp_heun_2_x0, sample_exp_heun_2_x0_sde
from .sa_solver import sample_sa_solver
from .res_multistep import sample_res_multistep
from .gradient_estimation import sample_gradient_estimation


SAMPLERS = {
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "er_sde": sample_er_sde,
    "exp_heun_2_x0": sample_exp_heun_2_x0,
    "exp_heun_2_x0_sde": sample_exp_heun_2_x0_sde,
    "sa_solver": sample_sa_solver,
    "res_multistep": sample_res_multistep,
    "gradient_estimation": sample_gradient_estimation,
}

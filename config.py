# -*- coding: utf-8 -*-
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # ===== 基本运行 =====
    rl_dir: str = r"E:\python\DDPG\rl_io"
    mode: str = "train"                  # "train" | "val" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 编码器可执行路径（main.py 可覆盖）=====
    encoder_path: str = r"E:\Git\qav1_ori\qav1\build\vs2022\x64\Debug\qav1enc.exe"

    # ===== 切换：单视频命令 vs 数据集模式 =====
    use_dataset: bool = False            # True=数据集模式；False=单视频模式

    # ===== QP 边界 =====
    qp_min: int = 80
    qp_max: int = 200
    delta_qp_max: float = 20.0
    fps: int = 30

    # ===== 算法选择 =====
    algo: str = "td3_lag"               # "td3_lag" | "td3"
    # TD3/TD3-Lagrangian 超参
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 2e-4
    batch_size: int = 32
    replay_size: int = 50000
    warmup_steps: int = 2000
    train_steps_per_env_step: int = 4
    policy_noise: float = 0.10
    noise_clip: float = 0.20
    policy_delay: int = 2
    expl_noise_std: float = 0.15
    action_eps_train: float = 0.10
    action_eps_infer: float = 0.00
    target_discretize: bool = True

    # ===== Lagrangian：miniGOP 末端约束（对照 2-pass）=====
    ref_bits_tol: float = 0.10          # 允许 ±10%
    lag_b_init: float = 0.0
    lag_q_init: float = 0.0
    lag_eta_b:  float = 0.5
    lag_eta_q:  float = 0.5
    lag_b_max:  float = 50.0
    lag_q_max:  float = 50.0

    # ===== Reward 形状项（逐帧）=====
    psnr_min_db: float = 38.0
    nash_eps: float = 1e-6
    ud_ema_beta: float = 0.90

    smooth_ema_beta: float = 0.90
    smooth_huber_delta: float = 0.50
    w_smooth: float = 0.35

    grad_huber_delta: float = 0.70
    w_grad: float = 0.20
    sc_grad_amp: float = 0.80

    inter_smooth_enable: bool = True
    inter_global_ema_beta: float = 0.98
    inter_smooth_first_k: int = 3
    inter_smooth_huber_delta: float = 0.80
    w_inter: float = 0.15
    inter_gate_sc: float = 0.0

    use_per_frame_lambda_bits: bool = False
    lambda_init: float = 1e-3
    lambda_lo: float = 1e-6
    lambda_hi: float = 1e+2
    lambda_eta: float = 0.5
    bit_gate_hi: float = 1.0
    bit_gate_lo: float = 0.25
    avg_psnr_ema_beta: float = 0.98
    psnr_target_db: float = 40.5
    min_bpf: float = 500.0

    mg_tol: float = 0.05
    mg_huber_delta: float = 0.05
    mg_early_amp: float = 1.0
    mg_early_exp: float = 0.9

    sc_p_quality_boost: float = 0.80
    sc_p_bit_gate: float = 0.60

    reward_balance_auto: bool = True
    reward_balance_momentum: float = 0.95
    reward_balance_target_mag: float = 0.8
    reward_clip: float = 1.5
    reward_scale: float = 1.0

    # ===== 安全层（QP 限幅与回退）=====
    safety_layer_enable: bool = False
    safety_slack: float = 1.05
    safety_qp_step: int = 2

    # ===== 2-pass 基线日志路径（main 会按 --stat-in 自动推导）=====
    twopass_log_path: str = ""

    # ===== 打印/日志 =====
    print_every_sec: float = 2.0
    loss_ema_beta: float = 0.20
    metrics_csv: str = "epoch_metrics.csv"
    encoder_log_to_file: bool = True
    encoder_log_dir: str = "./logs/encoder"
    show_encoder_output: bool = False
    hide_encoder_console_window: bool = True
    # config.py
    use_nash: bool = True
    nash_scale: float = 1.0
    linq_scale: float = 1.0  # 关掉nash时，线性质量项的缩放

    # ==== Reward 策略 ====
    reward_variant: str = "mg_end_lagrange_only"  # 新增：逐帧仅形状项，末帧用 2-pass 对齐
    use_budget_features_in_state: bool = False  # 新增：状态里屏蔽“预算/进度”等预计量

    # （可选）miniGOP软预算项也关掉：只依赖2-pass
    disable_mg_soft_budget_term: bool = True


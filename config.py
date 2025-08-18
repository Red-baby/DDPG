# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # 运行
    rl_dir: str = r"E:\python\DDPG\rl_io"
    mode: str = "train"                       # "train" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 编码器
    encoder_path: str = r"E:\Git\qav1_ori\qav1\build\vs2022\x64\Release\qav1enc.exe"          # 修改为你的编码器路径
    # 你可以在 main.py 里按视频逐个传入命令行参数；这里仅做兜底
    encoder_args: list[str] = field(default_factory=lambda: ["-i", "input.yuv", "-o", "out.ivf"])
    kill_encoder_on_exit: bool = True

    # QP 边界（固定常数，不再放入状态）
    qp_min: int = 25
    qp_max: int = 55

    # 训练
    gamma: float = 0.98
    tau:   float = 0.005
    actor_lr:  float = 2e-4
    critic_lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 400_000
    warmup_steps: int = 100
    train_steps_per_env_step: int = 1
    save_every_steps: int = 20_000
    ckpt_dir: str = "./checkpoints"

    # 探索
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1.0
    action_eps_train: float = 0.10
    action_eps_infer: float = 0.00

    # 奖励（保持你现有逻辑；这里仅保留与PSNR/平滑/GOP相关）
    psnr_norm: float = 45.0
    w_psnr: float = 1.0
    w_over: float = 1.2
    w_under: float = 0.4
    min_bpf: float = 200.0
    w_smooth: float = 0.25
    smooth_ref_db: float = 5.0
    alpha_credit_share: float = 0.5
    w_gop_risk: float = 0.2

    # 观测：采用 PSNR（"y" 或 "yuv"）
    psnr_mode: str = "y"  # "y" | "yuv"

    # 归一化
    feature_clip: float = 10.0
    norm_momentum: float = 0.01

    # 是否跨段/跨GOP继承上一帧观测（你要的功能：默认开启）
    carry_prev_across_mg: bool = True
    carry_prev_across_gop: bool = True

    # 日志
    print_every_sec: float = 2.0

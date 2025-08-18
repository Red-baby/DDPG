# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch, os

@dataclass
class Config:
    # 运行
    rl_dir: str = r"D:\Python\DDPG\rl_io"
    mode: str = "train"                       # "train" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 编码器
    encoder_path: str = r"D:/Python/DDPG/qav1enc.exe"
    encoder_args: list[str] = field(default_factory=lambda: ["-i", "input.yuv", "-o", "out.ivf"])

    # QP 边界
    qp_min: int = 10
    qp_max: int = 200

    # 训练
    gamma: float = 0.98
    tau:   float = 0.005
    actor_lr:  float = 2e-4
    critic_lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 2000
    warmup_steps: int = 200
    train_steps_per_env_step: int = 1

    # 检查点
    ckpt_dir: str = "./checkpoints"
    save_every_epoch: bool = True

    # 探索
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1.0
    action_eps_train: float = 0.10
    action_eps_infer: float = 0.00

    # ====== 奖励配置（总尺度与优先级）======
    reward_scale: float = 0.01          # 建议保留较小，避免 Q 爆
    psnr_norm: float = 45.0
    # 主项：目标得分与目标比特的权重（前二位最重要）
    w_score_main: float = 1.0           # (PSNR - target_score) / psnr_norm
    w_bit_main: float = 0.8             # - ((bits - target_bpf)/target_bpf)^2
    # 次要：质量稳定
    w_smooth: float = 0.30
    smooth_ref_db: float = 5.0

    # 旧的基础比特惩罚（仍备用，不再门控）
    w_over: float = 0.10
    w_under: float = 0.05
    min_bpf: float = 200.0

    # ====== RL 侧目标（开启后优先使用）======
    use_rl_targets: bool = True
    # 目标比特：二选一（若两者皆给，优先 target_bpf）
    target_bpf: float = 2125.0             # 每帧目标比特（bits per frame）
    target_bitrate_kbps: float = 0.0    # 目标码率（kbps）
    target_fps: float = 30.0             # FPS（用于由码率推导 bpf）
    # 目标得分（PSNR, dB）
    target_score_avg: float = 42
    target_score_min: float = 38.5

    # mini-GOP 级强惩罚
    overshoot_factor_mg: float = 2.0    # 超过目标的倍数阈值（默认 2 倍）
    w_mg_overshoot_hard: float = 2.0    # 超阈强惩罚权重
    w_mg_score_below_hard: float = 2.0  # mini-GOP 平均 PSNR 低于目标的强惩罚

    # 全局最终偏差强惩罚
    w_global_bits_dev: float = 0.5      # 全局平均 bpf 偏差（超过容忍度）强惩罚
    global_bits_dev_tol: float = 0.15   # 容忍度（相对偏差 15%）

    # 观测：PSNR 取法
    psnr_mode: str = "y"  # "y" | "yuv"

    # GOP credit 开关
    use_gop_credit: bool = True
    alpha_credit_share: float = 0.5
    w_gop_risk: float = 0.2

    # 归一化
    feature_clip: float = 10.0
    norm_momentum: float = 0.01

    # 状态继承
    carry_prev_across_mg: bool = True
    carry_prev_across_gop: bool = True

    # ====== 日志/打印 ======
    print_every_sec: float = 2.0
    loss_ema_beta: float = 0.20
    show_encoder_output: bool = False
    hide_encoder_console_window: bool = True
    encoder_log_to_file: bool = True
    encoder_log_dir: str = "./logs/encoder"
    kill_encoder_on_exit: bool = True

    # 数据（以视频为单位的 epoch）
    video_list: list[str] = None

    def __post_init__(self):
        if self.video_list is None:
            self.video_list = ["./videos/seq1.y4m"]
        os.makedirs(self.rl_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.encoder_log_dir, exist_ok=True)

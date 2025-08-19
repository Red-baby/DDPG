# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch, os

@dataclass
class Config:
    # 运行
    rl_dir: str = r"E:\Python\DDPG\rl_io"
    mode: str = "train"                       # "train" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 编码器
    encoder_path: str = r"D:/Python/DDPG/qav1enc.exe"          # 修改为你的编码器路径
    # 你可以在 main.py 里按视频逐个传入命令行参数；这里仅做兜底
    encoder_args: list[str] = field(default_factory=lambda: ["-i", "input.yuv", "-o", "out.ivf"])

    # QP 边界（固定常数，不再放入状态）
    qp_min: int = 30
    qp_max: int = 200

    # 训练
    gamma: float = 0.98
    tau:   float = 0.005
    actor_lr:  float = 2e-4
    critic_lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 2000
    warmup_steps: int = 200
    train_steps_per_env_step: int = 4

    # 检查点
    ckpt_dir: str = "./checkpoints"
    save_every_epoch: bool = True

    # 探索
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1.0
    action_eps_train: float = 0.10
    action_eps_infer: float = 0.00

    # 奖励（PSNR 优先，其次平滑，最后比特）
    psnr_norm: float = 45.0
    w_psnr: float = 1.5
    # 比特偏差基准权重（会被PSNR门控缩放）
    w_over: float = 10
    w_under: float = 0.05
    min_bpf: float = 200.0

    # 平滑权重略高于比特
    w_smooth: float = 0.30
    smooth_ref_db: float = 5.0

    # GOP 信用项/风险项
    alpha_credit_share: float = 0.5
    w_gop_risk: float = 0.2
    use_gop_credit: bool = False  # <<< 宏：关闭后不使用 gop_credit

    # 观测：采用 PSNR（"y" 或 "yuv"）
    psnr_mode: str = "y"  # "y" | "yuv"

    # 当 PSNR 达标时更重视省比特；未达标时弱化比特惩罚
    psnr_target_db: float = 42.5   # 若 rq 里有 score_avg/score_min，会优先用 rq 的
    bit_gate_hi: float = 1.0       # psnr >= target：比特权重倍率
    bit_gate_lo: float = 0.25      # psnr <  target：比特权重倍率

    # 归一化
    feature_clip: float = 10.0
    norm_momentum: float = 0.01

    # 是否跨段/跨GOP继承上一帧观测（默认开启）
    carry_prev_across_mg: bool = True
    carry_prev_across_gop: bool = True

    # ====== 日志/打印 ======
    print_every_sec: float = 2.0
    loss_ema_beta: float = 0.20  # 打印时的EMA平滑
    show_encoder_output: bool = False                # 不把编码器 stdout/stderr 打到窗口
    hide_encoder_console_window: bool = True         # Windows 下隐藏子进程控制台窗口
    encoder_log_to_file: bool = True                 # 将编码器 stdout/stderr 落到文件
    encoder_log_dir: str = "./logs/encoder"          # 编码器日志目录
    kill_encoder_on_exit: bool = True
    print_epoch_gop_stats: bool = False
    # ===== 帧/mini-GOP 分梯度惩罚与效率相关超参 =====
    # 帧级：PSNR 达标时，对“省码”（低于预估）给少量正向奖励；未达标时不奖励省码
    w_save_bonus: float = 0.2

    # 帧级：当该帧实际 ≥ 2× 预估时的“硬惩罚”系数与阈值
    over_hard_ratio_frame: float = 2.0
    w_over_hard_frame: float = 3.0

    # 帧级：低效超码（ΔPSNR 很小却高码）惩罚
    w_ineff_over: float = 1.0
    eff_gain_eps: float = 0.10  # dB，小于此认为“提升不明显”

    # mini-GOP 级：累计超出分梯度惩罚（每帧都生效）
    w_mg_over: float = 0.8  # 常规累计超出惩罚（分梯度）
    mg_over_hard_ratio: float = 2.0  # 累计严重超出阈值（如 ≥2×）
    w_mg_over_hard: float = 3.5  # 累计严重超出额外惩罚

    # ====== 数据（以视频为单位的 epoch）======
    # main.py 会循环这些视频作为多个 epoch
    video_list: list[str] = None

    def __post_init__(self):
        if self.video_list is None:
            self.video_list = ["./videos/seq1.y4m"]
        os.makedirs(self.rl_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.encoder_log_dir, exist_ok=True)

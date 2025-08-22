# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import torch, os

@dataclass
class Config:
    # 运行
    rl_dir: str = r"D:\Python\DDPG\rl_io"
    mode: str = "train"                       # "train" | "infer"| "val"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 编码器
    encoder_path: str = r"D:/Python/DDPG/qav1enc.exe"          # 修改为你的编码器路径
    # 你可以在 main.py 里按视频逐个传入命令行参数；这里仅做兜底
    encoder_args: list[str] = field(default_factory=lambda: ["-i", "input.yuv", "-o", "out.ivf"])

    # QP 边界（固定常数，不再放入状态）
    qp_min: int = 80
    qp_max: int = 200
    fps: int = 30

    # === Algorithm ===
    algo: str = "dual"  # "ddpg" or "td3"
    over_budget_factor: float = 1.0  # 超预算判定阈值因子（预算×因子）
    rR_target_factor:  float = 1.0   # 可选：rR 的目标也按该因子缩放（默认1.0不变）
    # Exploration (online action noise)
    expl_noise_std: float = 0.15  # 给 select_action 的高斯噪声（取代 OU）

    # TD3 core
    policy_noise: float = 0.10  # 目标策略平滑噪声 N(0, policy_noise)
    noise_clip: float = 0.20  # 上述噪声裁剪幅度
    policy_delay: int = 2  # 每隔多少次 Critic 更新才更新一次 Actor
    target_discretize: bool = True  # [可选] 目标动作也按 QP 量化再映回 a01，降低“学到的目标”和真实环境不一致

    # 训练
    gamma: float = 0.98
    tau:   float = 0.005
    actor_lr:  float = 1e-4
    critic_lr: float = 2e-4
    batch_size: int = 4
    replay_size: int = 10000
    warmup_steps: int = 1000
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
    w_psnr: float = 3
    # 比特偏差基准权重（会被PSNR门控缩放）
    w_over: float = 5
    w_under: float = 0.05
    min_bpf: float = 200.0

    # 平滑权重略高于比特
    w_smooth: float = 5.0
    smooth_ref_db: float = 5.0

    # GOP 信用项/风险项
    alpha_credit_share: float = 0.5
    w_gop_risk: float = 0.2
    use_gop_credit: bool = False  # <<< 宏：关闭后不使用 gop_credit

    # 观测：采用 PSNR（"y" 或 "yuv"）
    psnr_mode: str = "y"  # "y" | "yuv"

    # 当 PSNR 达标时更重视省比特；未达标时弱化比特惩罚
    psnr_target_db: float = 42.5   # 若 rq 里有 score_avg/score_min，会优先用 rq 的
    psnr_min_db: float = 40.0
    bit_gate_hi: float = 1.0       # psnr >= target：比特权重倍率
    bit_gate_lo: float = 0.25      # psnr <  target：比特权重倍率

    # mini-GOP 早超惩罚：越早越重
    mg_early_amp: float = 1.0  # 放大量，1.0 表示最高可再乘 2 倍（=1+1）
    mg_early_exp: float = 0.9  # 曲线指数；>1 更偏向最早几帧，<1 更平滑

    # 当 PSNR 未达标时，平滑项的缩放系数（0 = 直接关闭；0.25 = 只保留 25%）
    w_smooth_under_scale: float = 0.25

    # 可选：随着欠标幅度(以 smooth_ref_db 归一)再额外衰减；0 表示不随欠标变化
    smooth_under_boost: float = 0.5

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

    # 失真项映射
    q_mid_db = 38.0  # PSNR 中位（数据集大致水平）
    q_span_db = 2.0  # 每 2dB 作为 1 个尺度 -> tanh(-1..1)

    # 帧级码率项
    pf_tol = 0.20  # 单帧容忍 ±20%
    pf_huber_delta = 0.20
    w_bf = 0.3

    # mini-GOP 级码率项
    mg_tol = 0.05  # mini-GOP 容忍 ±5%
    mg_huber_delta = 0.05
    w_bmg = 2.5

    # 自适应平衡
    reward_balance_auto = True
    reward_balance_momentum = 0.95
    reward_balance_target_mag = 0.8  # 期望两路项的平均幅度

    # === Logging ===
    metrics_csv: str = "epoch_losses_822.csv"  # 记录每个 epoch 的平均 lossa/lossc

    # 其他
    w_q = 1.0
    min_bpf = 500.0
    reward_clip = 1.5
    reward_scale = 1.0

    # lookahead_cost 的量级缩放（把几十万缩到~1，再 log1p）
    lookahead_scale: float = 1e5  # 你的实际量级如果不是 1e5，自行改这个

    # mini-GOP 级：累计超出分梯度惩罚（每帧都生效）
    w_mg_over: float = 0.8  # 常规累计超出惩罚（分梯度）
    mg_over_hard_ratio: float = 2.0  # 累计严重超出阈值（如 ≥2×）
    w_mg_over_hard: float = 3.5  # 累计严重超出额外惩罚
    # q_gain 分段形状（幅度可按需调）
    q_between_neg: float = 0.30  #  在 psnr=psnr_min 时，q_gain 约为 -q_between_neg（“略靠近0的负”）
    q_under_min_scale: float = 1.00  # 低于最低线时的负向斜率（越大越负）
    q_above_scale: float = 1.00  # 高于 target 的正向斜率（到 psnr_norm 处大约到 +q_above_scale）
    q_gain_cap: float = 1.00  # 夹紧幅度，限制 q 的绝对值（防止过大）
    # ====== 数据（以视频为单位的 epoch）======
    # main.py 会循环这些视频作为多个 epoch
    video_list: list[str] = None

    # 以 baseQP 为中心的动作映射窗口半径
    base_qp_window: int = 20

    def __post_init__(self):
        if self.video_list is None:
            self.video_list = ["./videos/seq1.y4m"]
        os.makedirs(self.rl_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.encoder_log_dir, exist_ok=True)

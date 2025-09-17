# -*- coding: utf-8 -*-
"""
config.py — 全量可直接替换版本

说明：
- 适配 miniGOP 向量动作（一次性输出当前 miniGOP 所有帧的 QP）。
- 与 main.py / io_runner.py / agent.py / models.py 兼容。
- 若需覆盖默认值，可在命令行或外部脚本里按需修改对应字段。
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ===== 基本设置 =====
    rl_dir: str = r"./rl_io"                     # 与编码器端 rl_set_dir 一致
    mode: str = "train"                          # "train" | "val" | "infer"
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 编码器相关 =====
    encoder_path: str = r"./qav1enc.exe"         # 可被 main.py --encoder 覆盖
    fps: int = 24                                # 仅用于日志/统计（kbps 等）

    # ===== miniGOP 设置 =====
    mg_size: int = 16                            # 常规 miniGOP 显示帧数（不足用“复制末帧”补齐为 16）
    qp_min: int = 0                              # 绝对 QP 下界（含）
    qp_max: int = 255                            # 绝对 QP 上界（含）

    # ===== Actor/Critic 网络与优化器 =====
    hidden_dim: int = 512                        # 主干隐藏维度
    depth: int = 4                               # ResBlock 层数（越大越强但更慢）
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    gamma: float = 0.99                          # 折扣因子
    tau: float = 0.005                           # 软更新系数 Polyak
    policy_noise: float = 0.05                   # 目标策略噪声幅度（动作空间 [0,1]）
    noise_clip: float = 0.20                     # 目标策略噪声裁剪
    policy_delay: int = 2                        # TD3 策略延迟更新步数
    batch_size: int = 32
    replay_size: int = 10000
    explore_eps: float = 0.10                    # 训练时 Actor 输出加性高斯噪声强度

    # ===== 训练节奏与日志 =====
    train_steps_per_env_step: int = 4            # 每次收到一个 FB 做几步训练
    loss_ema_beta: float = 0.2                   # 打印时的损失 EMA 系数
    print_every_sec: float = 2.0                 # 控制台打印间隔（秒）
    ckpt_prefix: str = "ckpt"                    # 保存/加载时的前缀（如需）
    # 训练相关
    warmup_min_transitions: int = 16

    # ===== 回报函数（miniGOP 级）=====
    # 约束1：bit_avg ∈ [rate_band_low, rate_band_high] × ref_bit_avg
    rate_band_low: float = 0.90
    rate_band_high: float = 1.05
    mg_bits_penalty_gain: float = 2.0            # 码率越界的惩罚斜率（已废弃）

    # 硬约束惩罚参数
    overbit_hard_penalty: float = 10.0           # 超过上限时的强惩罚值（已废弃）
    overbit_bypass_clip: bool = True              # 硬惩罚是否绕过reward_clip
    low_bit_penalty: float = 4.0                 # 低于下限时的惩罚强度（已废弃）
    
    # 双重硬约束参数（新设计）
    hard_penalty_bitrate: float = 15.0           # 比特率违约强惩罚
    hard_penalty_quality: float = 20.0           # 质量违约强惩罚（更重）

    # 约束2：vmaf_avg ≥ ref_vmaf_avg（质量硬约束 + 奖励）
    mg_vmaf_gain_pos: float = 0.30               # VMAF 高于参考的奖励斜率（只有正奖励）
    mg_vmaf_gain_neg: float = 0.60               # 已废弃（现在用硬约束）
    
    # 比特率效率奖励
    bit_efficiency_gain: float = 0.10            # 比特率效率奖励（越省比特率越好）

    # ===== 其它可选（兼容/备用）=====
    delta_qp_max: int = 20                       # 若改为 ΔQP 方案时可复用（当前为绝对 QP 向量输出）
    twopass_log_path: str = ""                   # 由 main.py/_run_one_video 自动推导并注入
    # 是否把编码器的 stdout/stderr 打到控制台（True）还是丢弃（False）
    show_encoder_output: bool = False

    # 仅 Windows 生效：是否隐藏编码器控制台窗口
    hide_encoder_console_window: bool = True
    # ===== 编码器日志 =====
    encoder_log_to_file: bool = True  # 打开文件日志
    encoder_log_dir: str = r"./logs/encoder"  # 日志目录
    
    # ===== 训练数据记录 =====
    log_dir: str = r"./logs"  # 训练数据和图表保存目录
    
    # ===== 模型保存设置 =====
    checkpoint_dir: str = r"./checkpoints"    # 模型保存目录
    save_every_epochs: int = 10                # 每隔几个epoch保存一次模型
    keep_last_n_checkpoints: int = 5           # 保留最近几个checkpoint

    # （可选）让日志文件名带上 epoch
    curr_epoch: int = 0  # main/runner 在每个 epoch 开始时更新它

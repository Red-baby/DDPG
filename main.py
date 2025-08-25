# -*- coding: utf-8 -*-
"""
以“视频为单位”的 epoch 训练/验证入口（每个 epoch 启动一次编码器）：
- 方式一：--videos 手工传多条命令（每条内部用“|”分隔）；每个 epoch 轮转一条；
- 方式二：数据集模式（推荐多序列）：--dataset-* 自动生成每条 2-pass 命令（含 --stat-in）；
- 不与 run_2pass.py 耦合；数据集逻辑下沉到 dataset.py。
"""

import os
import argparse
import threading
import glob
from typing import Optional, List

from pathlib import Path
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor
import dataset as ds


# ---------------------
# 参数解析
# ---------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # 基本参数
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="起始 epoch 编号，用于继续训练时的显示与命名")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "val", "infer"])
    ap.add_argument("--encoder", type=str,
                    default=r"E:\Git\qav1_ori\qav1\build\vs2022\x64\Release\qav1enc.exe")
    ap.add_argument("--resume", type=str, default=r"",
                    help="可选：ckpt 文件或目录；从该 checkpoint 继续训练/验证")
    ap.add_argument("--ckpt-prefix", type=str, default="ckpt",
                    help="保存检查点的前缀名")

    # 方式一：与原来一致，手工传多条 --videos（每条内部用“|”分隔参数）
    ap.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=[],  # 若不使用数据集模式，则需要显式给 --videos
        help="每条是一段完整的编码器 2-pass 命令（内部用 | 分隔），每个 epoch 依次取一条运行",
    )

    # 方式二：数据集模式（新增，参数由 dataset.py 注入）
    ds.add_dataset_args(ap)

    return ap.parse_args()


# ---------------------
# 恢复 ckpt
# ---------------------
def _latest_ckpt(path_or_dir: str) -> Optional[str]:
    if not path_or_dir:
        return None
    p = os.path.abspath(path_or_dir)
    if os.path.isdir(p):
        cands = sorted(glob.glob(os.path.join(p, "*.pth")) + glob.glob(os.path.join(p, "*.pt")))
        return cands[-1] if cands else None
    if os.path.isfile(p):
        return p
    return None


# ---------------------
# 主流程
# ---------------------
def main():
    args = parse_args()

    cfg = Config(rl_dir=args.rl_dir, mode=args.mode)
    if args.encoder:
        cfg.encoder_path = args.encoder

    if not os.path.exists(cfg.encoder_path):
        print(f"[ERROR] 编码器不存在：{cfg.encoder_path}")
        print("请检查 config.py 的 encoder_path 或使用 --encoder 指定正确路径")
        return

    # Runner（RL agent + I/O）
    runner = RLRunner(cfg)

    # 断点恢复（可给目录或具体文件）
    ckpt_path = _latest_ckpt(args.resume)
    if ckpt_path:
        try:
            ckpt = runner.agent.load_checkpoint(ckpt_path)
            try:
                sn = ckpt.get("state_norm", None)
                if sn is not None:
                    runner.sb.norm.load_state_dict(sn)
                    print(f"[MAIN] resumed from: {ckpt_path} (with state_norm)")
                else:
                    print(f"[MAIN] resumed from: {ckpt_path} (no state_norm)")
            except Exception as e:
                print(f"[MAIN][WARN] failed to restore state_norm: {e}")
        except Exception as e:
            print(f"[MAIN][WARN] failed to load checkpoint: {e}")

    # 计算 epoch 范围（用于显示/命名）
    start_ep = int(max(1, args.start_epoch))
    end_ep   = start_ep + int(max(1, args.epochs)) - 1
    total_for_print = end_ep

    # 构造“每个 epoch 一条命令”的列表
    video_cmds: List[List[str]] = []

    if args.dataset_inputs:
        # 数据集模式：自动从 --dataset-* 生成 2-pass 命令（含 --stat-in / --csv / --stat-out / --o）
        video_cmds = ds.build_cmds_from_dataset(args, cfg)
    else:
        # 兼容原用法：--videos 多条命令，每条内部用 |
        for vstr in args.videos:
            parts = [p for p in vstr.split("|") if p]
            if parts:
                video_cmds.append(parts)

    if not video_cmds:
        print("[MAIN][ERROR] 需要至少一种输入：要么给 --dataset-inputs，要么给 --videos")
        return

    # ===== epoch 循环（保持原风格：每个 epoch 跑 1 条命令，顺序轮转）=====
    # 说明：下方“铺平循环”的写法，确保 epoch 数量 > 命令数量时会从头继续轮转。
    repeat = (end_ep - start_ep + 1 + len(video_cmds) - 1) // len(video_cmds)
    flat_cmds = video_cmds * repeat

    for i, vcmd in enumerate(flat_cmds):
        ep_idx = start_ep + i
        if ep_idx > end_ep:
            break

        runner.set_epoch(ep_idx, total_for_print)
        print(f"\n[MAIN] ===== Epoch {ep_idx}/{total_for_print} | mode={cfg.mode} =====")
        print(f"[MAIN] Launch encoder args: {vcmd}")

        stop_evt = threading.Event()
        enc = launch_encoder(cfg, vcmd)
        mon_thr = start_monitor(enc, cfg, runner, stop_evt)

        try:
            runner.serve_loop(stop_evt)
        except KeyboardInterrupt:
            print("\n[MAIN] keyboard interrupt; stopping.")
            stop_evt.set()
        finally:
            if cfg.kill_encoder_on_exit and (enc.poll() is None):
                enc.kill()
            mon_thr.join(timeout=1.0)

        # 保存 ckpt（仅训练）
        if cfg.mode == "train":
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            ckpt_name = f"{args.ckpt_prefix}_e{ep_idx:04d}.pth"
            ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
            extra = {"state_norm": runner.sb.norm.state_dict()}
            runner.agent.save_checkpoint(ckpt_path, extra=extra)
            print(f"[MAIN] checkpoint saved: {ckpt_path} (with state_norm)")

        # 重置跨 epoch 的统计（保持你原有节奏）
        runner.mg_bits_tgt_total = 0.0
        runner._seen_mg_ids.clear()
        runner._gop_plan_bits.clear()
        runner._gop_init_rem.clear()
        runner._curr_gop_id = 0
        runner._last_frames_left_gop = None

    print("[MAIN] all epochs done. bye.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
以“视频为单位”的 epoch 训练/验证入口：
- 每个 epoch 启动一次编码器；
- RLRunner 在同一个进程中轮询 rq/fb，产出 qp；
- 支持断点续训 (--resume) 与自定义当前 epoch (--start-epoch)；
- 支持验证模式 (--mode val)：只出动作，不训练；但照常接收反馈并统计指标。
"""

import os, argparse, threading, glob
from typing import Optional
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    # 仍然在 main 中输入编码器命令行：每个元素内部用“|”分隔
    ap.add_argument("--videos", type=str, nargs="+", default=[
        "--input|E:/Git/qav1_ori/qav1/workspace/park_mobile_1920x1080_24.yuv|"
        "--input-res|1920x1080|"
        "--frames|0|"
        "--o|E:/Git/qav1_ori/qav1/workspace/park_mobile_1920x1080_24.ivf|"
        "--csv|E:/Git/qav1_ori/qav1/workspace/park_mobile_1920x1080_24_821.csv|"
        "--bitrate|2125|"
        "--rc-mode|1|"
        "--pass|2|"
        "--stat-in|E:/Git/qav1_ori/qav1/workspace/1pass.log|"
        "--stat-out|E:/Git/qav1_ori/qav1/workspace/2pass.log|"
        "--score-max|50.5|"
        "--score-avg|40.5|"
        "--score-min|38.5|"
        "--fps|30|"
        "--preset|0|"
        "--keyint|225|"
        "--bframes|15|"
        "--threads|1|"
        "--parallel-frames|1|"
        "--bitrate|2125"
    ])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--start-epoch", type=int, default=5, help="当前(起始)epoch编号，用于继续训练的显示与命名")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "val", "infer"])
    ap.add_argument("--encoder", type=str, default=r"E:\Git\qav1_ori\qav1\build\vs2022\x64\Release\qav1enc.exe")
    ap.add_argument("--resume", type=str, default=r"E:\python\DDPG\checkpoints\ckpt_e0005.pth", help="可选：指向 ckpt 文件或目录；从该 checkpoint 继续训练/验证")
    ap.add_argument("--ckpt-prefix", type=str, default="ckpt", help="保存检查点的前缀名")
    return ap.parse_args()


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

def main():
    args = parse_args()
    cfg = Config(rl_dir=args.rl_dir, mode=args.mode)
    if args.encoder:
        cfg.encoder_path = args.encoder
    if not os.path.exists(cfg.encoder_path):
        print(f"[ERROR] 编码器路径不存在: {cfg.encoder_path}")
        print("请检查 config.py 中的 encoder_path 配置或使用 --encoder 参数指定正确的路径")
        return

    # === Runner（RL agent + IO循环）
    runner = RLRunner(cfg)

    # === 断点恢复：既可给具体文件，也可给目录（取最新 *.pth/*.pt）
    ckpt_path = _latest_ckpt(args.resume)
    if ckpt_path:
        try:
            ckpt = runner.agent.load_checkpoint(ckpt_path)
            # 若 ckpt 中包含 state_norm，则恢复到 StateBuilder 的 RunningNorm
            try:
                sn = ckpt.get("state_norm", None)
                if sn is not None:
                    runner.sb.norm.load_state_dict(sn)
                    print(f"[MAIN] resumed from: {ckpt_path} (with state_norm)")
                else:
                    print(f"[MAIN] resumed from: {ckpt_path} (no state_norm in ckpt)")
            except Exception as e:
                print(f"[MAIN][WARN] failed to restore state_norm: {e}")
        except Exception as e:
            print(f"[MAIN][WARN] failed to load checkpoint: {e}")

    # === 计算 epoch 范围（用于显示/命名）
    start_ep = int(max(1, args.start_epoch))
    end_ep   = start_ep + int(max(1, args.epochs)) - 1
    total_for_print = end_ep

    # === 视频任务：每个元素是完整命令行，用 '|' 切分
    video_cmds = []
    for vstr in args.videos:
        parts = [p for p in vstr.split("|") if p]
        video_cmds.append(parts)
    if not video_cmds:
        print("[MAIN][ERROR] --videos 不能为空")
        return

    # === epoch 循环 ===
    for i, vcmd in enumerate(video_cmds * ((end_ep - start_ep + 1 + len(video_cmds) - 1)//len(video_cmds))):
        ep_idx = start_ep + i
        if ep_idx > end_ep:
            break

        runner.set_epoch(ep_idx, total_for_print)
        print(f"\n[MAIN] ===== Epoch {ep_idx}/{total_for_print} | mode={cfg.mode} =====")
        print(f"[MAIN] Launch encoder args: {vcmd}")

        # 启动监控 + 编码器
        stop_evt = threading.Event()
        enc = launch_encoder(cfg, vcmd)
        mon_thr  = start_monitor(enc, cfg, runner, stop_evt)

        try:
            runner.serve_loop(stop_evt)
        except KeyboardInterrupt:
            print("\n[MAIN] keyboard interrupt; stopping.")
            stop_evt.set()
        finally:
            if cfg.kill_encoder_on_exit and (enc.poll() is None):
                enc.kill()
            mon_thr.join(timeout=1.0)

        # 仅在训练模式下保存 checkpoint
        if cfg.mode == "train":
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            ckpt_name = f"{args.ckpt_prefix}_e{ep_idx:04d}.pth"
            ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
            # 打包 RunningNorm 的统计，一并写入同一个 ckpt 文件
            extra = {"state_norm": runner.sb.norm.state_dict()}
            runner.agent.save_checkpoint(ckpt_path, extra=extra)
            print(f"[MAIN] checkpoint saved: {ckpt_path} (with state_norm)")

        # 重置统计
        runner.mg_bits_tgt_total = 0.0
        runner._seen_mg_ids.clear()
        runner._gop_plan_bits.clear()
        runner._gop_init_rem.clear()
        runner._curr_gop_id = 0
        runner._last_frames_left_gop = None

    print("[MAIN] all epochs done. bye.")

if __name__ == "__main__":
    main()

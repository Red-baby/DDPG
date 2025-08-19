# -*- coding: utf-8 -*-
"""
以“视频为单位”的 epoch 训练：
- 每个 epoch 启动一次编码器；
- RLRunner 在同一个进程中轮询 rq/fb，产出 qp；
- epoch 结束后（编码器退出+目录静默）进入下一视频/下一epoch。
"""

import os, argparse, threading, time
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    ap.add_argument("--videos", type=str, nargs="+", default=[
        "--input|D:/Python/DDPG/park_mobile_1920x1080_24.yuv|"
        "--input-res|1920x1080|"
        "--frames|0|"
        "--o|D:/Python/DDPG/park_mobile_1920x1080_24.ivf|"
        "--csv|D:/Python/DDPG/park_mobile_1920x1080_24_819.csv|"
        "--bitrate|2125|"
        "--rc-mode|1|"
        "--pass|2|"
        "--stat-in|D:/Python/DDPG/1pass.log|"
        "--stat-out|D:/Python/DDPG/2pass.log|"
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
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    ap.add_argument("--encoder", type=str, default=r"D:/Python/DDPG/qav1enc.exe")
    ap.add_argument("--resume", type=str, default="", help="可选：加载某个 checkpoint 继续训练/推理")
    return ap.parse_args()

def parse_video_args(arg_str: str) -> list[str]:
    # 把 "-i|a.yuv|-o|a.ivf" 解析为 ["-i","a.yuv","-o","a.ivf"]
    parts = [p for p in arg_str.split("|") if p != ""]
    return parts

def main():
    args = parse_args()
    cfg = Config(rl_dir=args.rl_dir, mode=args.mode)
    if args.encoder:
        cfg.encoder_path = args.encoder
    if not os.path.exists(cfg.encoder_path):
        print(f"[ERROR] 编码器路径不存在: {cfg.encoder_path}")
        print("请检查 config.py 中的 encoder_path 配置或使用 --encoder 参数指定正确的路径")
        return 1

    os.makedirs(cfg.rl_dir, exist_ok=True)
    runner = RLRunner(cfg)

    # 可选恢复
    if args.resume:
        try:
            runner.agent.load_checkpoint(args.resume)
            print(f"[MAIN] resumed from: {args.resume}")
        except Exception as e:
            print(f"[MAIN][WARN] failed to load checkpoint: {e}")

    for ep in range(args.epochs):
        runner.set_epoch(ep + 1, args.epochs)
        print(f"\n[MAIN] ===== Epoch {ep+1}/{args.epochs} =====")
        for v_idx, v in enumerate(args.videos):
            video_args = parse_video_args(v)
            print(f"[MAIN] Launch encoder for video {v_idx+1}/{len(args.videos)}: {video_args}")

            # 启动编码器
            enc = launch_encoder(cfg, video_args)
            stop_evt = threading.Event()
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
        # === Epoch 结束统计 ===
        # === Epoch 结束统计（可选打印） ===
        if getattr(cfg, "print_epoch_gop_stats", False):
            plan_total = sum(runner._gop_plan_bits.values())
            initrem_total = sum(runner._gop_init_rem.values())
            print(f"[STATS] Epoch {ep + 1} 结束：")
            print(f"  - miniGOP 计划预算总和 = {runner.mg_bits_tgt_total:.2f}")
            print(f"  - GOP 计划预算总和     = {plan_total:.2f}  (sum of mg_bits_tgt over mg_id)")
            print(f"  - GOP 初始剩余总和     = {initrem_total:.2f}  (first snapshot of gop_bits_rem)")
            for gid in sorted(set(list(runner._gop_plan_bits.keys()) + list(runner._gop_init_rem.keys()))):
                plan = runner._gop_plan_bits.get(gid, 0.0)
                initr = runner._gop_init_rem.get(gid, 0.0)
                print(f"    GOP #{gid}: plan={plan:.2f}, init_rem={initr:.2f}")

        # 保存 checkpoint（每 epoch）
        if cfg.save_every_epoch:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ep{ep+1:03d}.pth")
            runner.agent.save_checkpoint(ckpt_path)
            print(f"[MAIN] checkpoint saved: {ckpt_path}")

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
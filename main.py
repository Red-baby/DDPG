# -*- coding: utf-8 -*-
import os, argparse, threading
from typing import List
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor
from dataset import add_dataset_args, build_cmds_from_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--mode", type=str, default="train", choices=["train","val","infer"])
    ap.add_argument("--encoder", type=str, default=Config.encoder_path)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt-prefix", type=str, default="ckpt")

    # 切换：单视频/数据集
    ap.add_argument("--use-dataset", action="store_true", help="启用数据集模式（从 --dataset-inputs 自动发现 YUV）")

    # 单视频命令模式（每条内部用 | 分隔）
    ap.add_argument("--videos", type=str, nargs="+", default=[
        "--input|E:/Git/qav1/workspace/park_mobile_1920x1080_24.yuv|--input-res|1920x1080|--frames|0|"
        "--o|E:/out/demo.ivf|--csv|E:/out/demo.csv|--bitrate|2125|--rc-mode|1|--pass|2|"
        "--stat-in|./pass1.log|--stat-out|E:/Git/qav1/workspace/demo_pass2.log|"
        "--score-max|50.5|--score-avg|40.5|--score-min|38.5|--fps|24|--preset|1|"
        "--keyint|225|--bframes|15|--threads|1|--parallel-frames|1"
    ])

    # 注入数据集相关参数（不使用 manifest）
    add_dataset_args(ap)
    return ap.parse_args()

def _split_cmd_bar(cmd_str: str) -> List[str]:
    # 把 " --k|v|--k2|v2 " 拆成 argv
    return [p for p in cmd_str.split("|") if p]

def _find_arg(args: List[str], key: str, default: str = "") -> str:
    try:
        i = args.index(key)
        return args[i+1] if i+1 < len(args) else default
    except ValueError:
        return default

def _infer_twopass_from_stat_in(stat_in: str) -> str:
    if not stat_in:
        return ""
    # 只把 pass1 替换成 pass2；路径其它部分不动
    return stat_in.replace("pass1", "pass2")

def _run_one_video(runner: RLRunner, cfg: Config, argv: List[str], epoch_id: int, epoch_total: int):
    # 自动推导 2-pass 基线
    stat_in = _find_arg(argv, "--stat-in", "")
    tp_path = _infer_twopass_from_stat_in(stat_in)
    if tp_path and not os.path.exists(tp_path):
        print(f"[MAIN][WARN] 2-pass baseline not found: {tp_path}")

    # 继承 FPS（用于统计 kbps）
    fps = _find_arg(argv, "--fps", "")
    if fps:
        try: cfg.fps = int(fps)
        except: pass

    # 把基线路径挂到 cfg（io_runner/reward 内部如果需要会从 cfg 读取）
    runner.cfg.twopass_log_path = tp_path

    # 设置 epoch（不改变其它回放/计数器逻辑）
    runner.set_epoch(idx=epoch_id, total=epoch_total, twopass_log_path=tp_path)

    # 启动编码器并进入服务循环
    enc = launch_encoder(cfg, argv)
    stop_evt = threading.Event()
    _ = start_monitor(enc, cfg, runner, stop_evt)
    runner.serve_loop(stop_evt)

def main():
    args = parse_args()
    cfg = Config(rl_dir=args.rl_dir, mode=args.mode)
    if args.encoder:
        cfg.encoder_path = args.encoder

    runner = RLRunner(cfg)

    start_ep = int(max(1, args.start_epoch))
    end_ep   = start_ep + int(max(1, args.epochs)) - 1

    if bool(args.use_dataset):
        # ===== 数据集模式：从 --dataset-inputs 构建所有 2-pass 命令 =====
        cmds = build_cmds_from_dataset(args, cfg)  # list[list[str]]
        if not cmds:
            print("[MAIN] no dataset commands built; exit.")
            return

        epoch_total = (end_ep - start_ep + 1) * max(1, len(cmds))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            # 简单按原顺序；如需打乱可在这里 random.shuffle(cmds.copy())
            for cmd_argv in cmds:
                _run_one_video(runner, cfg, cmd_argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
        print("[MAIN] dataset training finished.")
    else:
        # ===== 单视频命令模式（支持 --epochs）=====
        epoch_total = (end_ep - start_ep + 1) * max(1, len(args.videos))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            # 如需每个 epoch 打乱顺序，可以：cmds = args.videos.copy(); random.shuffle(cmds)
            for cmd_bar in args.videos:
                argv = _split_cmd_bar(cmd_bar)
                _run_one_video(runner, cfg, argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
        print("[MAIN] single-video list finished.")

if __name__ == "__main__":
    main()

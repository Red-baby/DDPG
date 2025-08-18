# -*- coding: utf-8 -*-
"""
启动/监控编码器进程（每个 epoch 一次），与 RL 通过目录文件同步。
这里不强依赖 POC/DOC，只要编码器与 RL 的 rq/fb/qp 文件前缀一致即可。
"""

import os, subprocess, threading, time, glob
from utils import now_ms

def launch_encoder(cfg, video_args: list[str]):
    """
    video_args: 针对当前视频的参数，如 ["-i", in_path, "-o", out_path, ...]
    返回 (Popen对象, 监控线程 stop_event)
    """
    cmd = [cfg.encoder_path] + video_args
    env = os.environ.copy()
    env["QAV1_RL_DIR"] = cfg.rl_dir   # 让编码器写入/读取该目录

    enc = subprocess.Popen(cmd, env=env)
    return enc

def start_monitor(enc, cfg, runner, stop_evt):
    """
    监控线程：等待编码器退出；随后等待目录静默/pending清空，通知RL结束当前epoch。
    """
    def monitor_thread():
        # 等编码器退出
        while enc.poll() is None:
            time.sleep(0.2)
        # 尾包处理：目录无 rq/fb 且 pending 清空 or 超时
        deadline = time.time() + 5.0
        while time.time() < deadline:
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "doc_*.rq.json")) or
                          glob.glob(os.path.join(cfg.rl_dir, "frame_*.rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "doc_*.fb.json")) or
                          glob.glob(os.path.join(cfg.rl_dir, "frame_*.fb.json")))
            in_flight = len(runner.pending) > 0
            if not has_rq and not has_fb and not in_flight:
                break
            time.sleep(0.1)
        stop_evt.set()

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return t

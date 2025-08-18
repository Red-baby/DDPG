# -*- coding: utf-8 -*-
"""
启动/监控编码器进程（每个 epoch 一次），与 RL 通过目录文件同步。
这里不强依赖 POC/DOC，只要编码器与 RL 的 rq/fb/qp 文件前缀一致即可。
"""

import os, subprocess, threading, time, glob, datetime, sys
from utils import now_ms

def _win_no_window_flags(cfg):
    # Windows 隐藏控制台窗口
    if os.name != "nt":
        return {}
    flags = {}
    if getattr(cfg, "hide_encoder_console_window", False):
        flags["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        # 兼容某些平台（如老 Python）：
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            flags["startupinfo"] = si
        except Exception:
            pass
    return flags

def _maybe_open_log_files(cfg):
    """根据配置返回 (stdout, stderr) 文件对象或 None。"""
    if not getattr(cfg, "encoder_log_to_file", False):
        # 不落地日志
        return (subprocess.PIPE if cfg.show_encoder_output else subprocess.DEVNULL,
                subprocess.STDOUT if cfg.show_encoder_output else subprocess.DEVNULL)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"encoder_{ts}.log"
    os.makedirs(cfg.encoder_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.encoder_log_dir, log_name)
    # 追加模式，合并 stdout/stderr
    fp = open(log_path, "a", buffering=1, encoding="utf-8", errors="ignore")
    return (fp, fp)

def launch_encoder(cfg, video_args: list[str]):
    """
    video_args: 针对当前视频的参数，如 ["-i", in_path, "-o", out_path, ...]
    返回 Popen对象
    """
    cmd = [cfg.encoder_path] + video_args
    env = os.environ.copy()
    env["QAV1_RL_DIR"] = cfg.rl_dir   # 让编码器写入/读取该目录

    stdout_fd, stderr_fd = _maybe_open_log_files(cfg)

    popen_kwargs = dict(env=env, stdout=stdout_fd, stderr=stderr_fd)
    popen_kwargs.update(_win_no_window_flags(cfg))

    if not getattr(cfg, "show_encoder_output", False) and stdout_fd is subprocess.PIPE:
        # 如果想在控制台实时看输出，可在主进程读取；默认不需要
        pass

    enc = subprocess.Popen(cmd, **popen_kwargs)
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

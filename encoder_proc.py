# -*- coding: utf-8 -*-
import os, subprocess, threading, time, glob, datetime
from utils import now_ms

def _win_no_window_flags(cfg):
    if os.name != "nt": return {}
    flags = {}
    if getattr(cfg, "hide_encoder_console_window", False):
        flags["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        import msvcrt
        flags["close_fds"] = False
    except Exception:
        pass
    return flags

def _maybe_open_log_files(cfg):
    if not getattr(cfg, "encoder_log_to_file", False):
        return (subprocess.PIPE if cfg.show_encoder_output else subprocess.DEVNULL,
                subprocess.STDOUT if cfg.show_encoder_output else subprocess.DEVNULL)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"encoder_{ts}.log"
    os.makedirs(cfg.encoder_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.encoder_log_dir, log_name)
    fp = open(log_path, "a", buffering=1, encoding="utf-8", errors="ignore")
    return (fp, fp)

def launch_encoder(cfg, video_args: list[str]):
    cmd = [cfg.encoder_path] + video_args
    env = os.environ.copy()
    env["QAV1_RL_DIR"] = cfg.rl_dir
    stdout_fd, stderr_fd = _maybe_open_log_files(cfg)
    popen_kwargs = dict(stdout=stdout_fd, stderr=stderr_fd, env=env, cwd=os.getcwd(), **_win_no_window_flags(cfg))
    return subprocess.Popen(cmd, **popen_kwargs)

def start_monitor(enc, cfg, runner, stop_evt):
    def monitor_thread():
        while enc.poll() is None:
            time.sleep(0.2)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "frame_*.rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "frame_*.fb.json")))
            in_flight = len(runner.pending) > 0
            if not has_rq and not has_fb and not in_flight:
                break
            time.sleep(0.1)
        stop_evt.set()
    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return t

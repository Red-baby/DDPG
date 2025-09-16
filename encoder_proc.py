# -*- coding: utf-8 -*-
import os, sys, glob, time, threading, subprocess, datetime
from utils import now_ms

# ---------- Windows: 隐藏子进程窗口 ----------
def _win_no_window_flags(cfg):
    if os.name != "nt":
        return {}
    flags = {}
    if bool(getattr(cfg, "hide_encoder_console_window", False)):
        flags["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    # 某些旧 Python on Windows 需要 close_fds=False 才能重定向句柄
    try:
        import msvcrt  # noqa
        flags["close_fds"] = False
    except Exception:
        pass
    return flags

# ---------- 日志文件路径 ----------
def _build_log_path(cfg):
    """
    例：logs/encoder/encoder_2025-09-16_14-33-21_ep11.log
    如无 epoch 信息则去掉 _epXX
    """
    base_dir = str(getattr(cfg, "encoder_log_dir", "./logs/encoder"))
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ep = getattr(cfg, "curr_epoch", None) or getattr(cfg, "epoch_idx", None)
    tag = f"_ep{int(ep)}" if ep is not None else ""
    return os.path.join(base_dir, f"encoder_{ts}{tag}.log")

# ---------- 启动编码器 + Tee 到文件 ----------
def launch_encoder(cfg, video_args: list[str]):
    """
    启动编码器；返回 subprocess.Popen 对象。
    - 若 cfg.encoder_log_to_file=True：将 stdout/stderr Tee 到日志文件
    - 若 cfg.show_encoder_output=True：同时在控制台打印
    """
    cmd = [cfg.encoder_path] + list(video_args)

    # 传给编码器的环境变量（让它知道 rl 目录）
    env = os.environ.copy()
    env["QAV1_RL_DIR"] = str(getattr(cfg, "rl_dir", "./rl_io"))

    # 统一从 PIPE 读，这样我们可以 Tee
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,  # 按行读取
        bufsize=1,                # 行缓冲
        env=env,
        **_win_no_window_flags(cfg)
    )

    log_path = None
    log_fp = None
    to_file = bool(getattr(cfg, "encoder_log_to_file", True))
    to_console = bool(getattr(cfg, "show_encoder_output", True))

    if to_file:
        log_path = _build_log_path(cfg)
        # 记在 cfg 上，方便别处引用
        try:
            cfg.last_encoder_log_path = log_path
        except Exception:
            pass
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

        # 写入启动信息
        if log_fp:
            log_fp.write(f"[ENC] launch: {' '.join(cmd)}\n")
            log_fp.write(f"[ENC] rl_dir={env.get('QAV1_RL_DIR','')}\n")

    # Tee 线程：把子进程输出按行写到文件/控制台
    def _tee():
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                if to_console:
                    # 打印到控制台，避免重复换行
                    try:
                        sys.stdout.write(line)
                    except Exception:
                        pass
                if log_fp:
                    try:
                        log_fp.write(line)
                    except Exception:
                        pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            if log_fp:
                try:
                    log_fp.flush()
                    log_fp.close()
                except Exception:
                    pass

    t = threading.Thread(target=_tee, daemon=True)
    t.start()

    print(f"[ENC] started | pid={proc.pid} | log={'on' if to_file else 'off'}"
          + (f" -> {log_path}" if to_file else ""))

    return proc

# ---------- 编码器退出后的目录清理/收尾 ----------
def start_monitor(enc: subprocess.Popen, cfg, runner, stop_evt: threading.Event):
    """
    监视编码器进程；它退出后等 RL 侧把未处理完的 rq/fb 清空，再发 stop_evt 结束 serve_loop。
    """
    def monitor_thread():
        # 等编码器退出
        while enc.poll() is None:
            time.sleep(0.2)

        # 等待 RL 侧处理完 in-flight 的请求（给 5 秒缓冲）
        deadline = time.time() + 5.0
        while time.time() < deadline:
            # 这里按你的 RL 命名习惯，两种前缀都检查一下
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "mg_*.rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "mg_*.fb.json")))
            in_flight = len(getattr(runner, "pending", {})) > 0
            if not has_rq and not has_fb and not in_flight:
                break
            time.sleep(0.1)

        stop_evt.set()
        print("[ENC] finished; stop signal sent to RL loop.")

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return t

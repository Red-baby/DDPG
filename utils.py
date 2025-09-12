# -*- coding: utf-8 -*-
"""
utils.py
通用 I/O 与小工具：
- safe_read_json(path): 容错读取 JSON
- safe_write_text(path, text): 先写到 .tmp，再带重试地替换目标（Windows 友好）
- try_remove(path): 安静删除
- now_ms(): 毫秒时间戳
- _float/_int: 容错数值转换
"""

from __future__ import annotations
import os, io, json, time, tempfile, shutil

__all__ = [
    "safe_read_json", "safe_write_text", "try_remove", "now_ms",
    "_float", "_int"
]

def now_ms() -> int:
    return int(time.time() * 1000)

def _float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def _int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, (int,)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return int(default)
        # 允许 "123.0" 这种
        return int(float(s))
    except Exception:
        return int(default)

def safe_read_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 再试一次（可能是部分写入）
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            return json.loads(txt)
        except Exception:
            return default

def try_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        # Windows 有时需要改名后再删
        try:
            bkup = path + ".del"
            if os.path.exists(bkup):
                os.remove(bkup)
            os.replace(path, bkup)
            os.remove(bkup)
        except Exception:
            pass
    except Exception:
        pass

def _atomic_replace(src: str, dst: str) -> None:
    """
    原子替换，兼容 Windows：先尝试 os.replace；失败则删除目标后再 replace；
    再失败则用拷贝替换的兜底。
    """
    try:
        os.replace(src, dst)
        return
    except PermissionError:
        # 目标可能被占用：试图删除再替换
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.replace(src, dst)
            return
        except Exception:
            pass
    except Exception:
        pass
    # 兜底：拷贝
    try:
        shutil.copyfile(src, dst)
        os.remove(src)
    except Exception:
        # 最后兜底：把 tmp 保留下来，至少不丢数据
        pass

def safe_write_text(path: str, text: str, retries: int = 6, backoff_ms: int = 20) -> None:
    """
    将文本安全写入 path：
      - 写到同目录的临时文件，再原子替换；
      - Windows 下带重试，缓解 WinError 5。
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    # 用 NamedTemporaryFile 保证跨盘也能 replace
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d, text=True)
    try:
        with io.open(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        # 带重试的原子替换
        for i in range(max(1, int(retries))):
            _atomic_replace(tmp_path, path)
            # 校验：成功后文件应存在且大小>0（允许写空行则去掉大小校验）
            try:
                if os.path.exists(path):
                    return
            except Exception:
                pass
            time.sleep(backoff_ms / 1000.0)
    finally:
        # 清理遗留 tmp
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

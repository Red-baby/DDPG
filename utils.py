# -*- coding: utf-8 -*-
import os, json, time

def now_ms() -> int:
    return int(time.time() * 1000)

def try_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

def safe_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def safe_read_json(path: str, retries: int = 50, sleep_ms: int = 2):
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            time.sleep(sleep_ms / 1000.0)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

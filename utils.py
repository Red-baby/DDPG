# -*- coding: utf-8 -*-
import os, json, time

def now_ms() -> int:
    return int(time.time() * 1000)

def try_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

def safe_write_text(path: str, text: str, retries: int = 64, backoff_ms: int = 2) -> None:
    """
    原子写入（.tmp -> 目标）+ Windows 共享冲突友好：
    1) 先写到 path+".tmp"，flush + fsync；
    2) 尝试 os.replace(tmp, path)；
       - 若目标被其他进程占用（WinError 5），指数退避重试；
    3) 多次失败后，降级为“直接写目标文件”（同样带重试与 fsync）；
    4) 最后清理 .tmp（忽略错误）。
    """
    import os, time, io

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"

    # 1) 写临时文件
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # 某些文件系统可能不支持 fsync，到此也没关系

    # 2) 尝试原子替换（优先方案）
    last_err = None
    for i in range(max(1, int(retries))):
        try:
            os.replace(tmp, path)  # 成功即返回
            return
        except PermissionError as e:
            # 目标可能被占用（WinError 5），退避后重试
            last_err = e
            # 指数退避，但上限 256ms，避免阻塞太久
            sleep_ms = min(backoff_ms * (2 ** i), 256)
            time.sleep(sleep_ms / 1000.0)
        except OSError as e:
            # 其他 OSError：短暂等待再试
            last_err = e
            time.sleep(backoff_ms / 1000.0)

    # 3) 降级为“直接写目标文件”（同样带重试）
    for i in range(max(1, int(retries))):
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f2:
                f2.write(text)
                f2.flush()
                try:
                    os.fsync(f2.fileno())
                except OSError:
                    pass
            # 写成功就退出
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return
        except PermissionError as e:
            last_err = e
            sleep_ms = min(backoff_ms * (2 ** i), 256)
            time.sleep(sleep_ms / 1000.0)
        except OSError as e:
            last_err = e
            time.sleep(backoff_ms / 1000.0)

    # 4) 仍失败：尽量清理 tmp，并抛出更友好的错误
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass
    raise PermissionError(
        f"safe_write_text: failed to write '{path}' after retries; "
        f"target may be locked by another process. last_err={last_err}"
    )

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

# -*- coding: utf-8 -*-
"""
环境I/O与训练主循环（DOC优先；支持兼容frame_前缀）：
- 轮询 rl_dir，读取编码器写出的 rq.json / fb.json；
- 构建状态 → 选择动作 → 写 qp.txt（沿用 rq 的前缀）；
- 读取反馈 → 计算奖励 → 存回放 → 训练一步；
- 以 mini-GOP 为 episode（frames_left_mg==1 判 done）。
"""

import os, glob, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict
from utils import safe_read_json, safe_write_text, try_remove, now_ms, _float, _int
from state import StateBuilder, STATE_FIELDS
from agent import DDPG
from reward import compute_reward

def _scan_rq_files(rl_dir: str):
    paths = sorted(glob.glob(os.path.join(rl_dir, "doc_*.rq.json")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(rl_dir, "frame_*.rq.json")))
    return paths

def _scan_fb_files(rl_dir: str):
    paths = sorted(glob.glob(os.path.join(rl_dir, "doc_*.fb.json")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(rl_dir, "frame_*.fb.json")))
    return paths

@dataclass
class Pending:
    state: torch.Tensor
    meta: dict
    action_a01: float
    qp_used: int
    next_state: Optional[torch.Tensor] = None
    done: bool = False

class RLRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sb = StateBuilder(cfg)
        self.agent = DDPG(state_dim=len(STATE_FIELDS), cfg=cfg)
        self.pending: Dict[int, Pending] = {}
        self.last_doc_in_mg: Optional[int] = None

        # mini-GOP 与全局统计（用于强惩罚）
        self._seen_mg_ids = set()
        self._curr_gop_id = 0
        self._last_frames_left_gop = None
        self.mg_bits_tgt_total = 0.0
        self._gop_plan_bits = {}
        self._gop_init_rem = {}

        self.mg_acc: Dict[tuple, dict] = {}  # (gop_id, mg_index) -> {"bits":..,"frames":..,"psnr":..}
        self.global_bits_sum = 0.0
        self.global_frames = 0

        # 打印相关
        self.epoch_idx = 0
        self.epoch_total = 0
        self.loss_ema_a = None
        self.loss_ema_c = None

    def set_epoch(self, idx: int, total: int):
        self.epoch_idx = idx
        self.epoch_total = total

    def _update_loss_ema(self, ret):
        if ret is None:
            return
        lc, la = ret
        b = self.cfg.loss_ema_beta
        if self.loss_ema_c is None:
            self.loss_ema_c = lc
            self.loss_ema_a = la
        else:
            self.loss_ema_c = (1 - b) * self.loss_ema_c + b * lc
            self.loss_ema_a = (1 - b) * self.loss_ema_a + b * la

    def serve_loop(self, stop_evt):
        print(f"[RL] watching: {self.cfg.rl_dir} | mode={self.cfg.mode}")
        last_print = now_ms()
        while not stop_evt.is_set():
            progressed = False
            progressed |= self.handle_requests()
            progressed |= self.handle_feedbacks()
            if self.cfg.mode == "train":
                self._update_loss_ema(self.agent.train_step())

            now = now_ms()
            if now - last_print > int(self.cfg.print_every_sec * 1000):
                if self.loss_ema_a is not None and self.loss_ema_c is not None:
                    loss_str = f" | loss_a={self.loss_ema_a:.4f} loss_c={self.loss_ema_c:.4f}"
                else:
                    loss_str = ""
                print(f"[RL] epoch {self.epoch_idx}/{self.epoch_total} | "
                      f"steps env/train: {self.agent.total_env_steps}/{self.agent.total_train_steps} | "
                      f"replay={len(self.agent.buf)}{loss_str}")
                last_print = now

            if not progressed:
                time.sleep(0.003)

        # 尾声：尽量处理残余反馈
        for _ in range(200):
            any_left = self.handle_feedbacks()
            if not any_left:
                break
            time.sleep(0.003)
        print("[RL] serve loop exit.")

    def handle_requests(self) -> bool:
        rq_paths = _scan_rq_files(self.cfg.rl_dir)
        anyp = False
        for rq_path in rq_paths:
            try:
                rq = safe_read_json(rq_path)
            except Exception as e:
                print(f"[RL][WARN] bad rq json {rq_path}: {e}")
                try_remove(rq_path); continue

            s, meta = self.sb.build(rq)

            # GOP 边界检测
            flg = _int(rq.get("frames_left_gop", -1))
            if self._last_frames_left_gop is None:
                self._last_frames_left_gop = flg
            else:
                if flg > self._last_frames_left_gop:
                    self._curr_gop_id += 1
                self._last_frames_left_gop = flg

            # 记录当前 gop_id 进入 meta，便于 mini-GOP 统计聚合
            meta["gop_id"] = self._curr_gop_id

            # mini-GOP 预算累计（用于统计）
            mg_id = _int(rq.get("mg_id", -1))
            if mg_id >= 0 and mg_id not in self._seen_mg_ids:
                self._seen_mg_ids.add(mg_id)
                mg_bits_tgt = _float(rq.get("mg_bits_tgt", 0.0))
                self.mg_bits_tgt_total += mg_bits_tgt
                self._gop_plan_bits[self._curr_gop_id] = self._gop_plan_bits.get(self._curr_gop_id, 0.0) + mg_bits_tgt
            if self._curr_gop_id not in self._gop_init_rem:
                self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))

            explore = (self.cfg.mode == "train")
            qp = self.agent.select_action(s, explore=explore)

            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            safe_write_text(qp_path, f"{qp}\n")
            try_remove(rq_path)

            denom = max(1, (self.cfg.qp_max - self.cfg.qp_min))
            a01 = float((qp - self.cfg.qp_min) / denom)

            doc = _int(meta.get("doc", -1))
            self.pending[doc] = Pending(
                state=s, meta=meta, action_a01=a01, qp_used=qp, next_state=None, done=False
            )

            if self.last_doc_in_mg is not None and self.last_doc_in_mg in self.pending:
                prev = self.pending[self.last_doc_in_mg]
                if prev.next_state is None:
                    prev.next_state = s  # s_{t+1}

            self.last_doc_in_mg = doc
            self.agent.total_env_steps += 1
            anyp = True
        return anyp

    def handle_feedbacks(self) -> bool:
        fb_paths = _scan_fb_files(self.cfg.rl_dir)
        anyp = False
        for fb_path in fb_paths:
            try:
                fb = safe_read_json(fb_path)
            except Exception as e:
                print(f"[RL][WARN] bad fb json {fb_path}: {e}")
                try_remove(fb_path); continue

            doc = _int(fb.get("doc", -1))
            if doc not in self.pending:
                print(f"[RL][WARN] feedback for unknown DOC={doc}")
                try_remove(fb_path); continue

            pend = self.pending[doc]

            # 观测 PSNR（与奖励一致）
            psnr_y = _float(fb.get("psnr_y", 0.0))
            if self.cfg.psnr_mode == "yuv":
                pu = _float(fb.get("psnr_u", 0.0)); pv = _float(fb.get("psnr_v", 0.0))
                psnr_obs = (6*psnr_y + pu + pv) / 8.0 if (psnr_y>0 and pu>0 and pv>0) else psnr_y
            else:
                psnr_obs = psnr_y

            bits_obs = _float(fb.get("bits", 0.0))
            self.global_bits_sum += bits_obs
            self.global_frames += 1

            # mini-GOP 聚合（用于强惩罚）
            gop_id = int(pend.meta.get("gop_id", 0))
            mg_idx = int(pend.meta.get("mg_index", 0))
            key = (gop_id, mg_idx)
            acc = self.mg_acc.get(key)
            if acc is None:
                acc = {"bits": 0.0, "frames": 0, "psnr": 0.0}
                self.mg_acc[key] = acc
            acc["bits"] += bits_obs
            acc["psnr"] += psnr_obs
            acc["frames"] += 1

            # 是否 mini-GOP 结束
            done = False
            flm = pend.meta.get("frames_left_mg", None)
            if (flm is not None) and int(flm) == 1:
                done = True

            mg_info = None
            if done:
                frames = max(1, acc["frames"])
                mg_info = {
                    "done": True,
                    "avg_bits": acc["bits"] / frames,
                    "avg_psnr": acc["psnr"] / frames,
                    "frames": frames,
                }
                # 用完就释放，避免积累
                self.mg_acc.pop(key, None)

            r = compute_reward(
                self.cfg,
                fb,
                rq_meta=pend.meta,
                prev_psnr_cached=float(self.sb.prev_psnr),
                mg_info=mg_info,
                global_info={"avg_bits": self.global_bits_sum / max(1, self.global_frames),
                             "frames": self.global_frames}
            )

            if pend.next_state is None:
                pend.next_state = pend.state.clone() if not done else torch.zeros_like(pend.state)

            self.agent.buf.push(
                pend.state.numpy(),
                np.array([[pend.action_a01]], dtype=np.float32),
                np.array([[r]], dtype=np.float32),
                pend.next_state.numpy(),
                np.array([[1.0 if done else 0.0]], dtype=np.float32)
            )

            # 更新上一帧真实观测（供平滑项使用）
            self.sb.update_prev_meas(bits=bits_obs, psnr=float(psnr_obs), qp=int(pend.qp_used))

            try_remove(fb_path)
            if done:
                self.last_doc_in_mg = None
                # 清理过旧 pending
                ks = sorted(list(self.pending.keys()))
                for k in ks[:-2]:
                    self.pending.pop(k, None)

            anyp = True
        return anyp

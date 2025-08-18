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
    # 兼容两种命名：doc_*.rq.json 优先，其次 frame_*.rq.json
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
        # ==== 预算统计 ====
        self.mg_bits_tgt_total = 0.0
        self._seen_mg_ids = set()  # 只对每个 mg_id 计一次
        self._gop_plan_bits = {}  # gop_id -> 累计计划预算（sum of mg_bits_tgt）
        self._gop_init_rem = {}  # gop_id -> 第一次看到的 gop_bits_rem 快照
        self._curr_gop_id = 0
        self._last_frames_left_gop = None

    def serve_loop(self, stop_evt):
        print(f"[RL] watching: {self.cfg.rl_dir} | mode={self.cfg.mode}")
        last_print = now_ms()
        while not stop_evt.is_set():
            progressed = False
            progressed |= self.handle_requests()
            progressed |= self.handle_feedbacks()
            if self.cfg.mode == "train":
                self.agent.train_step()

            now = now_ms()
            if now - last_print > int(self.cfg.print_every_sec * 1000):
                print(f"[RL] steps env/train: {self.agent.total_env_steps}/{self.agent.total_train_steps} | "
                      f"replay={len(self.agent.buf)}")
                last_print = now

            if not progressed:
                time.sleep(0.003)

        # 退出前，尽量处理残余反馈
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
            # 主键用 doc；若缺失则退化用 poc；再不行从文件名解析
            doc = _int(meta.get("doc", -1))
            if doc < 0:
                doc = _int(rq.get("doc", -1))
            if doc < 0:
                # 从文件名提取数字
                base = os.path.basename(rq_path)
                doc = _int(''.join([c for c in base if c.isdigit()]), -1)
            if doc < 0:
                print(f"[RL][WARN] rq without doc: {rq_path}")
                try_remove(rq_path); continue
            # ==== GOP 边界检测（frames_left_gop 回跳/增大 -> 新 GOP）====
            flg = _int(rq.get("frames_left_gop", -1))
            if self._last_frames_left_gop is None:
                self._last_frames_left_gop = flg
                # 首次进入：记录 init_rem 一次
                if flg >= 0 and 0 not in self._gop_init_rem:
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
            else:
                if flg > self._last_frames_left_gop:
                    # 进入新 GOP
                    self._curr_gop_id += 1
                    # 为新 GOP 记录一次 gop_bits_rem 初始快照
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
                self._last_frames_left_gop = flg

            # ==== mini-GOP 只记一次：依赖 mg_id ====
            mg_id = _int(rq.get("mg_id", -1))
            if mg_id >= 0 and mg_id not in self._seen_mg_ids:
                self._seen_mg_ids.add(mg_id)
                mg_bits_tgt = _float(rq.get("mg_bits_tgt", 0.0))
                self.mg_bits_tgt_total += mg_bits_tgt

                # 同时把本 mini-GOP 的目标预算计入“当前 GOP 的计划预算”
                self._gop_plan_bits[self._curr_gop_id] = self._gop_plan_bits.get(self._curr_gop_id, 0.0) + mg_bits_tgt

            explore = (self.cfg.mode == "train")
            qp = self.agent.select_action(s, explore=explore)

            # 写动作：沿用 rq 的前缀，保证与编码器匹配
            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            safe_write_text(qp_path, f"{qp}\n")
            try_remove(rq_path)

            denom = max(1, (self.cfg.qp_max - self.cfg.qp_min))
            a01 = float((qp - self.cfg.qp_min) / denom)

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

            # 用 doc 匹配 pending
            doc = _int(fb.get("doc", -1))
            if doc not in self.pending:
                print(f"[RL][WARN] feedback for unknown DOC={doc}")
                try_remove(fb_path); continue

            pend = self.pending[doc]

            # 奖励：传入上一帧 PSNR 缓存
            r = compute_reward(self.cfg, fb, rq_meta=pend.meta, prev_psnr_cached=float(self.sb.prev_psnr))

            # episode 截止（mini-GOP末尾：frames_left_mg==1）
            done = False
            flm = pend.meta.get("frames_left_mg", None)
            if (flm is not None) and int(flm) == 1:
                done = True

            if pend.next_state is None:
                pend.next_state = pend.state.clone() if not done else torch.zeros_like(pend.state)

            self.agent.buf.push(
                pend.state.numpy(),
                np.array([[pend.action_a01]], dtype=np.float32),
                np.array([[r]], dtype=np.float32),
                pend.next_state.numpy(),
                np.array([[1.0 if done else 0.0]], dtype=np.float32)
            )

            # 更新 RL 端“上一帧真实观测”缓存（PSNR）
            # PSNR选择：按 cfg.psnr_mode 取Y或YUV融合
            psnr_y = _float(fb.get("psnr_y", 0.0))
            if self.cfg.psnr_mode == "yuv":
                pu = _float(fb.get("psnr_u", 0.0)); pv = _float(fb.get("psnr_v", 0.0))
                psnr_obs = (6*psnr_y + pu + pv) / 8.0 if (psnr_y>0 and pu>0 and pv>0) else psnr_y
            else:
                psnr_obs = psnr_y
            self.sb.update_prev_meas(bits=_float(fb.get("bits", 0.0)),
                                     psnr=float(psnr_obs),
                                     qp=int(pend.qp_used))

            try_remove(fb_path)
            if done:
                self.last_doc_in_mg = None
                # 清除过旧pending，防泄漏
                ks = sorted(list(self.pending.keys()))
                for k in ks[:-2]:
                    self.pending.pop(k, None)

            anyp = True
        return anyp

# -*- coding: utf-8 -*-
"""
环境I/O与训练主循环（DOC优先；兼容 frame_ 前缀）：
- 轮询 rl_dir，读取编码器写出的 rq.json / fb.json；
- 构建状态 → 选择动作 → 写 qp.txt（沿用 rq 的前缀）；
- 读取反馈 → 计算奖励 → 存回放 → 训练一步；
- 以 mini-GOP 为 episode（frames_left_mg==1 判 done）；
- 在 mini-GOP 结束时输出：真实比特和、预估 mini-GOP 预算（mg_bits_tgt）、平均 PSNR。

注意：
1) 保留/恢复了你之前的 GOP 初始剩余预算快照记录（_gop_init_rem），包括“首次进入”和“新 GOP”两个时机。
2) mini-GOP 统计按 (gop_id, mg_index) 聚合，不依赖帧到达的严格顺序。
"""

import csv, os, glob, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict
from utils import safe_read_json, safe_write_text, try_remove, now_ms, _float, _int
from state import StateBuilder, STATE_FIELDS
from agent import DDPG,TD3
from reward import compute_reward



def _scan_rq_files(rl_dir: str):

    paths = sorted(glob.glob(os.path.join(rl_dir, "frame_*.rq.json")))

    return paths


def _scan_fb_files(rl_dir: str):

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
    created_at_ms: int = 0


class RLRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sb = StateBuilder(cfg)
        algo = str(getattr(self.cfg, "algo", "ddpg")).lower()
        if algo == "td3":
            self.agent = TD3(state_dim=len(STATE_FIELDS), cfg=cfg)
        else:
            self.agent = DDPG(state_dim=len(STATE_FIELDS), cfg=cfg)
        self.pending: Dict[int, Pending] = {}
        self.last_doc_in_mg: Optional[int] = None

        self._rew_ema_q = 0.0  # reward balance: |失真项| 的 EMA
        self._rew_ema_b = 0.0  # reward balance: |码率项| 的 EMA

        # ==== GOP / mini-GOP 计划统计 ====
        self._seen_mg_ids = set()
        self._curr_gop_id = 0
        self._last_frames_left_gop = None
        self.mg_bits_tgt_total = 0.0
        self._gop_plan_bits = {}   # gop_id -> sum of mg_bits_tgt
        self._gop_init_rem = {}    # gop_id -> 初次看到的 gop_bits_rem 快照（原逻辑保留/恢复）

        # ==== mini-GOP 实际统计（打印用） ====
        # key = (gop_id, mg_index) -> {"bits":累计真实bit, "psnr":累计psnr, "frames":累计帧数, "budget":mg_bits_tgt}
        self.mg_stats: Dict[tuple, dict] = {}

        # 打印相关
        self.epoch_idx = 0
        self.epoch_total = 0
        # epoch 内累积器
        self._ep_loss_sum_a = 0.0
        self._ep_loss_sum_c = 0.0
        self._ep_updates = 0
        # epoch 内“帧级业务指标”累积器
        self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0
        self._ep_frames   = 0


        # csv 路径
        self._metrics_csv = getattr(self.cfg, "metrics_csv", "epoch_losses.csv")
        self._metrics_csv_inited = False
        self.loss_ema_a = None
        self.loss_ema_c = None

    def set_epoch(self, idx: int, total: int):
        # 在“进入新 epoch”前，先把上一轮的均值写入 CSV（epoch_idx>0 才写）
        if self.epoch_idx > 0 and (self._ep_updates > 0 or self._ep_frames > 0):
            self._write_epoch_metrics(self.epoch_idx)

        # 更新当前 epoch 标记
        self.epoch_idx = int(idx)
        self.epoch_total = int(total)

        # 清零本轮累积器
        self._ep_loss_sum_a = 0.0
        self._ep_loss_sum_c = 0.0
        self._ep_updates = 0

        self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0
        self._ep_frames = 0
        self._rew_ema_q = 0.0
        self._rew_ema_b = 0.0

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
            if self.cfg.mode == "train" and progressed:
                k = int(getattr(self.cfg, "train_steps_per_env_step", 1))
                for _ in range(max(1, k)):
                    ret = self.agent.train_step()
                    self._update_loss_ema(ret)
                    if ret is not None:
                        lc, la = ret
                        if lc is not None: self._ep_loss_sum_c += float(lc)
                        if la is not None: self._ep_loss_sum_a += float(la)
                        self._ep_updates += 1

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
                try_remove(rq_path)
                continue

            s, meta = self.sb.build(rq)

            # ==== GOP 边界检测（frames_left_gop 回跳/增大 -> 新 GOP）====
            flg = _int(rq.get("frames_left_gop", -1))
            if self._last_frames_left_gop is None:
                # 首次进入：记录 init_rem 一次（你的原始逻辑）  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if flg >= 0 and 0 not in self._gop_init_rem:
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
                self._last_frames_left_gop = flg
            else:
                if flg > self._last_frames_left_gop:
                    # 进入新 GOP
                    self._curr_gop_id += 1
                    # 为新 GOP 记录一次 gop_bits_rem 初始快照（你的原始逻辑） <<<<<<<<<<<<<<
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
                self._last_frames_left_gop = flg

            # 把 gop_id 记入 meta，便于 mini-GOP 聚合
            meta["gop_id"] = self._curr_gop_id

            # ==== mini-GOP 预算统计（仅首次见到该 mg_index 时）====
            mg_idx = _int(rq.get("mg_index", 0))
            key = (self._curr_gop_id, mg_idx)
            if key not in self.mg_stats:
                self.mg_stats[key] = {
                    "bits": 0.0, "psnr": 0.0, "frames": 0,
                    # 预估 mini-GOP 编码预算（来自 rq 的 mg_bits_tgt；若缺失则为 0）
                    "budget": _float(rq.get("mg_bits_tgt", 0.0))
                }

            # 仍保留原来“全局计划统计”（用于 epoch 总结/对齐你的原结构）
            mg_id = _int(rq.get("mg_id", -1))
            if mg_id >= 0 and mg_id not in self._seen_mg_ids:
                self._seen_mg_ids.add(mg_id)
                mg_bits_tgt = _float(rq.get("mg_bits_tgt", 0.0))
                self.mg_bits_tgt_total += mg_bits_tgt
                self._gop_plan_bits[self._curr_gop_id] = self._gop_plan_bits.get(self._curr_gop_id, 0.0) + mg_bits_tgt
            if self._curr_gop_id not in self._gop_init_rem:
                # 若上面“首次进入/新 GOP”没记上（极少数异常），这里兜底一次
                self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))

            # ==== 选动作，写回 qp ====
            explore = (self.cfg.mode == "train")
            qp = self.agent.select_action(s, explore=explore)

            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            safe_write_text(qp_path, f"{qp}\n")
            try_remove(rq_path)

            denom = max(1, (self.cfg.qp_max - self.cfg.qp_min))
            a01 = float((qp - self.cfg.qp_min) / denom)

            # 用 doc 作为 pending key
            doc = _int(meta.get("doc", -1))
            self.pending[doc] = Pending(
                state=s, meta=meta, action_a01=a01, qp_used=qp, next_state=None, done=False,
                created_at_ms=now_ms()  # <<< 新增
            )

            # 把当前 s 作为上一个样本的 s'（顺序前提）
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
                try_remove(fb_path)
                continue

            # 匹配 pending
            doc = _int(fb.get("doc", -1))
            if doc not in self.pending:
                print(f"[RL][WARN] feedback for unknown DOC={doc}")
                try_remove(fb_path)
                continue

            pend = self.pending[doc]

            # === 观测（与奖励一致）===
            psnr_y = _float(fb.get("psnr_y", 0.0))
            if self.cfg.psnr_mode == "yuv":
                pu = _float(fb.get("psnr_u", 0.0)); pv = _float(fb.get("psnr_v", 0.0))
                psnr_obs = (6*psnr_y + pu + pv) / 8.0 if (psnr_y > 0 and pu > 0 and pv > 0) else psnr_y
            else:
                psnr_obs = psnr_y

            bits_obs = _float(fb.get("bits", 0.0))

            # === 【新增】累计到“本 epoch 帧级指标” ===
            self._ep_bits_sum += float(bits_obs)
            self._ep_psnr_sum += float(psnr_obs)
            self._ep_frames += 1

            # === 累加到 mini-GOP 统计（按 gop_id + mg_index 聚合）===
            gop_id = int(pend.meta.get("gop_id", 0))
            mg_idx = int(pend.meta.get("mg_index", 0))
            key = (gop_id, mg_idx)
            st = self.mg_stats.get(key)
            if st is None:
                # 理论不会发生；兜底
                st = {"bits": 0.0, "psnr": 0.0, "frames": 0, "budget": float(pend.meta.get("mg_bits_tgt", 0.0))}
                self.mg_stats[key] = st
            st["bits"] += bits_obs
            st["psnr"] += psnr_obs
            st["frames"] += 1

            # === 是否 mini-GOP 结束（接受 0 或 1 都视为末帧）===
            flm_val = fb.get("frames_left_mg", pend.meta.get("frames_left_mg", None))
            done = (flm_val is not None) and (int(flm_val==0))
            frames_so_far = max(1, st["frames"])
            mg_ctx = {
                "frames_so_far": int(frames_so_far),
                "ema_abs_q": float(self._rew_ema_q),
                "ema_abs_b": float(self._rew_ema_b),
            }
            # === 计算奖励（用 prev_psnr 做平滑项）===
            r = compute_reward(
                self.cfg,
                fb,
                rq_meta=pend.meta,
                prev_psnr_cached=float(self.sb.prev_psnr),
                mg_ctx=mg_ctx
            )
            # 回收 EMA（compute_reward 内部会更新这两个键）
            self._rew_ema_q = float(mg_ctx.get("ema_abs_q", self._rew_ema_q))
            self._rew_ema_b = float(mg_ctx.get("ema_abs_b", self._rew_ema_b))

            # === 填 next_state / 终止 ===
            if pend.next_state is None:
                pend.next_state = pend.state.clone() if not done else torch.zeros_like(pend.state)

            self.agent.buf.push(
                pend.state.numpy(),
                np.array([[pend.action_a01]], dtype=np.float32),
                np.array([[r]], dtype=np.float32),
                pend.next_state.numpy(),
                np.array([[1.0 if done else 0.0]], dtype=np.float32)
            )

            # === 更新 RL 端上一帧观测缓存 ===
            self.sb.update_prev_meas(bits=bits_obs, psnr=float(psnr_obs), qp=int(pend.qp_used))

            # === mini-GOP 收尾打印（三项：真实比特和、预估预算、平均PSNR）===
            if done:
                frames = max(1, st["frames"])
                actual_bits_sum = st["bits"]
                avg_psnr = st["psnr"] / frames
                est_budget = st.get("budget", float(pend.meta.get("mg_bits_tgt", 0.0)))
                mg_id = int(pend.meta.get("mg_id", 0))
                # print(f"[MG] gop={gop_id} mg_id={mg_id} | "
                #       f"actual_bits_sum={actual_bits_sum:.0f} | "
                #       f"est_budget(mg_bits_tgt)={est_budget:.0f} | "
                #       f"avg_psnr={avg_psnr:.2f} dB")

                # 用完清除，避免累积
                self.mg_stats.pop(key, None)

                self.last_doc_in_mg = None
                # 仅清理“非常旧”的 pending：保留最近 K 个，并给宽限时间
                keep_latest = int(getattr(self.cfg, "pending_keep_latest", 64))
                grace_ms = int(getattr(self.cfg, "pending_grace_ms", 3000))  # 3秒
                now = now_ms()
                ks = sorted(self.pending.keys())
                for k in ks[:-keep_latest]:
                    p = self.pending.get(k)
                    if p and (now - getattr(p, "created_at_ms", now)) > grace_ms:
                        self.pending.pop(k, None)

            try_remove(fb_path)
            anyp = True
        return anyp

    def _write_epoch_metrics(self, epoch_id: int):
        """把上一轮 epoch 的平均 lossa/lossc 以及帧平均 bits/PSNR 落到 CSV。"""
        try:
            mean_a = (self._ep_loss_sum_a / max(1, self._ep_updates))
            mean_c = (self._ep_loss_sum_c / max(1, self._ep_updates))
            avg_bpf = (self._ep_bits_sum / max(1, self._ep_frames))     # bits per frame
            avg_psnr = (self._ep_psnr_sum / max(1, self._ep_frames))    # dB

            # 如需 kbps，可在 config 里提供 fps
            fps = float(getattr(self.cfg, "fps", 0.0))
            avg_kbps = (avg_bpf * fps / 1000.0) if fps > 0 else ""

            row = {
                "epoch": epoch_id,
                "updates": self._ep_updates,
                "lossa_mean": f"{mean_a:.6f}",
                "lossc_mean": f"{mean_c:.6f}",
                "frames": self._ep_frames,
                "avg_bits_per_frame": f"{avg_bpf:.2f}",
                "avg_psnr_db": f"{avg_psnr:.3f}",
                "avg_kbps": f"{avg_kbps:.2f}" if avg_kbps != "" else "",
                "timestamp": int(time.time()),
            }

            file_exists = os.path.exists(self._metrics_csv)
            with open(self._metrics_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists and not self._metrics_csv_inited:
                    w.writeheader()
                    self._metrics_csv_inited = True
                w.writerow(row)
            print(f"[RL][CSV] epoch={epoch_id} avg_lossa={row['lossa_mean']} avg_lossc={row['lossc_mean']} "
                  f"| frames={self._ep_frames} avg_bpf={row['avg_bits_per_frame']} avg_psnr={row['avg_psnr_db']}"
                  + (f" avg_kbps={row['avg_kbps']}" if avg_kbps != "" else ""))
        except Exception as e:
            print(f"[warn] failed to write metrics csv: {e}")


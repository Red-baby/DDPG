# -*- coding: utf-8 -*-
"""
环境I/O与训练主循环（DOC优先；兼容 frame_ 前缀）：
- 轮询 rl_dir，读取编码器写出的 rq.json / fb.json；
- 构建状态 → 选择动作 → 写 qp.txt（沿用 rq 的前缀）；
- 读取反馈 → （训练模式下）计算奖励 → 存回放 → 训练若干步；
- 以 mini-GOP 为 episode（frames_left_mg==1 判 done）；
- 在 mini-GOP 结束时输出：真实比特和、预估 mini-GOP 预算（mg_bits_tgt）、平均 PSNR。

新增：
- 验证/推理模式：cfg.mode in {"val","infer"} 时，仅出动作，不训练，不写回放；但统计/CSV 仍正常。
"""

import csv, os, glob, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict
from utils import safe_read_json, safe_write_text, try_remove, now_ms, _float, _int
from state import StateBuilder, STATE_FIELDS
from agent import DDPG, TD3, DualCriticDDPG
from reward import compute_reward, compute_reward_dual


def _scan_rq_files(rl_dir: str):
    return sorted(glob.glob(os.path.join(rl_dir, "frame_*.rq.json")))


def _scan_fb_files(rl_dir: str):
    return sorted(glob.glob(os.path.join(rl_dir, "frame_*.fb.json")))


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
        elif algo in ("dual", "dual_ddpg", "dual_td3"):
            self.agent = DualCriticDDPG(state_dim=len(STATE_FIELDS), cfg=cfg)
        else:
            self.agent = DDPG(state_dim=len(STATE_FIELDS), cfg=cfg)

        self.pending: Dict[int, Pending] = {}
        self.last_doc_in_mg: Optional[int] = None

        self._rew_ema_q = 0.0
        self._rew_ema_b = 0.0

        self._seen_mg_ids = set()
        self._curr_gop_id = 0
        self._last_frames_left_gop = None
        self.mg_bits_tgt_total = 0.0
        self._gop_plan_bits = {}
        self._gop_init_rem = {}

        self.mg_stats: Dict[tuple, dict] = {}

        self.epoch_idx = 0
        self.epoch_total = 0
        self._ep_loss_sum_a = 0.0
        self._ep_loss_sum_c = 0.0
        self._ep_updates = 0
        self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0
        self._ep_frames = 0

        self._metrics_csv = getattr(self.cfg, "metrics_csv", "epoch_losses.csv")
        self._metrics_csv_inited = False
        self.loss_ema_a = None
        self.loss_ema_c = None

    def set_epoch(self, idx: int, total: int):
        # 写上一个 epoch 的指标（如果有）
        if self.epoch_idx > 0 and (self._ep_updates > 0 or self._ep_frames > 0):
            self._write_epoch_metrics(self.epoch_idx)
        self.epoch_idx = int(idx)
        self.epoch_total = int(total)
        self._ep_loss_sum_a = 0.0
        self._ep_loss_sum_c = 0.0
        self._ep_updates = 0
        self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0
        self._ep_frames = 0
        self._rew_ema_q = 0.0
        self._rew_ema_b = 0.0

    def _update_loss_ema(self, ret):
        """EMA for losses. Safe when la is None (dual-critic actor updates only at terminals)."""
        if ret is None:
            return
        lc, la = ret
        b = float(getattr(self.cfg, "loss_ema_beta", 0.2))
        # critic EMA
        if lc is not None:
            lc = float(lc)
            if self.loss_ema_c is None:
                self.loss_ema_c = lc
            else:
                self.loss_ema_c = (1.0 - b) * self.loss_ema_c + b * lc
        # actor EMA (skip if la is None)
        if la is not None:
            la = float(la)
            if self.loss_ema_a is None:
                self.loss_ema_a = la
            else:
                self.loss_ema_a = (1.0 - b) * self.loss_ema_a + b * la

    def serve_loop(self, stop_evt):
        print(f"[RL] watching: {self.cfg.rl_dir} | mode={self.cfg.mode}")
        last_print = now_ms()
        while not stop_evt.is_set():
            progressed = False
            progressed |= self.handle_requests()
            progressed |= self.handle_feedbacks()

            # 训练仅在 train 模式下进行
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
                print(
                    f"[RL] epoch {self.epoch_idx}/{self.epoch_total} | steps env/train: {getattr(self.agent, 'total_env_steps', 0)}/{getattr(self.agent, 'total_train_steps', 0)} | replay={len(getattr(self.agent, 'buf', getattr(self.agent, 'bufC', [])))}{loss_str}")
                last_print = now

            if not progressed:
                time.sleep(0.003)

        # 退出前，尽量处理残余反馈
        for _ in range(200):
            any_left = self.handle_feedbacks()
            if not any_left:
                break
            time.sleep(0.003)
        # 将本 epoch 的指标写入 CSV（包括验证/推理模式）
        self._write_epoch_metrics(self.epoch_idx)
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

            # GOP 变化检测（只影响统计）
            flg = _int(rq.get("frames_left_gop", -1))
            if self._last_frames_left_gop is None:
                if flg >= 0 and 0 not in self._gop_init_rem:
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
                self._last_frames_left_gop = flg
            else:
                if flg > self._last_frames_left_gop:
                    self._curr_gop_id += 1
                    self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))
                self._last_frames_left_gop = flg

            meta["gop_id"] = self._curr_gop_id

            mg_idx = _int(rq.get("mg_index", 0))
            key = (self._curr_gop_id, mg_idx)
            if key not in self.mg_stats:
                self.mg_stats[key] = {
                    "bits": 0.0, "psnr": 0.0, "frames": 0,
                    "budget": _float(rq.get("mg_bits_tgt", 0.0))
                }

            mg_id = _int(rq.get("mg_id", -1))
            if mg_id >= 0 and mg_id not in self._seen_mg_ids:
                self._seen_mg_ids.add(mg_id)
                mg_bits_tgt = _float(rq.get("mg_bits_tgt", 0.0))
                self.mg_bits_tgt_total += mg_bits_tgt
                self._gop_plan_bits[self._curr_gop_id] = self._gop_plan_bits.get(self._curr_gop_id, 0.0) + mg_bits_tgt
            if self._curr_gop_id not in self._gop_init_rem:
                self._gop_init_rem[self._curr_gop_id] = _float(rq.get("gop_bits_rem", 0.0))

            # ==== 选动作，写回 qp ====
            explore = (self.cfg.mode == "train")
            # 若 mini-GOP 剩余预算已为 0（或接近 0），直接强制最大 QP
            mg_rem = _float(rq.get("mg_bits_rem", 0.0))
            if mg_rem <= 0.0:
                qp = self.cfg.qp_max
                meta["forced_max_qp"] = True
            else:
                qp = self.agent.select_action(s, explore=explore)
                meta["forced_max_qp"] = False
            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            safe_write_text(qp_path, f"{qp}\n")
            try_remove(rq_path)

            denom = max(1, (self.cfg.qp_max - self.cfg.qp_min))
            a01 = float((qp - self.cfg.qp_min) / denom)

            doc = _int(meta.get("doc", -1))

            self.pending[doc] = Pending(
                state=s, meta=meta, action_a01=a01, qp_used=qp, next_state=None, done=False,
                created_at_ms=now_ms()
            )

            # Dual-critic: 收集无噪声 rollout 的状态，用于 mini-GOP 终止时更新 actor
            if str(getattr(self.cfg, 'algo', '')).lower() in ('dual', 'dual_ddpg', 'dual_td3'):
                if not bool(meta.get("forced_max_qp", False)):
                    try:
                        self.agent.rollout_collect_state(s)
                    except Exception:
                        pass

            # 为上一帧补 next_state
            if self.last_doc_in_mg is not None and self.last_doc_in_mg in self.pending:
                prev = self.pending[self.last_doc_in_mg]
                if prev.next_state is None:
                    prev.next_state = s
            self.last_doc_in_mg = doc

            if hasattr(self.agent, "total_env_steps"):
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

            doc = _int(fb.get("doc", -1))
            if doc not in self.pending:
                print(f"[RL][WARN] feedback for unknown DOC={doc}")
                try_remove(fb_path)
                continue

            pend = self.pending[doc]

            # 观测
            psnr_y = _float(fb.get("psnr_y", 0.0))
            if getattr(self.cfg, "psnr_mode", "y") == "yuv":
                pu = _float(fb.get("psnr_u", 0.0));
                pv = _float(fb.get("psnr_v", 0.0))
                psnr_obs = (6 * psnr_y + pu + pv) / 8.0 if (psnr_y > 0 and pu > 0 and pv > 0) else psnr_y
            else:
                psnr_obs = psnr_y
            bits_obs = _float(fb.get("bits", 0.0))

            self._ep_bits_sum += float(bits_obs)
            self._ep_psnr_sum += float(psnr_obs)
            self._ep_frames += 1

            gop_id = int(pend.meta.get("gop_id", 0))
            mg_id = int(pend.meta.get("mg_id", 0))
            key = (gop_id, mg_id)
            st = self.mg_stats.get(key)
            if st is None:
                st = {"bits": 0.0, "psnr": 0.0, "frames": 0, "budget": float(pend.meta.get("mg_bits_tgt", 0.0))}
                self.mg_stats[key] = st
            st["bits"] += bits_obs
            st["psnr"] += psnr_obs
            st["frames"] += 1
            # === 把“当前帧之前的累计用比特”带给 reward，避免 rem 被夹到0失真 ===
            used_before = float(st["bits"] - bits_obs)  # 截止上一帧的真实累计
            meta2 = dict(pend.meta)
            meta2["mg_used_before"] = used_before

            # 是否 episode 终止
            flm_val = fb.get("frames_left_mg", pend.meta.get("frames_left_mg", None))
            done = (flm_val is not None) and (int(flm_val) == 0)

            # === 推入回放 + 训练（仅训练模式） ===
            if self.cfg.mode == "train":
                algo = str(getattr(self.cfg, 'algo', '')).lower()
                if algo in ('dual', 'dual_ddpg', 'dual_td3'):
                    rD, rR = compute_reward_dual(self.cfg, fb, rq_meta=meta2)
                    if pend.next_state is None:
                        pend.next_state = pend.state.clone() if not done else torch.zeros_like(pend.state)
                    # push into dual buffer
                    self.agent.bufC.push(
                        pend.state.numpy(),
                        np.array([[pend.action_a01]], dtype=np.float32),
                        np.array([[rD]], dtype=np.float32),
                        np.array([[rR]], dtype=np.float32),
                        pend.next_state.numpy(),
                        np.array([[1.0 if done else 0.0]], dtype=np.float32)
                    )
                else:
                    mg_ctx = {"frames_so_far": int(max(1, st["frames"])), "ema_abs_q": float(self._rew_ema_q),
                              "ema_abs_b": float(self._rew_ema_b)}
                    r = compute_reward(self.cfg, fb, rq_meta=pend.meta,
                                       prev_psnr_cached=float(self.sb.prev_psnr),
                                       mg_ctx=mg_ctx)
                    self._rew_ema_q = float(mg_ctx.get("ema_abs_q", self._rew_ema_q))
                    self._rew_ema_b = float(mg_ctx.get("ema_abs_b", self._rew_ema_b))

                    if pend.next_state is None:
                        pend.next_state = pend.state.clone() if not done else torch.zeros_like(pend.state)

                    self.agent.buf.push(
                        pend.state.numpy(),
                        np.array([[pend.action_a01]], dtype=np.float32),
                        np.array([[r]], dtype=np.float32),
                        pend.next_state.numpy(),
                        np.array([[1.0 if done else 0.0]], dtype=np.float32)
                    )
            else:
                # 验证/推理模式：不入回放、不训练
                pass

            # 更新上一帧缓存
            self.sb.update_prev_meas(bits=float(bits_obs), psnr=float(psnr_obs), qp=int(pend.qp_used))

            if done:
                frames = max(1, st["frames"])
                actual_bits_sum = st["bits"]
                avg_psnr = st["psnr"] / frames
                est_budget = st.get("budget", float(pend.meta.get("mg_bits_tgt", 0.0)))
                # 可打印 mini-GOP 汇总（按需开启）
                # print(f"[MG] gop={gop_id} mg={mg_idx} budget={est_budget:.0f} used={actual_bits_sum:.0f} avgPSNR={avg_psnr:.2f}")

                algo = str(getattr(self.cfg, 'algo', '')).lower()
                if algo in ('dual', 'dual_ddpg', 'dual_td3'):
 # gate: treat "over budget" as actual_bits > (estimated_budget * over_budget_factor)
                    fac = float(getattr(self.cfg, "over_budget_factor", 1.0))
                    thr = est_budget * fac if est_budget > 0 else est_budget
                    over_budget = bool(actual_bits_sum > thr)
                    try:
                        which, loss_a = self.agent.finish_rollout_and_update_actor(over_budget)
                        if loss_a is not None:
                            self._ep_loss_sum_a += float(loss_a)
                            self._ep_updates += 1
                    except Exception as e:
                        print(f"[RL][WARN] dual-critic actor update failed: {e}")
                self.mg_stats.pop(key, None)
                self.last_doc_in_mg = None

                # 清理过旧 pending
                keep_latest = int(getattr(self.cfg, "pending_keep_latest", 64))
                grace_ms = int(getattr(self.cfg, "pending_grace_ms", 3000))
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
        try:
            mean_a = (self._ep_loss_sum_a / max(1, self._ep_updates))
            mean_c = (self._ep_loss_sum_c / max(1, self._ep_updates))
            avg_bpf = (self._ep_bits_sum / max(1, self._ep_frames))
            avg_psnr = (self._ep_psnr_sum / max(1, self._ep_frames))
            fps = float(getattr(self.cfg, "fps", 0.0))
            avg_kbps = (avg_bpf * fps / 1000.0) if fps > 0 else ""
            row = {
                "epoch": epoch_id,
                "mode": str(self.cfg.mode),
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
            print(
                f"[RL][CSV] epoch={epoch_id} mode={self.cfg.mode} avg_lossa={row['lossa_mean']} avg_lossc={row['lossc_mean']} | frames={self._ep_frames} avg_bpf={row['avg_bits_per_frame']} avg_psnr={row['avg_psnr_db']}" + (
                    f" avg_kbps={row['avg_kbps']}" if row['avg_kbps'] != "" else ""))
        except Exception as e:
            print(f"[warn] failed to write metrics csv: {e}")

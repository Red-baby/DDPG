# -*- coding: utf-8 -*-
"""
环境 I/O 与训练主循环（frame_* 前缀）：
- 轮询 rl_dir 读 *.rq.json / *.fb.json
- 构建状态 -> 选择动作 -> 写 *.qp.txt
- 终端帧：基于 2-pass 参考做 Lagrangian 成本（更新 lambda_b/lambda_q）
"""
import csv, os, glob, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from utils import safe_read_json, safe_write_text, try_remove, now_ms, _float, _int
from state import StateBuilder, STATE_FIELDS
from agent import get_agent
from reward import compute_reward
from baseline import TwoPassBaseline

def _scan_rq_files(rl_dir: str): return sorted(glob.glob(os.path.join(rl_dir, "frame_*.rq.json")))
def _scan_fb_files(rl_dir: str): return sorted(glob.glob(os.path.join(rl_dir, "frame_*.fb.json")))

# -------- 跨 miniGOP 上下文（仅做全局 PSNR EMA / mg 内计数）--------
class _CrossMGCtx:
    def __init__(self, psnr_beta=0.98):
        self.gema = 0.0
        self.mg_idx = 0
        self.has_sc = False
        self.beta = float(psnr_beta)
    def on_new_request(self, rq: dict) -> Tuple[dict, dict]:
        mg_size = max(1, int(rq.get("mg_size", 16)))
        frames_left = max(1, int(rq.get("frames_left_mg", 1)))
        if frames_left == mg_size:
            self.mg_idx = 1; self.has_sc = False
        else:
            self.mg_idx = max(1, self.mg_idx + 1)
        is_sc = int(rq.get("scene_cut", rq.get("is_scene_cut", 0))) != 0
        if is_sc: self.has_sc = True
        rq2 = dict(rq)
        rq2["global_psnr_ema"] = float(self.gema) if self.gema > 0 else 0.0
        rq2["mg_frame_idx"] = int(self.mg_idx)
        rq2["has_sc_in_mg"] = int(self.has_sc)
        mg_ctx = {
            "global_psnr_ema": rq2["global_psnr_ema"],
            "mg_frame_idx": rq2["mg_frame_idx"],
            "has_sc_in_mg": bool(self.has_sc),
        }
        return rq2, mg_ctx
    def on_feedback(self, fb: dict, rq: dict):
        psnr = float(fb.get("psnr_y", fb.get("psnr", 0.0)) or 0.0)
        if psnr > 0.0:
            b = 1.0 - self.beta
            self.gema = psnr if self.gema <= 0.0 else (1.0 - b) * self.gema + b * psnr
        if int(rq.get("scene_cut", rq.get("is_scene_cut", 0))) != 0:
            self.has_sc = True

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
        self.mgctx = _CrossMGCtx(psnr_beta=getattr(self.cfg, "inter_global_ema_beta", 0.98))

        self.agent = get_agent(len(STATE_FIELDS), cfg)

        self.pending: Dict[int, Pending] = {}
        self.last_doc_in_mg: Optional[int] = None

        # reward 平衡的 EMA
        self._rew_ema_q = 0.0
        self._rew_ema_b = 0.0

        # GOP/miniGOP 统计
        self._curr_gop_id = 0
        self._last_frames_left_gop = None
        self.mg_stats: Dict[Tuple[int,int], dict] = {}  # (gop_id, mg_idx) -> stats

        # Lagrangian 乘子
        self.lambda_b = float(getattr(cfg, "lag_b_init", 0.0))
        self.lambda_q = float(getattr(cfg, "lag_q_init", 0.0))

        # 2-pass 基线
        self.baseline = None
        if isinstance(getattr(self.cfg, "twopass_log_path", ""), str) and self.cfg.twopass_log_path:
            self._try_load_baseline(self.cfg.twopass_log_path)

        # 训练统计
        self.epoch_idx = 0; self.epoch_total = 0
        self._ep_loss_sum_a = 0.0; self._ep_loss_sum_c = 0.0
        self._ep_updates = 0; self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0; self._ep_frames = 0
        self._metrics_csv = getattr(self.cfg, "metrics_csv", "epoch_metrics.csv")
        self._metrics_csv_inited = False
        self.loss_ema_a = None; self.loss_ema_c = None

    def _try_load_baseline(self, path: str):
        try:
            self.baseline = TwoPassBaseline(path)
            print(f"[RL] loaded 2-pass baseline: {path} ({len(self.baseline.map)} frames)")
        except Exception as e:
            print(f"[RL][WARN] failed to load 2-pass baseline from {path}: {e}")
            self.baseline = None

    def set_epoch(self, idx: int, total: int, twopass_log_path: Optional[str] = None):
        if self.epoch_idx > 0 and (self._ep_updates > 0 or self._ep_frames > 0):
            self._write_epoch_metrics(self.epoch_idx)
        self.epoch_idx = int(idx); self.epoch_total = int(total)
        self._ep_loss_sum_a = self._ep_loss_sum_c = 0.0
        self._ep_updates = 0; self._ep_bits_sum = 0.0
        self._ep_psnr_sum = 0.0; self._ep_frames = 0
        self._rew_ema_q = 0.0; self._rew_ema_b = 0.0
        if twopass_log_path:
            self.cfg.twopass_log_path = twopass_log_path
            self._try_load_baseline(twopass_log_path)

    def _update_loss_ema(self, ret):
        if ret is None: return
        lc, la = ret
        b = float(getattr(self.cfg, "loss_ema_beta", 0.2))
        if lc is not None:
            self.loss_ema_c = float(lc) if self.loss_ema_c is None else (1.0 - b) * self.loss_ema_c + b * float(lc)
        if la is not None:
            self.loss_ema_a = float(la) if self.loss_ema_a is None else (1.0 - b) * self.loss_ema_a + b * float(la)

    def serve_loop(self, stop_evt):
        print(f"[RL] watching: {self.cfg.rl_dir} | mode={self.cfg.mode}")
        last_print = now_ms()
        while not stop_evt.is_set():
            progressed = self.handle_requests() or self.handle_feedbacks()
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
                loss_str = ""
                if self.loss_ema_a is not None and self.loss_ema_c is not None:
                    loss_str = f" | loss_a={self.loss_ema_a:.4f} loss_c={self.loss_ema_c:.4f}"
                print(f"[RL] epoch {self.epoch_idx}/{self.epoch_total} | steps env/train: "
                      f"{getattr(self.agent, 'total_env_steps', 0)}/{getattr(self.agent, 'total_train_steps', 0)} "
                      f"| replay={len(self.agent.buf)}{loss_str} | λb={self.lambda_b:.3f} λq={self.lambda_q:.3f}")
                last_print = now
            if not progressed:
                time.sleep(0.003)

        # 尾包：尽力消化反馈
        for _ in range(200):
            if not self.handle_feedbacks(): break
            time.sleep(0.003)
        self._write_epoch_metrics(self.epoch_idx)
        print("[RL] serve loop exit.")

    # ---------------- requests ----------------
    def handle_requests(self) -> bool:
        if self.baseline is None and isinstance(getattr(self.cfg, "twopass_log_path", ""),
                                                str) and self.cfg.twopass_log_path:
            self._try_load_baseline(self.cfg.twopass_log_path)
        rq_paths = _scan_rq_files(self.cfg.rl_dir)
        if not rq_paths: return False
        anyp = False
        for rq_path in rq_paths:
            try:
                rq = safe_read_json(rq_path)
            except Exception as e:
                print(f"[RL][WARN] bad rq json {rq_path}: {e}")
                try_remove(rq_path); continue

            # 兜底 mg_index：若编码器未提供，就用 poc/mg_size 推算
            mg_idx = _int(rq.get("mg_index", None))
            if mg_idx is None:
                poc = _int(rq.get("poc", -1))
                mg_size = max(1, _int(rq.get("mg_size", 16)))
                mg_idx = (poc - 1) // mg_size if (poc is not None and poc >= 0) else 0
                rq["mg_index"] = mg_idx  # 补回请求字典，后续 meta 会带上

            rq_enriched, mg_ctx_cross = self.mgctx.on_new_request(rq)
            s, meta = self.sb.build(rq_enriched)
            # 确保 meta 也携带 mg_index
            meta["mg_index"] = int(mg_idx)

            # GOP 计数（仅统计用）
            flg = _int(rq.get("frames_left_gop", -1))
            if self._last_frames_left_gop is None:
                self._last_frames_left_gop = flg
            else:
                if flg > self._last_frames_left_gop:  # gop 回绕
                    self._curr_gop_id += 1
                self._last_frames_left_gop = flg
            meta["gop_id"] = self._curr_gop_id

            # 建立 mg 统计
            key = (self._curr_gop_id, int(mg_idx))
            st = self.mg_stats.get(key)
            if st is None:
                st = {"bits": 0.0, "psnr": 0.0, "frames": 0,
                      "budget": _float(rq.get("mg_bits_tgt", 0.0)),
                      "ref_bits_total": 0.0, "ref_psnr_avg": 0.0}
                self.mg_stats[key] = st
                # 2-pass 参考：只在第一次看到该 mg 时放入
                poc = int(_int(rq_enriched.get("poc", rq.get("poc", -1))))
                if self.baseline is not None and poc >= 0:
                    try:
                        mg_size = int(_int(rq.get("mg_size", 16)))
                        ref = self.baseline.mg_stats(poc, mg_size=mg_size)
                        st["ref_bits_total"] = float(ref["bits_total"])
                        st["ref_psnr_avg"]   = float(ref["psnr_avg"])
                        st["ref_start_poc"]  = int(ref["start"])
                        st["ref_end_poc"]    = int(ref["end"])
                        # print(f"[RL][baseline] mg{key} ref_bits={st['ref_bits_total']} ref_psnr={st['ref_psnr_avg']:.3f}")
                    except Exception as e:
                        print(f"[RL][WARN] baseline mg_stats failed for poc={poc}: {e}")

            # 选动作
            explore = (self.cfg.mode == "train")
            qp = self.agent.select_action(s, meta["base_q"], explore=explore)

            # 安全层（可关）：基于剩余预算/剩余帧数的后置修正
            if bool(getattr(self.cfg, "safety_layer_enable", True)):
                mg_rem = float(_float(rq.get("mg_bits_rem", 0.0)))
                L = max(1, int(_int(rq.get("frames_left_mg", 1))))
                per_allow = (mg_rem / L) if mg_rem > 0 else None
                pred = float(_float(rq.get("bits_pred_frame", rq.get("bits_plan_frame", 0.0))))
                if per_allow and pred > 0:
                    slack = float(getattr(self.cfg, "safety_slack", 1.05))
                    step  = int(getattr(self.cfg, "safety_qp_step", 2))
                    if pred > slack * per_allow:
                        qp = min(qp + step, self.cfg.qp_max)
                    elif pred < per_allow / slack:
                        qp = max(qp - step, self.cfg.qp_min)

            qp_path = rq_path.replace(".rq.json", ".qp.txt")
            try:
                safe_write_text(qp_path, f"{qp}\n")
            except PermissionError as e:
                print(f"[RL][WARN] safe_write_text failed: {e}")

            try_remove(rq_path)

            # 把 “qp→a01” 存入 pending（便于回放）
            delta_max = float(getattr(self.cfg, "delta_qp_max", 20.0))
            a01 = 0.5 + (float(qp) - float(meta["base_q"])) / (2.0 * delta_max)
            a01 = float(np.clip(a01, 0.0, 1.0))
            doc = int(_int(meta.get("doc", rq.get("doc", -1))))
            self.pending[doc] = Pending(
                state=s, meta=meta, action_a01=a01, qp_used=qp,
                next_state=None, done=False, created_at_ms=now_ms()
            )

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

    # ---------------- feedbacks ----------------
    def handle_feedbacks(self) -> bool:
        fb_paths = _scan_fb_files(self.cfg.rl_dir)
        if not fb_paths: return False
        anyp = False
        for fb_path in fb_paths:
            try:
                fb = safe_read_json(fb_path)
            except Exception as e:
                print(f"[RL][WARN] bad fb json {fb_path}: {e}")
                try_remove(fb_path); continue

            doc = _int(fb.get("doc", -1))
            if doc not in self.pending:
                try_remove(fb_path); continue
            pend = self.pending[doc]

            # 观测
            psnr_y = _float(fb.get("psnr_y", fb.get("psnr", 0.0)))
            pu = _float(fb.get("psnr_u", 0.0)); pv = _float(fb.get("psnr_v", 0.0))
            psnr_obs = (6 * psnr_y + pu + pv) / 8.0 if (psnr_y>0 and pu>0 and pv>0) else psnr_y
            bits_obs = _float(fb.get("bits", 0.0))

            self._ep_bits_sum += float(bits_obs)
            self._ep_psnr_sum += float(psnr_obs)
            self._ep_frames += 1

            gop_id = int(pend.meta.get("gop_id", 0))
            mg_id  = int(pend.meta.get("mg_index", pend.meta.get("mg_id", 0)))
            key = (gop_id, mg_id)
            st = self.mg_stats.get(key)
            if st is None:
                st = {"bits": 0.0, "psnr": 0.0, "frames": 0,
                      "budget": float(pend.meta.get("mg_bits_tgt", 0.0)),
                      "ref_bits_total": 0.0, "ref_psnr_avg": 0.0}
                self.mg_stats[key] = st

            # 传给 reward 的“当前帧之前的累计”
            used_before = float(st["bits"])
            frames_so_far_pre = int(st["frames"])
            meta2 = dict(pend.meta)
            meta2["mg_used_before"] = used_before
            meta2["frames_so_far"]  = frames_so_far_pre
            meta2["mg_avg_psnr_so_far"] = (st["psnr"]/frames_so_far_pre) if frames_so_far_pre>0 else -1.0

            # 先累计（便于终止帧统计）
            st["bits"] += float(bits_obs)
            st["psnr"] += float(psnr_obs)
            st["frames"] += 1

            done = (_int(fb.get("frames_left_mg", pend.meta.get("frames_left_mg", 1))) == 0)

            # ---- 组 mg_ctx 并计算逐帧 r ----
            mg_ctx = {
                # 注意：传“当前帧之前”的计数，避免 off-by-one
                "frames_so_far": max(0, frames_so_far_pre),
                "ema_abs_q": float(self._rew_ema_q),
                "ema_abs_b": float(self._rew_ema_b),
                "global_psnr_ema": float(self.mgctx.gema) if self.mgctx.gema > 0 else 0.0,
                "mg_frame_idx": int(self.mgctx.mg_idx),
                "has_sc_in_mg": bool(self.mgctx.has_sc),
                # TD3-Lagrangian 的乘子（供终止帧惩罚使用）
                "lambda_b": float(self.lambda_b),
                "lambda_q": float(self.lambda_q),
            }
            # 如该 mg 有 2-pass 参考，把参考传进 reward
            if "ref_bits_total" in st and "ref_psnr_avg" in st:
                mg_ctx["ref_bits_total"] = float(st.get("ref_bits_total", 0.0))
                mg_ctx["ref_psnr_avg"]   = float(st.get("ref_psnr_avg", 0.0))

            r = compute_reward(self.cfg, fb, rq_meta=meta2,
                               prev_psnr_cached=float(self.sb.prev_psnr), mg_ctx=mg_ctx)

            # 带回逐帧 EMA（用于自动平衡）
            self._rew_ema_q = float(mg_ctx.get("ema_abs_q", self._rew_ema_q))
            self._rew_ema_b = float(mg_ctx.get("ema_abs_b", self._rew_ema_b))

            if pend.next_state is None:
                pend.next_state = (pend.state.clone() if not done else torch.zeros_like(pend.state))

            self.agent.buf.push(
                pend.state.numpy(),
                np.array([[pend.action_a01]], dtype=np.float32),
                np.array([[r]], dtype=np.float32),
                pend.next_state.numpy(),
                np.array([[1.0 if done else 0.0]], dtype=np.float32)
            )

            # 反馈后更新 prev_* 与跨 mg gEMA
            self.sb.update_prev_meas(bits=float(bits_obs), psnr=float(psnr_obs),
                                     qp=int(pend.qp_used),
                                     ref_bpf_cur=float(pend.meta.get("ref_bpf_cur", 0.0)))
            self.mgctx.on_feedback(fb, pend.meta)

            # ---- 终止帧：更新乘子 λb/λq（TD3-Lagrangian）----
            if done:
                ref_bits_total = float(st.get("ref_bits_total", 0.0))
                ref_psnr_avg   = float(st.get("ref_psnr_avg", 0.0))
                if ref_bits_total > 0.0 and st["frames"] > 0:
                    used_after = float(st["bits"])
                    rho = used_after / max(1.0, ref_bits_total)
                    tol = float(getattr(self.cfg, "ref_bits_tol", 0.10))
                    c_b = max(0.0, abs(rho - 1.0) - tol)
                    avg_rl = (st["psnr"] / max(1, st["frames"]))
                    c_q = 0.0 if abs(rho - 1.0) > tol else max(0.0, ref_psnr_avg - avg_rl)

                    # λ ← [λ + η * c]_+ 并截断上限
                    self.lambda_b = float(min(self.cfg.lag_b_max, max(0.0, self.lambda_b + self.cfg.lag_eta_b * c_b)))
                    self.lambda_q = float(min(self.cfg.lag_q_max, max(0.0, self.lambda_q + self.cfg.lag_eta_q * c_q)))

                    print(f"[RL][mg end] gop={gop_id} mg={mg_id} | bits_rl={used_after:.0f} "
                          f"bits_ref={ref_bits_total:.0f} rho={rho:.3f} | "
                          f"psnr_rl={avg_rl:.3f} psnr_ref={ref_psnr_avg:.3f} | "
                          f"c_b={c_b:.4f} c_q={c_q:.4f} | λb={self.lambda_b:.3f} λq={self.lambda_q:.3f}")

                # 清理该 mg 的状态
                self.mg_stats.pop(key, None)
                self.last_doc_in_mg = None

                # 清理过旧 pending
                keep_latest = 64; grace_ms = 3000
                now = now_ms()
                ks = sorted(self.pending.keys())
                for k in ks[:-keep_latest]:
                    p = self.pending.get(k)
                    if p and (now - getattr(p, "created_at_ms", now)) > grace_ms:
                        self.pending.pop(k, None)

            try_remove(fb_path)
            anyp = True
        return anyp

    # ---------------- CSV ----------------
    def _write_epoch_metrics(self, epoch_id: int):
        try:
            mean_a = (self._ep_loss_sum_a / max(1, self._ep_updates))
            mean_c = (self._ep_loss_sum_c / max(1, self._ep_updates))
            avg_bpf = (self._ep_bits_sum / max(1, self._ep_frames))
            avg_psnr = (self._ep_psnr_sum / max(1, self._ep_frames))
            fps = float(getattr(self.cfg, "fps", 0.0))
            avg_kbps = (avg_bpf * fps / 1000.0) if fps > 0 else ""
            row = {
                "epoch": epoch_id, "mode": str(self.cfg.mode), "updates": self._ep_updates,
                "lossa_mean": f"{mean_a:.6f}", "lossc_mean": f"{mean_c:.6f}",
                "frames": self._ep_frames, "avg_bits_per_frame": f"{avg_bpf:.2f}",
                "avg_psnr_db": f"{avg_psnr:.3f}",
                "avg_kbps": f"{avg_kbps:.2f}" if avg_kbps != "" else "", "timestamp": int(time.time()),
            }
            file_exists = os.path.exists(self._metrics_csv)
            with open(self._metrics_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists and not self._metrics_csv_inited:
                    w.writeheader(); self._metrics_csv_inited = True
                w.writerow(row)
            print(f"[RL][CSV] epoch={epoch_id} mode={self.cfg.mode} "
                  f"avg_lossa={row['lossa_mean']} avg_lossc={row['lossc_mean']} | "
                  f"frames={self._ep_frames} avg_bpf={row['avg_bits_per_frame']} "
                  f"avg_psnr={row['avg_psnr_db']}" + (f" avg_kbps={row['avg_kbps']}" if row['avg_kbps'] else ""))
        except Exception as e:
            print(f"[warn] failed to write metrics csv: {e}")

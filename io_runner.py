# -*- coding: utf-8 -*-
"""
MiniGOP I/O runner:
- Watch rl_dir for mg_*.rq.json / mg_*.fb.json (encoder handshake via rl_sync.*)
- Build a 16x7 feature block via state.build_mg_state (pad by replicating last row),
  then flatten to 1D for the policy.
- Actor outputs a per-frame QP vector (length MG_MAX). We slice to mg_size and
  write exactly mg_size integers into mg_%08d.qp.txt.
- Reward: compare fb.bit_avg / fb.vmaf_avg against reference averages from the
  2-pass log mapped by POC. Episode ends when fb.gop_end == 1.
"""
import os, glob, time, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import json
from datetime import datetime

from utils import safe_read_json, safe_write_text, try_remove, now_ms, _float, _int
from agent import get_agent
from reward import compute_reward_mg
from baseline import TwoPassBaseline
from state import build_mg_state  # ← 使用你刚更新的 state.py

def _scan_mg_rq_files(rl_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(rl_dir, "mg_*.rq.json")))
def _scan_mg_fb_files(rl_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(rl_dir, "mg_*.fb.json")))

@dataclass
class PendingMG:
    state: torch.Tensor
    meta: dict
    action_a01: np.ndarray   # shape [MG_MAX], in [0,1]
    qps_abs: np.ndarray      # shape [MG_MAX], ints in [qp_min, qp_max]
    next_state: Optional[torch.Tensor] = None
    poc_list: Optional[List[int]] = None
    done: bool = False
    created_at_ms: int = 0

class RLRunner:
    def __init__(self, cfg):
        self.cfg = cfg

        # ===== MiniGOP feature layout =====
        self.MG_MAX = int(getattr(cfg, "mg_size", 16))
        self.mg_state_dim = self.MG_MAX * 7  # 7 feats/frame per state.py

        # ===== Agent (vector action) =====
        self.agent = get_agent(self.mg_state_dim, cfg, action_dim=self.MG_MAX)

        # ===== 2-pass baseline =====
        self.baseline: Optional[TwoPassBaseline] = None
        self._maybe_load_baseline(getattr(self.cfg, "twopass_log_path", ""))

        # ===== replay staging by mg_id =====
        self.pending: Dict[int, PendingMG] = {}
        self._last_mg_id: Optional[int] = None

        # ===== training stats =====
        self.epoch_idx = 0; self.epoch_total = 0
        self.loss_ema_a = None; self.loss_ema_c = None
        self._ep_updates = 0
        
        # ===== 实时统计累积 =====
        self.epoch_start_env_steps = 0  # 初始化
        self.reset_epoch_stats()
        
        # ===== Episode tracking =====
        self.current_episode_return = 0.0
        self.episode_returns = []  # 存储每个episode的return
        self.loss_history = {"critic": [], "actor": []}  # 存储loss历史
        self.training_metrics = {"episodes": [], "returns": [], "losses_c": [], "losses_a": [], "timestamps": []}
        
        # 创建日志目录
        self.log_dir = getattr(cfg, "log_dir", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def reset_epoch_stats(self):
        """重置当前epoch的统计信息"""
        self.epoch_stats = {
            "total_vmaf": 0.0,
            "total_bits": 0.0, 
            "mg_count": 0,
            "vmaf_avg": 0.0,
            "bit_avg": 0.0
        }
        # 记录当前epoch开始时的env_steps，用于计算当前epoch的增量
        self.epoch_start_env_steps = getattr(self.agent, 'total_env_steps', 0)
    
    def get_epoch_stats(self) -> Dict[str, float]:
        """获取当前epoch的统计信息"""
        if self.epoch_stats["mg_count"] > 0:
            self.epoch_stats["vmaf_avg"] = self.epoch_stats["total_vmaf"] / self.epoch_stats["mg_count"]
            self.epoch_stats["bit_avg"] = self.epoch_stats["total_bits"] / self.epoch_stats["mg_count"]
        return self.epoch_stats.copy()
        
    def _save_training_data(self):
        """保存训练数据到JSON文件"""
        data_path = os.path.join(self.log_dir, "training_metrics.json")
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[RL][WARN] Failed to save training data: {e}")
    
    def _plot_training_curves(self):
        """绘制训练曲线图"""
        try:
            if len(self.episode_returns) < 2:
                return
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制episode return曲线
            episodes = list(range(1, len(self.episode_returns) + 1))
            ax1.plot(episodes, self.episode_returns, 'b-', alpha=0.7, label='Episode Return')
            
            # 计算滑动平均
            if len(self.episode_returns) > 5:
                window = min(10, len(self.episode_returns) // 2)
                moving_avg = []
                for i in range(len(self.episode_returns)):
                    start_idx = max(0, i - window + 1)
                    moving_avg.append(np.mean(self.episode_returns[start_idx:i+1]))
                ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Return')
            ax1.set_title('Episode Return Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制loss曲线
            if self.loss_ema_c is not None and self.loss_ema_a is not None:
                loss_episodes = self.training_metrics["episodes"]
                losses_c = self.training_metrics["losses_c"]
                losses_a = self.training_metrics["losses_a"]
                
                if len(loss_episodes) > 1:
                    ax2.plot(loss_episodes, losses_c, 'g-', alpha=0.7, label='Critic Loss')
                    ax2.plot(loss_episodes, losses_a, 'orange', alpha=0.7, label='Actor Loss')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Loss')
                    ax2.set_title('Training Loss Over Time')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_yscale('log')  # 使用对数刻度
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(self.log_dir, f"training_curves_ep{len(self.episode_returns)}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[RL] Training curves saved to {plot_path}")
            
        except Exception as e:
            print(f"[RL][WARN] Failed to plot training curves: {e}")
    # --------------------- public API ---------------------
    def set_epoch(self, idx: int, total: int, twopass_log_path: Optional[str] = None):
        self.epoch_idx = int(idx); self.epoch_total = int(total)
        if twopass_log_path:
            self.cfg.twopass_log_path = twopass_log_path
        self._maybe_load_baseline(getattr(self.cfg, "twopass_log_path", ""))
        # 重置统计信息
        self.reset_epoch_stats()

    # --------------------- helpers ------------------------
    def _maybe_load_baseline(self, path: str):
        if not path:
            return
        try:
            self.baseline = TwoPassBaseline(path)
            print(f"[RL] loaded 2-pass baseline: {path}")
        except Exception as e:
            print(f"[RL][WARN] failed to load 2-pass baseline: {e}")
            self.baseline = None

    def _write_qp_file(self, rq_path: str, qps: List[int]):
        """
        原子写入 mg_XXXX.qp.txt，并立刻校验行数；打印调试信息，便于和编码器端日志对齐。
        返回 (ok: bool, qp_path: str, n_lines: int)
        """
        qp_path = rq_path.replace(".rq.json", ".qp.txt")
        # 原子写：先写 .tmp，再 rename
        tmp = qp_path + ".tmp"
        payload = "\n".join(str(int(q)) for q in qps) + "\n"
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(payload)
            
            # 在Windows上使用更安全的文件替换方法
            import shutil
            if os.path.exists(qp_path):
                try:
                    os.remove(qp_path)
                except PermissionError:
                    # 如果目标文件被占用，等待片刻再试
                    import time
                    time.sleep(0.1)
                    if os.path.exists(qp_path):
                        os.remove(qp_path)
            
            shutil.move(tmp, qp_path)
        except Exception as e:
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except Exception:
                pass
            print(f"[RL][WARN] write qp failed: {e} -> {qp_path}")
            return False, qp_path, 0

        # 复读校验（确保编码器不会读到半文件）
        try:
            with open(qp_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip() != ""]
            n_lines = len(lines)
        except Exception as e:
            print(f"[RL][WARN] verify qp failed: {e} -> {qp_path}")
            return False, qp_path, 0

        #print(f"[RL][QP] wrote {n_lines} lines -> {os.path.abspath(qp_path)} | head: {lines[:min(4, n_lines)]}")
        return (n_lines == len(qps)), qp_path, n_lines

    def _compute_ref_avgs(self, poc_list: List[int]) -> Dict[str, float]:
        """
        Use existing TwoPassBaseline.map to compute avg bits/vmaf for given POC list (exclude I/K/O).
        """
        if not self.baseline or not poc_list:
            return {"bit_avg": 0.0, "vmaf_avg": 0.0, "count": 0}
        m = self.baseline.map  # {poc: {type,bits,psnr,vmaf}}
        tot_bits = 0.0
        tot_vmaf = 0.0
        cnt = 0
        for p in poc_list:
            rec = m.get(int(p))
            if not rec: continue
            t = (rec.get("type","") or "").upper()
            if t in ("I","K","O"):  # exclude key/overlay
                continue
            tot_bits += float(rec.get("bits", 0.0) or 0.0)
            tot_vmaf += float(rec.get("vmaf", 0.0) or 0.0)
            cnt += 1
        bit_avg = (tot_bits / cnt) if cnt > 0 else 0.0
        vmaf_avg = (tot_vmaf / cnt) if cnt > 0 else 0.0
        return {"bit_avg": float(bit_avg), "vmaf_avg": float(vmaf_avg), "count": int(cnt)}

    # -------------------- mg handlers ---------------------
    def handle_mg_requests(self) -> bool:
        rq_paths = _scan_mg_rq_files(self.cfg.rl_dir)
        if not rq_paths: return False
        progressed = False
        for rq_path in rq_paths:
            try:
                rq = safe_read_json(rq_path)
            except Exception as e:
                print(f"[RL][WARN] bad mg rq json {rq_path}: {e}")
                try_remove(rq_path); continue

            # 使用 state.build_mg_state
            s, meta, poc_list = build_mg_state(rq, self.cfg)
            mg_id = int(meta.get("mg_id", -1))
            mg_size = int(meta.get("mg_size", self.MG_MAX))
            if mg_id < 0 or mg_size <= 0:
                try_remove(rq_path); continue

            # action: per-frame QP absolute (vector)
            explore = (self.cfg.mode == "train")
            a01_vec, qp_vec = self.agent.select_action_vector(
                s, mg_size=mg_size,
                qp_min=self.cfg.qp_min, qp_max=self.cfg.qp_max,
                base_q_vec=np.array(meta.get("base_q_list", []), dtype=np.float32),
                explore=explore
            )

            # write exactly mg_size integers
            qps_to_write = list(map(int, qp_vec[:mg_size + 1]))
            ok, qp_path, n_lines = self._write_qp_file(rq_path, qps_to_write)
            if not ok:
                print(f"[RL][WARN] mg_id={mg_id} write-count-mismatch exp={len(qps_to_write)} got={n_lines}")

            # stage pending transition
            self.pending[mg_id] = PendingMG(
                state=s, meta=meta, action_a01=a01_vec.copy(), qps_abs=qp_vec.copy(),
                next_state=None, poc_list=poc_list, done=False, created_at_ms=now_ms()
            )
            # chain next_state for previous mg
            if self._last_mg_id is not None and self._last_mg_id in self.pending:
                prev = self.pending[self._last_mg_id]
                if prev.next_state is None:
                    prev.next_state = s
            self._last_mg_id = mg_id

            if hasattr(self.agent, "total_env_steps"):
                self.agent.total_env_steps += 1
            try_remove(rq_path)
            progressed = True
        return progressed

    def handle_mg_feedbacks(self) -> bool:
        fb_paths = _scan_mg_fb_files(self.cfg.rl_dir)
        if not fb_paths: return False
        progressed = False
        for fb_path in fb_paths:
            try:
                fb = safe_read_json(fb_path)
            except Exception as e:
                print(f"[RL][WARN] bad mg fb json {fb_path}: {e}")
                try_remove(fb_path); continue

            mg_id = int(_int(fb.get("mg_id", -1)))
            if mg_id not in self.pending:
                try_remove(fb_path); continue
            pend = self.pending[mg_id]

            # reference averages
            ref = self._compute_ref_avgs(pend.poc_list or [])
            
            # 在reward计算时直接累积统计数据
            fb_vmaf = float(_float(fb.get("vmaf_avg", 0.0)))
            fb_bits = float(_float(fb.get("bit_avg", 0.0)))
            
            # 只统计有效的VMAF和比特率数据
            if fb_vmaf > 0.0 and fb_bits > 0.0:
                self.epoch_stats["total_vmaf"] += fb_vmaf
                self.epoch_stats["total_bits"] += fb_bits
                self.epoch_stats["mg_count"] += 1

            # reward
            r = compute_reward_mg(self.cfg, fb, ref)
            
            # 累积episode return
            self.current_episode_return += r

            # terminal?
            done = (int(_int(fb.get("gop_end", fb.get("gopend", 0)))) == 1)
            pend.done = done
            
            # 如果episode结束，记录return并重置
            if done:
                self.episode_returns.append(self.current_episode_return)
                print(f"[RL] Episode {len(self.episode_returns)} completed | Return: {self.current_episode_return:.4f}")
                
                # 记录训练指标
                self.training_metrics["episodes"].append(len(self.episode_returns))
                self.training_metrics["returns"].append(self.current_episode_return)
                self.training_metrics["losses_c"].append(self.loss_ema_c if self.loss_ema_c is not None else 0.0)
                self.training_metrics["losses_a"].append(self.loss_ema_a if self.loss_ema_a is not None else 0.0)
                self.training_metrics["timestamps"].append(datetime.now().isoformat())
                
                # 保存训练数据
                self._save_training_data()
                
                # 每10个episode绘制一次图
                if len(self.episode_returns) % 10 == 0:
                    self._plot_training_curves()
                
                self.current_episode_return = 0.0

            # next_state fallback
            s2 = pend.next_state if pend.next_state is not None else pend.state

            # push to replay
            if hasattr(self.agent, "buf"):
                self.agent.buf.push(
                    pend.state.numpy(),
                    pend.action_a01.reshape(1, -1).astype(np.float32),
                    np.array([[r]], dtype=np.float32),
                    s2.numpy(),
                    np.array([[1.0 if done else 0.0]], dtype=np.float32),
                )

            # allow training step(s)
            if self.cfg.mode == "train" and hasattr(self.agent, "train_step"):
                k = int(getattr(self.cfg, "train_steps_per_env_step", 1))
                for _ in range(max(1, k)):
                    ret = self.agent.train_step()
                    if ret is not None:
                        lc, la = ret
                        b = float(getattr(self.cfg, "loss_ema_beta", 0.2))
                        if lc is not None:
                            self.loss_ema_c = float(lc) if self.loss_ema_c is None else (1.0 - b) * self.loss_ema_c + b * float(lc)
                        if la is not None:
                            self.loss_ema_a = float(la) if self.loss_ema_a is None else (1.0 - b) * self.loss_ema_a + b * float(la)
                        self._ep_updates += 1

            try_remove(fb_path)
            del self.pending[mg_id]
            progressed = True
        return progressed

    # -------------------- main serve loop -----------------
    def serve_loop(self, stop_evt):
        print(f"[RL] watching: {self.cfg.rl_dir} | mode={self.cfg.mode}")
        last_print = now_ms()
        while not stop_evt.is_set():
            progressed = False
            progressed |= self.handle_mg_requests()
            progressed |= self.handle_mg_feedbacks()

            # periodic log
            now = now_ms()
            if now - last_print > int(getattr(self.cfg, "print_every_sec", 2.0) * 1000):
                loss_str = ""
                if self.loss_ema_a is not None and self.loss_ema_c is not None:
                    loss_str = f" | loss_a={self.loss_ema_a:.4f} loss_c={self.loss_ema_c:.4f}"
                print(f"[RL] epoch {self.epoch_idx}/{self.epoch_total} | steps env/train: "
                      f"{getattr(self.agent, 'total_env_steps', 0) - self.epoch_start_env_steps}/{getattr(self.agent, 'total_train_steps', 0)} "
                      f"| replay={len(getattr(self.agent, 'buf', []))}{loss_str}")
                last_print = now

            if not progressed:
                time.sleep(0.003)

        # drain remaining feedbacks
        for _ in range(100):
            if not self.handle_mg_feedbacks(): break
            time.sleep(0.003)
        print("[RL] serve loop exit.")

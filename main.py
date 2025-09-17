# -*- coding: utf-8 -*-
import os, argparse, threading, re
from typing import List, Dict, Tuple
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor
from dataset import add_dataset_args, build_cmds_from_dataset
from baseline import TwoPassBaseline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--mode", type=str, default="train", choices=["train","val","infer"])
    ap.add_argument("--encoder", type=str, default=Config.encoder_path)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt-prefix", type=str, default="ckpt")

    # 切换：单视频/数据集
    ap.add_argument("--use-dataset", action="store_true", help="启用数据集模式（从 --dataset-inputs 自动发现 YUV）")

    # 单视频命令模式（每条内部用 | 分隔）
    ap.add_argument("--videos", type=str, nargs="+", default=[
        "--input|E:/Git/qav1/workspace/park_mobile_1920x1080_24.yuv|--input-res|1920x1080|--frames|0|"
        "--o|./demo.ivf|--csv|./demo.csv|--bitrate|2125|--rc-mode|1|--pass|2|"
        "--stat-in|./pass1.log|--stat-out|./demo_pass2.log|"
        "--score-max|50.5|--score-avg|40.5|--score-min|38.5|--fps|24|--preset|1|"
        "--keyint|225|--bframes|15|--threads|1|--print-vmaf|1|--parallel-frames|1"
    ])

    # 注入数据集相关参数（不使用 manifest）
    add_dataset_args(ap)
    return ap.parse_args()

def _split_cmd_bar(cmd_str: str) -> List[str]:
    # 把 " --k|v|--k2|v2 " 拆成 argv
    return [p for p in cmd_str.split("|") if p]

def _find_arg(args: List[str], key: str, default: str = "") -> str:
    try:
        i = args.index(key)
        return args[i+1] if i+1 < len(args) else default
    except ValueError:
        return default

def _infer_twopass_from_stat_in(stat_in: str) -> str:
    if not stat_in:
        return ""
    # 只把 pass1 替换成 pass2；路径其它部分不动
    return stat_in.replace("pass1", "pass2")

def _get_epoch_stats(cfg: Config, twopass_log_path: str, runner: RLRunner) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    获取当前epoch的统计信息：
    - 2-pass日志的总平均VMAF和比特率
    - RL控制的编码统计信息（直接从runner中获取累积数据）
    """
    # 获取2-pass统计信息
    two_pass_stats = {"vmaf_avg": 0.0, "bit_avg": 0.0, "count": 0}
    if twopass_log_path and os.path.exists(twopass_log_path):
        try:
            baseline = TwoPassBaseline(twopass_log_path)
            # 获取所有POC的统计信息，排除I/K/O帧
            all_pocs = []
            for poc, rec in baseline.map.items():
                frame_type = (rec.get("type", "") or "").upper()
                if frame_type not in ("I", "K", "O"):  # 只统计P/B帧
                    all_pocs.append(poc)
            
            if all_pocs:
                two_pass_stats = baseline.mg_stats_from_pocs(all_pocs)
        except Exception as e:
            print(f"[STATS] Error parsing 2-pass log: {e}")
    
    # 直接从runner获取RL统计信息（reward计算时累积的数据）
    rl_stats = runner.get_epoch_stats()
    
    print(f"[STATS] RL stats from reward accumulation: {rl_stats['mg_count']} miniGOPs processed")
    
    # 显示当前epoch的env_steps增量俥调试
    current_env_steps = getattr(runner.agent, 'total_env_steps', 0) - runner.epoch_start_env_steps
    print(f"[STATS] Current epoch env_steps: {current_env_steps}")
    
    return two_pass_stats, rl_stats

def _print_epoch_stats(epoch_id: int, two_pass_stats: Dict[str, float], rl_stats: Dict[str, float], runner: RLRunner):
    """
    打印epoch统计信息
    """
    print(f"=== Epoch {epoch_id} Statistics ===")
    print(f"2-pass baseline:")
    print(f"  - Average VMAF: {two_pass_stats.get('vmaf_avg', 0.0):.2f}")
    print(f"  - Average Bits: {two_pass_stats.get('bit_avg', 0.0):.2f}")
    print(f"  - Frame count:  {two_pass_stats.get('count', 0)}")
    
    print(f"RL controlled encoding:")
    print(f"  - Average VMAF: {rl_stats.get('vmaf_avg', 0.0):.2f}")
    print(f"  - Average Bits: {rl_stats.get('bit_avg', 0.0):.2f}")
    print(f"  - MiniGOP count: {rl_stats.get('mg_count', 0)}")
    
    # 计算差异
    vmaf_diff = rl_stats.get('vmaf_avg', 0.0) - two_pass_stats.get('vmaf_avg', 0.0)
    bit_diff = rl_stats.get('bit_avg', 0.0) - two_pass_stats.get('bit_avg', 0.0)
    bit_ratio = rl_stats.get('bit_avg', 0.0) / two_pass_stats.get('bit_avg', 1.0) if two_pass_stats.get('bit_avg', 0.0) > 0 else 0.0
    
    print(f"Comparison:")
    print(f"  - VMAF difference: {vmaf_diff:+.2f}")
    print(f"  - Bits difference: {bit_diff:+.2f}")
    print(f"  - Bits ratio: {bit_ratio:.3f}x")
    
    # 添加episode return信息
    if hasattr(runner, 'episode_returns') and len(runner.episode_returns) > 0:
        returns = runner.episode_returns
        print(f"Episode Returns:")
        print(f"  - Total Episodes: {len(returns)}")
        print(f"  - Latest Return: {returns[-1]:.4f}")
        print(f"  - Average Return: {sum(returns)/len(returns):.4f}")
        if len(returns) > 1:
            print(f"  - Return Std: {(sum((x - sum(returns)/len(returns))**2 for x in returns) / len(returns))**0.5:.4f}")
            
        # 显示最近几个episode的return
        recent_count = min(5, len(returns))
        recent_returns = returns[-recent_count:]
        print(f"  - Recent {recent_count} Returns: {', '.join(f'{r:.3f}' for r in recent_returns)}")
    
    print("=" * 40)

def _save_checkpoint_if_needed(runner: RLRunner, cfg: Config, epoch_id: int):
    """根据配置决定是否保存模型检查点"""
    save_every = getattr(cfg, 'save_every_epochs', 10)
    checkpoint_dir = getattr(cfg, 'checkpoint_dir', './checkpoints')
    keep_last_n = getattr(cfg, 'keep_last_n_checkpoints', 5)
    
    if epoch_id % save_every == 0:
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_id:04d}.pt")
        try:
            runner.agent.save(checkpoint_path)
            print(f"[CKPT] Model saved to: {checkpoint_path}")
            
            # 清理旧的检查点（保留最近的N个）
            _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
            
        except Exception as e:
            print(f"[CKPT][WARN] Failed to save checkpoint: {e}")

def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """清理旧的检查点文件，只保留最近的N个"""
    try:
        import glob
        pattern = os.path.join(checkpoint_dir, "model_epoch_*.pt")
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) > keep_last_n:
            # 按文件名排序（包含epoch编号）
            checkpoint_files.sort()
            files_to_remove = checkpoint_files[:-keep_last_n]
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"[CKPT] Removed old checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[CKPT][WARN] Failed to remove {file_path}: {e}")
                    
    except Exception as e:
        print(f"[CKPT][WARN] Failed to cleanup old checkpoints: {e}")

def _run_one_video(runner: RLRunner, cfg: Config, argv: List[str], epoch_id: int, epoch_total: int):
    # 自动推导 2-pass 基线
    stat_in = _find_arg(argv, "--stat-in", "")
    tp_path = _infer_twopass_from_stat_in(stat_in)
    if tp_path and not os.path.exists(tp_path):
        print(f"[MAIN][WARN] 2-pass baseline not found: {tp_path}")

    # 继承 FPS（用于统计 kbps）
    fps = _find_arg(argv, "--fps", "")
    if fps:
        try: cfg.fps = int(fps)
        except: pass

    # 把基线路径挂到 cfg（io_runner/reward 内部如果需要会从 cfg 读取）
    runner.cfg.twopass_log_path = tp_path

    # 设置 epoch（不改变其它回放/计数器逻辑）
    runner.set_epoch(idx=epoch_id, total=epoch_total, twopass_log_path=tp_path)

    # 启动编码器并进入服务循环
    enc = launch_encoder(cfg, argv)
    stop_evt = threading.Event()
    _ = start_monitor(enc, cfg, runner, stop_evt)
    runner.serve_loop(stop_evt)
    
    # 编码完成后打印统计信息
    two_pass_stats, rl_stats = _get_epoch_stats(cfg, tp_path, runner)
    _print_epoch_stats(epoch_id, two_pass_stats, rl_stats, runner)
    
    # 检查是否需要保存模型
    _save_checkpoint_if_needed(runner, cfg, epoch_id)

def _save_final_model(runner: RLRunner, cfg: Config, final_epoch: int):
    """保存最终训练完成的模型"""
    checkpoint_dir = getattr(cfg, 'checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, f"model_final_epoch_{final_epoch:04d}.pt")
    try:
        runner.agent.save(final_path)
        print(f"[CKPT] Final model saved to: {final_path}")
        
        # 也保存一个通用的最新模型
        latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
        runner.agent.save(latest_path)
        print(f"[CKPT] Latest model saved to: {latest_path}")
        
    except Exception as e:
        print(f"[CKPT][WARN] Failed to save final model: {e}")

def main():
    args = parse_args()
    cfg = Config(rl_dir=args.rl_dir, mode=args.mode)
    if args.encoder:
        cfg.encoder_path = args.encoder

    runner = RLRunner(cfg)

    start_ep = int(max(1, args.start_epoch))
    end_ep   = start_ep + int(max(1, args.epochs)) - 1

    if bool(args.use_dataset):
        # ===== 数据集模式：从 --dataset-inputs 构建所有 2-pass 命令 =====
        cmds = build_cmds_from_dataset(args, cfg)  # list[list[str]]
        if not cmds:
            print("[MAIN] no dataset commands built; exit.")
            return

        epoch_total = (end_ep - start_ep + 1) * max(1, len(cmds))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            # 简单按原顺序；如需打乱可在这里 random.shuffle(cmds.copy())
            for cmd_argv in cmds:
                _run_one_video(runner, cfg, cmd_argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
        print("[MAIN] dataset training finished.")
        # 训练结束后绘制最终图表
        if hasattr(runner, '_plot_training_curves'):
            runner._plot_training_curves()
        # 保存最终模型
        _save_final_model(runner, cfg, end_ep)
        print(f"[MAIN] Training completed. Check logs at: {getattr(cfg, 'log_dir', './logs')}")
    else:
        # ===== 单视频命令模式（支持 --epochs）=====
        epoch_total = (end_ep - start_ep + 1) * max(1, len(args.videos))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            # 如需每个 epoch 打乱顺序，可以：cmds = args.videos.copy(); random.shuffle(cmds)
            for cmd_bar in args.videos:
                argv = _split_cmd_bar(cmd_bar)
                _run_one_video(runner, cfg, argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
        print("[MAIN] single-video list finished.")
        # 训练结束后绘制最终图表
        if hasattr(runner, '_plot_training_curves'):
            runner._plot_training_curves()
        # 保存最终模型
        _save_final_model(runner, cfg, end_ep)
        print(f"[MAIN] Training completed. Check logs at: {getattr(cfg, 'log_dir', './logs')}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
dataset.py
-----------
数据集处理与 2-pass 命令构造（与 main.py 解耦）：

用法要点
- --dataset-inputs: 可传多个项；每项可以是【目录】【文件】【通配符路径】（如 E:/data/*.yuv）
  * 目录会被递归扫描（所有 .yuv/.YUV）
  * 文件若后缀为 .yuv 会被直接加入
  * 通配符会被展开并筛选出 .yuv 文件
- --stat-dir: 1-pass 日志目录（本模块只读不写）
- 1-pass 日志命名：{seq}_{enc}_pass1_{br}.log
- 2-pass 输出命名：{seq}_{enc}_pass2_{br}.log/.csv/.ivf
- {seq} = YUV 文件名（不含扩展名），例如 wedding_party_1920x1080_24
- {enc} = 编码器可执行文件短名：qav1enc_ori.exe -> ori；否则取可执行名去扩展名的 stem
- 分辨率/FPS 默认从文件名尾部解析：..._1920x1080_30.yuv / ..._1920x1080.yuv；解析不到用 fallback
"""

from __future__ import annotations

from typing import List, Tuple, Iterable
from pathlib import Path
import re
import os
import glob

__all__ = [
    "add_dataset_args",
    "build_cmds_from_dataset",
]


# -----------------------
# 参数注册
# -----------------------
def add_dataset_args(ap) -> None:
    """向 ArgumentParser 注入与数据集相关的命令行参数。"""
    ap.add_argument(
        "--dataset-inputs",
        type=str,
        nargs="*",
        default=[r"E:\python\DDPG\yuv_seq"],
        help="训练数据入口：可写【目录/文件/通配符】混合。目录将递归扫描所有 .yuv 文件。",
    )
    ap.add_argument(
        "--dataset-bitrates",
        type=int,
        nargs="*",
        default=[2125],
        help="码率列表，例如：2125 1700",
    )
    ap.add_argument(
        "--stat-dir",
        type=str,
        default=r"E:\python\DDPG\1pass_logs",
        help="1-pass 日志所在目录（本模块只读不写），按 {seq}_{enc}_pass1_{br}.log 拼接",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=r"E:\python\DDPG\outputs",
        help="2-pass 输出目录（ivf/csv/log 将写到这里）",
    )
    ap.add_argument(
        "--name-template",
        type=str,
        default="{seq}_{enc}",
        help="2-pass 输出命名前缀模板（通常保持默认即可）。可用变量：{seq}, {enc}",
    )
    ap.add_argument(
        "--fallback-res",
        type=str,
        default="1920x1080",
        help="当文件名解析不出分辨率时的默认值",
    )
    ap.add_argument(
        "--fallback-fps",
        type=int,
        default=24,
        help="当文件名解析不出帧率时的默认值",
    )
    ap.add_argument(
        "--extra",
        type=str,
        # 默认公共参数；命令行传入 --extra 会覆盖它
        default="--rc-mode|1|--preset|1|--keyint|225|--bframes|15|--threads|1|--parallel-frames|1|--score-max|50.5|--score-avg|40.5|--score-min|38.5",
        help=("公共附加参数（自动追加到每条命令后）。用 | 分隔键值，例如："
              "--rc-mode|1|--preset|1|--keyint|225|--bframes|15|--threads|4|--parallel-frames|5|"
              "--score-max|50.5|--score-avg|40.5|--score-min|38.5"),
    )


# -----------------------
# 内部小工具
# -----------------------
def _enc_short_name(encoder_path: str) -> str:
    """
    约定：qav1enc_xxx.exe -> xxx；否则取 stem（去扩展名的文件名）。
    仅用于命名，不影响实际编码器路径。
    """
    stem = Path(encoder_path).stem
    return stem[len("qav1enc_"):] if stem.startswith("qav1enc_") else stem


def _detect_res_fps_from_name(path: str, fallback_res: str, fallback_fps: int) -> Tuple[str, int]:
    """
    从文件名末尾解析：..._1920x1080_30.yuv 或 ..._1920x1080.yuv
    解析失败则使用 fallback。
    """
    name = Path(path).name
    m = re.search(r"_(\d{3,5}x\d{3,5})_(\d+)\.yuv$", name, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))
    m2 = re.search(r"_(\d{3,5}x\d{3,5})\.yuv$", name, re.IGNORECASE)
    if m2:
        return m2.group(1), fallback_fps
    return fallback_res, fallback_fps


def _split_extra(extra: str) -> List[str]:
    """将 --extra 的“|”分隔串拆成扁平 list[str]。"""
    if not extra:
        return []
    return [p for p in extra.split("|") if p]


def _is_yuv_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".yuv"


def _expand_dataset_inputs(inputs: Iterable[str]) -> List[Path]:
    """
    将 --dataset-inputs 展开为 YUV 文件列表：
    - 目录：递归扫描 .yuv
    - 文件：若后缀为 .yuv 则加入
    - 通配符：glob 展开后筛选 .yuv
    去重并按路径字符串排序，保证确定性。
    """
    found: List[Path] = []

    def add_path(p: Path):
        if _is_yuv_file(p):
            found.append(p.resolve())

    for entry in inputs:
        if not entry:
            continue
        p = Path(entry)
        if p.exists():
            if p.is_dir():
                # 递归扫描
                for sub in p.rglob("*"):
                    if _is_yuv_file(sub):
                        add_path(sub)
            else:
                add_path(p)
        else:
            # 按通配符尝试展开
            for g in glob.glob(entry, recursive=True):
                gp = Path(g)
                if _is_yuv_file(gp):
                    add_path(gp)

    # 去重 + 排序
    uniq = sorted({str(x): x for x in found}.values(), key=lambda x: str(x).lower())
    return uniq


# -----------------------
# 命令构造主函数
# -----------------------
def build_cmds_from_dataset(args, cfg) -> List[List[str]]:
    """
    根据 --dataset-* 与 cfg.encoder_path 生成 2-pass 命令列表（每条命令为 list[str]）：
      - 自动从 --dataset-inputs 的【目录/文件/通配符】中发现所有 .yuv
      - 只构造 PASS=2
      - --stat-in = {stat-dir}/{seq}_{enc}_pass1_{br}.log
      - --stat-out / --csv / --o 写到 out-dir，命名 {seq}_{enc}_pass2_{br}.log/.csv/.ivf
      - 分辨率/FPS 从文件名解析；失败则用 fallback-res/fallback-fps
      - 其它公共参数由 --extra 附加（可为空）
    """
    enc = _enc_short_name(getattr(cfg, "encoder_path", "encoder"))
    extra = _split_extra(getattr(args, "extra", ""))

    # 发现所有 yuv
    yuv_files = _expand_dataset_inputs(getattr(args, "dataset_inputs", []))
    if not yuv_files:
        print("[dataset] WARN: 未在 --dataset-inputs 中发现任何 .yuv 文件")
        return []

    cmds: List[List[str]] = []

    for yuv_path in yuv_files:
        in_yuv = str(yuv_path)
        seq = yuv_path.stem  # 例：wedding_party_1920x1080_24
        res, fps = _detect_res_fps_from_name(in_yuv, args.fallback_res, args.fallback_fps)

        # 2-pass 输出前缀（允许自定义模板）
        name_base = args.name_template.format(seq=seq, enc=enc)

        for br in getattr(args, "dataset_bitrates", [2125]):
            brs = str(br).replace(".", "_")

            # 1-pass 日志路径：{seq}_{enc}_pass1_{br}.log（位于 stat-dir）
            stat_in = os.path.join(args.stat_dir, f"{seq}_qav1enc_pass1_{brs}.log")

            # 2-pass 输出路径（位于 out-dir）
            stat_out = os.path.join(args.out_dir, f"{name_base}_pass2_{brs}.log")
            ivf_out  = os.path.join(args.out_dir, f"{name_base}_pass2_{brs}.ivf")
            csv_out  = os.path.join(args.out_dir, f"{name_base}_pass2_{brs}.csv")

            parts = [
                "--input", in_yuv,
                "--input-res", res,
                "--frames", "0",
                "--o", ivf_out,
                "--csv", csv_out,
                "--bitrate", str(br),
                "--pass", "2",
                "--stat-in", stat_in,
                "--stat-out", stat_out,
                "--fps", str(fps),
            ]
            parts += extra
            cmds.append(parts)

    print(f"[dataset] 收集到 {len(yuv_files)} 个 YUV，共展开 {len(cmds)} 条 2-pass 命令。")
    return cmds

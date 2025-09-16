# -*- coding: utf-8 -*-
import re
from typing import Dict, List

class TwoPassBaseline:
    """
    读取 2-pass 日志，构建 {POC -> {type,bits,psnr,vmaf}} 的映射。
    规则（按你的要求）：
      - 同一 POC 若同时出现非 O 与 O：
          bits    = 非 O 行的 bits
          vmaf    = O 行的 vmaf（若有且 >0；否则保留非 O 行的 vmaf）
          type    = 非 O 的类型（若没有非 O，则强制记为 'P'）
      - 统计时仅对 P/B 帧聚合（排除 I/K/O）
    """

    def __init__(self, path: str, *, poc_base: int = 0):
        """
        poc_base: 你的样例是行首两个整数，前者即 POC，通常从 0 起。
                  这里默认 0；若你的日志以 1 起，可传入 poc_base=1。
        """
        self.path = path
        self.poc_base = int(poc_base)  # 0 or 1
        self.map: Dict[int, Dict[str, float]] = {}
        self._parse()

    # ---------- 内部：解析 ----------
    def _parse(self):
        # 典型行格式（你的样例）：   "<poc> <doc> <type> ... vmaf <f> bits <d> ..."
        re_head = re.compile(r'^\s*(-?\d+)\s+(-?\d+)\s+([A-Za-z])\b')   # 捕获 poc, doc, type
        re_poc_tok  = re.compile(r'\bpoc\b\s*[:=]\s*(-?\d+)', re.I)     # 兜底：poc=xx
        re_type_tok = re.compile(r'\btype\b\s*[:=]\s*([OPBKI])\b', re.I)
        re_type_any = re.compile(r'\b([OPBKIb])\b', re.I)               # 再兜底：独立的单字母
        re_bits = re.compile(r'\bbits?\b\s*(?:[:=]\s*)?(\d+)', re.I)
        re_vmaf = re.compile(r'\bvmaf\b\s*(?:[:=]\s*)?([0-9]+(?:\.[0-9]+)?)', re.I)
        re_psnr = re.compile(r'\bpsnr\b\s*(?:[:=]\s*)?([0-9]+(?:\.[0-9]+)?)', re.I)  # 若日志含 psnr

        with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                poc = None
                typ = None

                m = re_head.match(line)
                if m:
                    poc = int(m.group(1))
                    typ = m.group(3).upper()
                else:
                    mp = re_poc_tok.search(line)
                    if mp:
                        poc = int(mp.group(1))
                    mt = re_type_tok.search(line) or re_type_any.search(line)
                    if mt:
                        typ = mt.group(1).upper()

                if poc is None:
                    # 再兜底：行首第一个整数作为 poc
                    m0 = re.match(r'^\s*(-?\d+)\b', line)
                    if m0:
                        poc = int(m0.group(1))
                if poc is None:
                    continue

                # 标准化 POC：若日志从 0 计，poc_base=0 则不偏移；若从 1 计，poc_base=1 可保持一致。
                # 你之前的旧实现会做 += (1 - poc_base)，这里不做额外偏移，直接用 poc。
                # 如需对齐旧代码：poc = poc + (1 - self.poc_base)
                if self.poc_base in (0, 1):
                    # 不做偏移，保留你样例中的自然 POC
                    pass

                # 类型兜底（小写 b -> B）
                if typ is None:
                    # 如果没解析出类型，可以不更新；后续合并时再决定
                    pass
                else:
                    if typ == 'B' or typ == 'b':
                        typ = 'B'
                    elif typ not in ('O', 'P', 'B', 'I', 'K'):
                        # 非法/未知类型，跳过该行的类型更新
                        typ = None

                # 数值
                bits = None
                vmaf = None
                psnr = None

                mb = re_bits.search(line)
                if mb:
                    try:
                        bits = int(mb.group(1))
                    except Exception:
                        pass

                mv = re_vmaf.search(line)
                if mv:
                    try:
                        vmaf = float(mv.group(1))
                    except Exception:
                        pass

                mp = re_psnr.search(line)
                if mp:
                    try:
                        psnr = float(mp.group(1))
                    except Exception:
                        pass

                rec = self.map.get(poc)
                if rec is None:
                    # 为每个 POC 维护一次合并记录：
                    rec = {
                        "type": "",          # 最终类型（非 O；若仅出现 O 则设为 'P'）
                        "bits": 0,           # 取非 O 行
                        "psnr": 0.0,         # 若日志有 psnr，取非 O 行；否则保持 0
                        "vmaf": 0.0,         # 最终 vmaf（优先取 O 行）
                        "_ov_vmaf": 0.0,     # 仅暂存 overlay 的 vmaf
                        "_has_nonO": False,  # 是否出现过非 O 行
                    }
                    self.map[poc] = rec

                # 合并逻辑
                if typ == 'O':
                    # overlay：只关心 vmaf
                    if vmaf is not None and vmaf > rec.get("_ov_vmaf", 0.0):
                        rec["_ov_vmaf"] = vmaf
                elif typ in ('P', 'B', 'I', 'K', None):
                    # 非 O（或未知）：更新 bits/psnr/vmaf/type
                    if bits is not None:
                        rec["bits"] = int(bits)
                    if psnr is not None:
                        rec["psnr"] = float(psnr)
                    if vmaf is not None and vmaf > 0.0:
                        # 非 O 行若也给了 vmaf，就先记上；稍后会被 _ov_vmaf 覆盖
                        rec["vmaf"] = float(vmaf)
                    # 更新类型（优先保留“更有意义”的类型）
                    if typ is not None:
                        # 若之前没类型，直接赋；否则按优先级 P > B > I > K
                        prev = rec.get("type", "")
                        if not prev:
                            rec["type"] = typ
                        else:
                            pri = {'P': 3, 'B': 2, 'I': 1, 'K': 1}
                            if pri.get(typ, 0) > pri.get(prev, 0):
                                rec["type"] = typ
                    rec["_has_nonO"] = True
                else:
                    # 其它不处理
                    pass

        # 结束后做一次收尾：把 overlay vmaf 覆盖到最终 vmaf；修正类型非 O
        for poc, rec in self.map.items():
            ov = float(rec.get("_ov_vmaf", 0.0) or 0.0)
            if ov > 0.0:
                rec["vmaf"] = ov
            # 类型不能为 O；若没有非 O 类型，强制当成 P
            if not rec.get("_has_nonO", False):
                # 没见到非 O 行，但又有 O 行（或者啥也没有），按你的要求记为 P
                rec["type"] = rec.get("type") if rec.get("type") in ('P', 'B', 'I', 'K') else 'P'
            elif rec.get("type", "") == 'O' or not rec.get("type", ""):
                rec["type"] = 'P'  # 容错
            # 清理临时键
            rec.pop("_ov_vmaf", None)
            rec.pop("_has_nonO", None)

    # ---------- 供 RL 使用：按 POC 列表聚合 ----------
    def mg_stats_from_pocs(self, poc_list: List[int]) -> Dict[str, float]:
        """
        基于给定的 POC 列表统计参考均值：
          - 只统计 P/B（排除 I/K/O）
          - 返回 bits_total / bit_avg / vmaf_avg / count
        """
        if not poc_list:
            return {"bits_total": 0.0, "bit_avg": 0.0, "vmaf_avg": 0.0, "count": 0}
        total_bits = 0.0
        total_vmaf = 0.0
        cnt = 0
        for p in poc_list:
            rec = self.map.get(int(p))
            if not rec:
                continue
            t = (rec.get("type", "") or "").upper()
            if t in ("I", "K", "O"):
                continue  # 只统计显示帧 P/B
            total_bits += float(rec.get("bits", 0.0) or 0.0)
            total_vmaf += float(rec.get("vmaf", 0.0) or 0.0)
            cnt += 1
        bit_avg = (total_bits / cnt) if cnt > 0 else 0.0
        vmaf_avg = (total_vmaf / cnt) if cnt > 0 else 0.0
        return {"bits_total": total_bits, "bit_avg": bit_avg, "vmaf_avg": vmaf_avg, "count": cnt}

    # ---------- 可选：保留旧接口（以“末尾 POC + 长度”推段） ----------
    def mg_stats(self, poc_end: int, mg_size: int = 16) -> Dict:
        """
        若你仍有场景需要“以末尾 POC + 长度”来估计 miniGOP，就用这个；
        但在你现在的 RL 流程里，建议优先使用 mg_stats_from_pocs(poc_list)。
        """
        end_ = int(poc_end)
        start_ = max(0, end_ - int(mg_size) + 1)

        total_bits = 0.0
        total_psnr = 0.0
        total_vmaf = 0.0
        count = 0

        for p in range(start_, end_ + 1):
            rec = self.map.get(p)
            if not rec:
                continue
            t = (rec.get("type", "") or "").upper()
            if t in ("I", "K", "O"):
                continue
            total_bits += float(rec.get("bits", 0.0) or 0.0)
            total_psnr += float(rec.get("psnr", 0.0) or 0.0)
            total_vmaf += float(rec.get("vmaf", 0.0) or 0.0)
            count += 1

        bits_avg = total_bits / max(1, count)
        psnr_avg = total_psnr / max(1, count)
        vmaf_avg = total_vmaf / max(1, count)
        return {
            "start": start_, "end": end_, "count": count,
            "bits_total": float(total_bits), "bits_avg": float(bits_avg),
            "psnr_avg": float(psnr_avg), "vmaf_avg": float(vmaf_avg),
        }

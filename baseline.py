# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple

# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple

class TwoPassBaseline:
    """
    读取 2-pass 日志，构建 {POC -> {type,bits,psnr}} 的基线映射，并按 DOC(编码顺序)
    以“当前帧是本 miniGOP 的末尾显示帧”的假设计算 miniGOP 参考统计。
    """
    def __init__(self, path: str, *, poc_base: int = 1):
        self.path = path
        self.poc_base = int(poc_base)  # POC 起始：通常 0 或 1
        self.map: Dict[int, Dict[str, float]] = {}
        self._parse()

    # -------- 解析：避免 O(overlay) 覆盖有效记录，按类型优先级保留更有用的一条 --------
    @staticmethod
    def _type_priority(t: str) -> int:
        """
        帧类型优先级：P/B(含小写 b) 最高，其次 I/K，其他再次，O(overlay) 最低且会被跳过。
        """
        t = (t or "").upper()
        if t in ("P", "B", "b"):  # 你日志里若只有 P/B 就够了
            return 3
        if t in ("I", "K"):
            return 2
        if t in ("O",):
            return 0
        return 1

    def _parse(self):
        pat_bits  = re.compile(r"\bbits\b\s*=?\s*([0-9]+)")
        pat_score = re.compile(r"\bscore\b\s*=?\s*([0-9]+(?:\.[0-9]+)?)")

        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if not parts or not parts[0].isdigit():
                    continue
                try:
                    poc = int(parts[0])
                except Exception:
                    continue

                ftype = parts[2].upper() if len(parts) >= 3 else ""
                m_bits  = pat_bits.search(s)
                m_score = pat_score.search(s)
                if not m_bits or not m_score:
                    continue

                # 直接跳过 overlay（防止覆盖）
                if ftype == "O":
                    continue

                bits = int(m_bits.group(1))
                psnr = float(m_score.group(1))
                prio = self._type_priority(ftype)

                prev = self.map.get(poc)
                if prev is None:
                    self.map[poc] = {"type": ftype, "bits": bits, "psnr": psnr}
                else:
                    # 只有当新记录更“有用”时才覆盖（例如 P/B 覆盖 I）
                    if self._type_priority(prev.get("type", "")) < prio:
                        self.map[poc] = {"type": ftype, "bits": bits, "psnr": psnr}
                    # 如果优先级相同，默认保留先到者；如需“后到者优先”，可改成直接覆盖。

    # -------- miniGOP 区间：以 DOC 为准，当前帧 POC 为 end，向前数 mg_size 个显示帧 --------
    def _mg_range_for_poc(self, poc: int, mg_size: int = 16) -> Tuple[int, int]:
        """
        以编码顺序(DOC)为准：
          end = 当前帧的 POC
          start = end - mg_size
        例如：poc=16, mg_size=15 -> [1, 16]
        """
        end_ = int(poc)
        start_ = max(self.poc_base, end_ - int(mg_size))
        return start_, end_

    # -------- miniGOP 参考统计：仅统计 P/B，排除 O 与 I/K --------
    def mg_stats(self, poc: int, mg_size: int = 16) -> Dict:
        """
        返回以当前 `poc` 作为 miniGOP 末尾的参考统计（只计入 P/B）：
        {
            'start': int, 'end': int, 'count': int,
            'bits_total': int, 'bits_avg': float,
            'psnr_avg': float
        }
        """
        start_, end_ = self._mg_range_for_poc(poc, mg_size)

        total_bits = 0
        total_psnr = 0.0
        count = 0

        for p in range(start_, end_ + 1):  # 闭区间
            rec = self.map.get(p)
            if not rec:
                continue
            t = (rec.get("type", "") or "").upper()
            if t in ("O", "I", "K"):
                continue  # 只统计 P/B
            total_bits += int(rec.get("bits", 0))
            total_psnr += float(rec.get("psnr", 0.0))
            count += 1

        bits_avg = total_bits / max(1, count)
        psnr_avg = total_psnr / max(1, count)
        return {
            "start": start_, "end": end_, "count": count,
            "bits_total": int(total_bits), "bits_avg": float(bits_avg),
            "psnr_avg": float(psnr_avg),
        }


"""
通用清洗逻辑, 供各数据集清洗脚本调用。

每个数据集的清洗脚本负责提取字段, 调用 BaseCleaner 完成实际过滤与规范化
"""

import re
import unicodedata

from pydantic import BaseModel, Field


class CleanConfig(BaseModel):
    min_chars: int = Field(default=256, description="文档最少字符数")
    max_chars: int = Field(default=100_000, description="文档最多字符数")
    min_chinese_ratio: float = Field(default=0.3, description="中文字符比例过滤（中文 / 所有非空白字符）")
    max_duplicate_line_ratio: float = Field(default=0.3, description="重复行过滤：重复行占总行数的比例上限")
    max_repeated_char_run: int = Field(default=20, description="重复字符过滤：单个字符连续出现次数上限")
    max_noise_char_ratio: float = Field(default=0.02, description="特殊字符过滤：非打印字符、控制字符占总字符比例上限")
    unicode_normalize: bool = Field(default=True, description="是否进行 Unicode NFC 规范化")
    block_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(请关注|扫码关注|微信公众号|点击链接|广告)",
            r"(版权所有|copyright\s*©)",
        ]
    )


class CleanResult(BaseModel):
    passed: bool
    text: str
    fail_reason: str = ""


class BaseCleaner:
    # 半角 → 全角标点映射表
    # 句号单独用正则处理，以避免误转小数点和省略号
    _PUNCT_TABLE: dict[int, str] = str.maketrans(
        {
            ord(","): "，",
            ord("?"): "？",
            ord("!"): "！",
            ord(":"): "：",
            ord(";"): "；",
            ord("("): "（",
            ord(")"): "）",
        }
    )
    # 省略号优先于句号处理；句号仅在非数字上下文中转换
    _ELLIPSIS_RE = re.compile(r"\.{2,}")
    _DOT_RE = re.compile(r"(?<!\d)\.(?!\d)")

    def __init__(self, config: CleanConfig | None = None):
        self.cfg = config or CleanConfig()
        self._block_re = (
            re.compile("|".join(self.cfg.block_patterns), re.IGNORECASE) if self.cfg.block_patterns else None
        )
        self._repeated_char_re = re.compile(r"(.)\1{" + str(self.cfg.max_repeated_char_run) + r",}")

    def clean(self, text: str) -> CleanResult:
        """
        对单条文本执行全部清洗步骤。
        步骤顺序：规范化 → 结构性清理 → 标点规范化 → 各项过滤。
        """
        # Unicode 规范化
        if self.cfg.unicode_normalize:
            text = unicodedata.normalize("NFC", text)

        # 结构性清理（空白、控制字符）
        text = self._normalize_whitespace(text)

        # 标点规范化（半角 → 全角）
        text = self._normalize_punctuation(text)

        # 引号规范化（半角直引号 → 全角弯引号）
        text = self._normalize_quotes(text)

        # 过滤检查（顺序：快速 → 慢速）
        checks = [
            self._check_length,
            self._check_noise_chars,
            self._check_repeated_chars,
            self._check_chinese_ratio,
            self._check_duplicate_lines,
            self._check_block_patterns,
        ]
        for check in checks:
            result = check(text)
            if result is not None:  # result 非 None 表示未通过
                return CleanResult(passed=False, text="", fail_reason=result)

        return CleanResult(passed=True, text=text)

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """
        将半角直引号转为全角弯引号。

        策略：状态追踪，奇数次出现 → 开引号，偶数次出现 → 闭引号。
          "  →  " （开）/  " （闭）
          '  →  ' （开）/  ' （闭）
        """
        # 双引号
        buf = []
        open_double = False
        for ch in text:
            if ch == '"':
                buf.append("\u201c" if not open_double else "\u201d")  # " / "
                open_double = not open_double
            else:
                buf.append(ch)
        text = "".join(buf)

        # 单引号
        buf = []
        open_single = False
        for ch in text:
            if ch == "'":
                buf.append("\u2018" if not open_single else "\u2019")  # ' / '
                open_single = not open_single
            else:
                buf.append(ch)
        return "".join(buf)

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """
        将半角标点统一转为全角标点

        特殊处理：
        - `...` / `....` 等省略号 → `……`
        - `.` 仅在非数字上下文中转为 `。`，保留小数点（如 `3.14`）
        - `\n`（换行）保持不变，不转为全角
        """
        # 省略号优先处理（避免后续逐个句号转换）
        text = cls._ELLIPSIS_RE.sub("……", text)

        # 字符级映射（空格、逗号、括号等）
        text = text.translate(cls._PUNCT_TABLE)

        # 句号：仅转换非数字上下文中的 .
        text = cls._DOT_RE.sub("。", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        - 去除零宽字符、BOM 等不可见字符
        - 压缩连续空行（保留最多一个空行）
        - strip 首尾空白
        """
        # 零宽字符 / BOM
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
        # 压缩连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 每行末尾多余空格
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()

    def _check_length(self, text: str) -> str | None:
        n = len(text)
        if n < self.cfg.min_chars:
            return f"too_short: {n} < {self.cfg.min_chars}"
        if n > self.cfg.max_chars:
            return f"too_long: {n} > {self.cfg.max_chars}"
        return None

    def _check_noise_chars(self, text: str) -> str | None:
        """过滤控制字符、替换字符等噪声字符占比过高的文档。"""
        if not text:
            return None
        noise = sum(1 for ch in text if unicodedata.category(ch) in ("Cc", "Cs", "Co", "Cn") and ch not in ("\n", "\t"))
        ratio = noise / len(text)
        if ratio > self.cfg.max_noise_char_ratio:
            return f"noise_chars: {ratio:.3f} > {self.cfg.max_noise_char_ratio}"
        return None

    def _check_repeated_chars(self, text: str) -> str | None:
        """过滤单个字符大量连续重复的文档（如 '啊啊啊啊啊…'）。"""
        if self._repeated_char_re.search(text):
            return f"repeated_chars: run >= {self.cfg.max_repeated_char_run}"
        return None

    def _check_chinese_ratio(self, text: str) -> str | None:
        """过滤中文字符比例过低的文档。"""
        non_space = [ch for ch in text if not ch.isspace()]
        if not non_space:
            return "empty_after_strip"
        chinese = sum(1 for ch in non_space if "\u4e00" <= ch <= "\u9fff")
        ratio = chinese / len(non_space)
        if ratio < self.cfg.min_chinese_ratio:
            return f"chinese_ratio: {ratio:.3f} < {self.cfg.min_chinese_ratio}"
        return None

    def _check_duplicate_lines(self, text: str) -> str | None:
        """过滤大量重复行的文档（常见于模板页、列表爬取噪声）。"""
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        unique = len(set(lines))
        dup_ratio = 1.0 - unique / len(lines)
        if dup_ratio > self.cfg.max_duplicate_line_ratio:
            return f"duplicate_lines: {dup_ratio:.3f} > {self.cfg.max_duplicate_line_ratio}"
        return None

    def _check_block_patterns(self, text: str) -> str | None:
        """过滤命中黑名单关键词的文档。"""
        if self._block_re and self._block_re.search(text):
            return "block_pattern_matched"
        return None

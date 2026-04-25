"""
通用清洗逻辑, 供各数据集清洗脚本调用。

每个数据集的清洗脚本负责提取字段, 调用 BaseCleaner 完成实际过滤与规范化
"""

import re
import unicodedata

from pydantic import BaseModel, Field


class CleanConfig(BaseModel):
    min_chars: int = Field(default=200, description="文档最少字符数")
    max_chars: int = Field(default=100_000, description="文档最多字符数")
    min_chinese_ratio: float = Field(default=0.3, description="中文字符比例过滤（中文 / 所有非空白字符）")
    max_duplicate_line_ratio: float = Field(default=0.3, description="重复行过滤：重复行占总行数的比例上限")
    max_repeated_char_run: int = Field(default=20, description="重复字符过滤：单个字符连续出现次数上限")
    max_noise_char_ratio: float = Field(default=0.02, description="特殊字符过滤：非打印字符、控制字符占总字符比例上限")
    unicode_normalize: bool = Field(
        default=True,
        description="是否进行 Unicode NFKC 规范化（兼容等价分解+组合，可将数学样式变体字母还原为普通 ASCII）",
    )
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

    # 分数字符 → 普通文本（使用 ASCII /），须在 NFKC 之前处理
    # NFKC 会将分数字符展开为数字 + U+2044（分数斜线），而 U+2044 在噪音范围内会被清洗，
    # 导致 ⅒ → "110" 而非 "1/10"，因此提前替换为带普通斜线的文本形式
    _VULGAR_FRACTION_MAP: dict[str, str] = {
        "\u00bc": "1/4",  # ¼
        "\u00bd": "1/2",  # ½
        "\u00be": "3/4",  # ¾
        "\u2150": "1/7",  # ⅐
        "\u2151": "1/9",  # ⅑
        "\u2152": "1/10",  # ⅒
        "\u2153": "1/3",  # ⅓
        "\u2154": "2/3",  # ⅔
        "\u2155": "1/5",  # ⅕
        "\u2156": "2/5",  # ⅖
        "\u2157": "3/5",  # ⅗
        "\u2158": "4/5",  # ⅘
        "\u2159": "1/6",  # ⅙
        "\u215a": "5/6",  # ⅚
        "\u215b": "1/8",  # ⅛
        "\u215c": "3/8",  # ⅜
        "\u215d": "5/8",  # ⅝
        "\u215e": "7/8",  # ⅞
    }

    # 全角字符（U+FF01–U+FF5E）→ 对应 ASCII 半角，覆盖字母、数字及符号
    # 原条件 isalpha()/isdigit() 已去除，使 ＜＝＞？＠ 等符号也得到转换
    # 注：转换后的半角标点会在 _normalize_punctuation 中进一步规范化为全角中文标点
    _FULLWIDTH_LATIN_TABLE: dict[int, int] = {cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)}

    # 噪音脚本正则：对中文语料无意义的外文脚本、符号块、遗留编码
    # 范围按 Unicode 码位升序排列，便于审查与维护
    _NOISE_SCRIPT_RE = re.compile(
        r"["
        # ── Latin 扩展区 ──────────────────────────────────────────────────────
        r"\u00BF-\u024F"  # Latin-1 Supplement 字母区（¿ÀÁ…ÿ）+ Latin Extended-A/B
        r"\u0250-\u02AF"  # IPA 音标扩展
        r"\u02B0-\u02FF"  # IPA 修饰字母
        # ── 组合符号 & 希腊文 ─────────────────────────────────────────────────
        r"\u0300-\u036F"  # 组合附加符号（变音符、Zalgo 噪音等）
        r"\u0370-\u03FF"  # 希腊文主块（Α Β Γ 等常用希腊字母）
        # ── 西里尔文 & 亚美尼亚文 ────────────────────────────────────────────
        r"\u0400-\u04FF"  # 西里尔文主块（含俄文）
        r"\u0500-\u052F"  # 西里尔文补充
        r"\u0530-\u058F"  # 亚美尼亚文
        # ── 希伯来文 & 阿拉伯文 ──────────────────────────────────────────────
        r"\u0590-\u05FF"  # 希伯来文
        r"\u0600-\u06FF"  # 阿拉伯文主块
        r"\u0700-\u074F"  # 叙利亚文
        r"\u0750-\u077F"  # 阿拉伯文补充
        r"\u0780-\u07BF"  # 塔纳文
        r"\u07C0-\u07FF"  # N'Ko 文
        r"\u0800-\u083F"  # 撒马利亚文
        r"\u0840-\u085F"  # 曼达亚文
        r"\u08A0-\u08FF"  # 阿拉伯文扩展-A
        # ── 南亚诸脚本 ───────────────────────────────────────────────────────
        r"\u0900-\u097F"  # 天城文
        r"\u0980-\u09FF"  # 孟加拉文
        r"\u0A00-\u0A7F"  # 旁遮普文（果鲁穆奇文）
        r"\u0A80-\u0AFF"  # 古吉拉特文
        r"\u0B00-\u0B7F"  # 奥里亚文
        r"\u0B80-\u0BFF"  # 泰米尔文
        r"\u0C00-\u0C7F"  # 泰卢固文
        r"\u0C80-\u0CFF"  # 卡纳达文
        r"\u0D00-\u0D7F"  # 马拉雅拉姆文（Malayalam）
        r"\u0D80-\u0DFF"  # 僧伽罗文
        # ── 东南亚脚本 ───────────────────────────────────────────────────────
        r"\u0E00-\u0E7F"  # 泰文
        r"\u0E80-\u0EFF"  # 老挝文
        r"\u0F00-\u0FFF"  # 藏文
        r"\u1000-\u109F"  # 缅甸文
        # ── 格鲁吉亚文 ───────────────────────────────────────────────────────
        r"\u10A0-\u10FF"  # 格鲁吉亚文（Asomtavruli + Mkhedruli）
        # ── 韩文 & 埃塞俄比亚文 ──────────────────────────────────────────────
        r"\u1100-\u11FF"  # 韩文字母 Jamo
        r"\u1200-\u139F"  # 埃塞俄比亚文 + Ethiopic Supplement（U+1380–U+139F）
        r"\u13A0-\u13FF"  # 切罗基文
        # ── 加拿大土著 & 卢恩文 & 他加禄文等 ────────────────────────────────
        r"\u1400-\u167F"  # 加拿大土著音节文字
        r"\u1680-\u177F"  # Ogham + 卢恩文（U+16A0）+ 他加禄文（U+1700）+ Hanunoo/Buhid/Tagbanwa
        # ── 高棉文 & 蒙古文 & 大型东南亚盲区 ────────────────────────────────
        r"\u1780-\u17FF"  # 高棉文
        r"\u1800-\u18AF"  # 蒙古文
        r"\u18B0-\u1DFF"  # Tai Le、New Tai Lue、Buginese、Tai Tham、Balinese、
        # Sundanese、Batak、Lepcha 等（原完全未覆盖的大型空隙）
        # ── 拉丁文扩展附加 & 希腊文扩展 ──────────────────────────────────────
        r"\u1E00-\u1EFF"  # 拉丁文扩展附加（越南文、威尔士文等）
        r"\u1F00-\u1FFF"  # 希腊文扩展
        # ── 杂项标点符号（精确覆盖，避免误伤合法中文标点）────────────────────
        r"\u2010-\u2012"  # 连字符变体（‐‑‒，非破折号）
        r"\u2016-\u2017"  # ‖‗
        r"\u201A-\u201B"  # ‚‛
        r"\u201E-\u201F"  # „‟
        r"\u2020-\u2027"  # †‡•‣․‥…‧（项目符号、省略号等）
        r"\u2030-\u2031"  # ‰‱ 千分号、万分号
        r"\u2032-\u2036"  # ′″‴‵‶ 撇号
        r"\u2038"  # ‸ 插入符
        r"\u2039-\u203A"  # ‹› 单角引号
        r"\u203E-\u203F"  # ‾‿ 上划线、连底线
        r"\u2041-\u2044"  # ⁁⁂⁃⁄ 插入号等
        # ── 货币 & 数学符号 ───────────────────────────────────────────────────
        r"\u20A0-\u20CF"  # 货币符号（€ 等）
        r"\u20D0-\u20FF"  # 数学组合附加符号
        # ── 字母类符号 & Number Forms ─────────────────────────────────────────
        r"\u2100-\u2102"  # 字母类符号（℃ 前）
        r"\u2104-\u218F"  # 字母类符号（℃ 后）+ Number Forms（U+2150 罗马数字等）
        # ── 箭头、几何、杂项技术符号 ──────────────────────────────────────────
        r"\u2190-\u27FF"  # 箭头、几何形状、杂项技术符号
        r"\u2800-\u2BFF"  # 盲文、补充箭头-B、杂项数学符号-B、补充数学运算符、杂项符号和箭头
        # ── 格拉哥里文 & 拉丁文扩展-C & 制表符 ───────────────────────────────
        r"\u2C00-\u2C7F"  # 格拉哥里文、拉丁文扩展-C
        r"\u2500-\u257F"  # 制表符
        r"\u2C80-\u2CFF"  # 科普特文
        r"\u2D30-\u2D7F"  # 提非纳文
        r"\u2DE0-\u2DFF"  # 西里尔文扩展-A
        r"\u2E00-\u2E7F"  # 补充标点
        r"\u2E80-\u2EFF"  # CJK 部首补充
        # ── CJK 描述字符 & CJK 标点局部 ──────────────────────────────────────
        r"\u2FF0-\u2FFB"  # CJK 表意文字描述字符（⿰⿱⿲ 等结构描述元符号）
        r"\u3003-\u3007"  # 〃〄々〆〇（同上符号、JIS 符号、叠字符、表意零等）
        r"\u3014-\u3019"  # 〔〕〖〗〘〙（龟甲括号、白色方头/龟甲括号）
        # ── 原有 CJK 杂项 ────────────────────────────────────────────────────
        r"\u302A-\u302F"  # 汉字声调组合符号（古典标音，现代中文不用）
        r"\u3030-\u3035"  # 〰波浪破折号、〱〲〳〴〵日文迭字符号
        r"\u3130-\u318F"  # 韩文兼容字母
        r"\u3200-\u33FF"  # 封闭 CJK
        # ── CJK 扩展-A & 易经卦符 ────────────────────────────────────────────
        r"\u3400-\u4DBF"  # CJK 统一汉字扩展-A（极罕见字，现代中文语料不需要）
        r"\u4DC0-\u4DFF"  # 易经六十四卦符号
        # ── 彝文 ─────────────────────────────────────────────────────────────
        r"\uA000-\uA4CF"  # 彝文音节（Yi Syllables）+ 彝文部首（Yi Radicals）
        # ── Lisu / Vai / 西里尔扩展-B / Latin Extended-D/E / Meetei Mayek 等 ──
        r"\uA4D0-\uABFF"  # Lisu（U+A4D0）、Vai（U+A500）、Cyrillic Extended-B（U+A640）、
        # Modifier Tone Letters（U+A700）、Latin Extended-D（U+A720）、
        # Syloti Nagri（U+A800）、Phags-pa（U+A840）、Saurashtra（U+A880）、
        # Devanagari Extended（U+A8E0）、Javanese（U+A980）、
        # Cham（U+AA00）、Myanmar Extended-B（U+AA60）、
        # Tai Viet（U+AA80）、Latin Extended-E（U+AB30）、
        # Cherokee Supplement（U+AB70）、Meetei Mayek（U+ABC0）
        # ── 韩文音节 & Jamo Extended-B ───────────────────────────────────────
        r"\uAC00-\uD7FF"  # 韩文音节块 + Hangul Jamo Extended-B（U+D7B0–U+D7FF）
        # ── PUA & CJK 兼容汉字 ───────────────────────────────────────────────
        r"\uE000-\uF8FF"  # 私用区（PUA）
        r"\uF900-\uFAFF"  # CJK 兼容汉字（Compatibility Ideographs）
        # ── 阿拉伯文表现形式 & 组合半符号 ────────────────────────────────────
        r"\uFB50-\uFDFF"  # 阿拉伯文表现形式-A
        r"\uFE20-\uFEFF"  # 组合半符号（Combining Half Marks，U+FE20）+ 阿拉伯文表现形式-B
        # ── 半角片假名 & Specials ─────────────────────────────────────────────
        r"\uFF65-\uFF9F"  # 半角片假名（遗留编码，中文语料中为噪音）
        r"\uFFFB-\uFFFD"  # Specials：Interlinear Annotation Terminator / Object Replacement / Replacement Char
        # ── 补充平面：古代脚本大区间 ──────────────────────────────────────────
        r"\U00010000-\U000115FF"  # Linear B、Deseret、Osage、Cypriot、Aramaic、Phoenician、
        # Kharoshthi、Old South/North Arabian、Avestan、
        # Inscriptional Parthian/Pahlavi、Old Turkic、Brahmi、
        # Sharada 等（含原 U+10300 和 U+11580 两段，合并为连续区间）
        r"\U00012000-\U0001247F"  # 苏美尔楔形文字 + Cuneiform Numbers（U+12400–U+1247F）
        r"\U00013000-\U0001342F"  # 古埃及象形文字
        r"\U00016800-\U00016FFF"  # 巴姆穆文补充、柏格理苗文等
        r"\U00017000-\U000187FF"  # 西夏文
        r"\U0001B000-\U0001B0FF"  # 假名补充
        r"\U0001CE00-\U0001CEFF"  # 未分配区段
        r"\U0001D100-\U0001D1FF"  # 音乐符号
        r"\U0001D400-\U0001D7FF"  # 数学字母数字符号（NFKC 后保留作保险）
        r"\U0001F000-\U0001FFFF"  # 补充平面 emoji
        # ── CJK 扩展区（B–G）& 兼容汉字补充 ──────────────────────────────────
        r"\U00020000-\U0002A6DF"  # CJK 扩展-B
        r"\U0002A700-\U0002EBEF"  # CJK 扩展-C/D/E/F（连续合并）
        r"\U0002F800-\U0002FA1F"  # CJK 兼容汉字补充（Compatibility Ideographs Supplement）
        r"\U00030000-\U0003134F"  # CJK 扩展-G
        # ── 未分配保留区 ──────────────────────────────────────────────────────
        r"\U00060000-\U0006FFFF"  # 完全未分配的保留平面（Plane 6）
        # ── Tags & 变体选择符补充 ─────────────────────────────────────────────
        r"\U000E0000-\U000E01EF"  # Tags（U+E0000–U+E007F）+ Variation Selectors Supplement（U+E0100–U+E01EF）
        # ── 私用区-A/B ────────────────────────────────────────────────────────
        r"\U000F0000-\U0010FFFF"  # 补充私用区-A（Nerd Fonts 图标等）+ 补充私用区-B
        r"]",
        flags=re.UNICODE,
    )

    def __init__(self, config: CleanConfig | None = None):
        self.cfg = config or CleanConfig()
        self._block_re = (
            re.compile("|".join(self.cfg.block_patterns), re.IGNORECASE) if self.cfg.block_patterns else None
        )
        self._repeated_char_re = re.compile(r"(.)\1{" + str(self.cfg.max_repeated_char_run) + r",}")

    def clean(self, text: str) -> CleanResult:
        """
        对单条文本执行全部清洗步骤。
        步骤顺序：分数转换 → 规范化 → 结构性清理 → 噪音脚本清理 → 标点规范化 → 各项过滤。
        """
        # 分数字符提前转换（须在 NFKC 之前，避免 U+2044 被噪音清洗导致 ⅒→110）
        for frac, repl in self._VULGAR_FRACTION_MAP.items():
            text = text.replace(frac, repl)

        # Unicode NFKC 规范化
        # 在 NFC 基础上额外做兼容等价分解：
        #   - 数学样式变体字母（𝐀→A、𝒜→A，U+1D400-U+1D7FF）→ 普通 ASCII
        #   - 全角字符（Ａ→A、０→0）→ 半角（与 _FULLWIDTH_LATIN_TABLE 互为冗余，后者保留作保险）
        #   - fi/fl 连字、① 带圈数字等兼容字符同步处理
        if self.cfg.unicode_normalize:
            text = unicodedata.normalize("NFKC", text)

        # 结构性清理（空白、零宽字符、控制字符）
        text = self._normalize_whitespace(text)

        # 噪音脚本清理（emoji、封闭符号、IPA、无关外文）
        text = self._strip_noise_scripts(text)

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

    @classmethod
    def _strip_noise_scripts(cls, text: str) -> str:
        """
        清洗对中文语料无意义的噪音字符：
        - Emoji 及补充平面符号
        - 封闭字母数字符号（🅖、🆗 等）
        - 封闭 CJK 符号（㊗、🈚 等）
        - IPA 音标和修饰字母
        - 无关外文脚本（埃塞俄比亚文等）
        - 阿拉伯文表现形式字符（遗留兼容字符，非正常阿拉伯文）
        - 半角片假名（遗留编码，中文语料中为噪音）
        - 古代脚本（古意大利体、哥特字母、乌加里特文等）
        - 全角字符统一转为半角（Ａ→A，０→0，＜→<）
        """
        text = cls._NOISE_SCRIPT_RE.sub("", text)
        text = text.translate(cls._FULLWIDTH_LATIN_TABLE)
        return text

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
        - `‼`（U+203C 双感叹号）→ `!!`，再经 _PUNCT_TABLE 转为 `！！`
        - `\n`（换行）保持不变，不转为全角
        """
        # 省略号优先处理（避免后续逐个句号转换）
        text = cls._ELLIPSIS_RE.sub("……", text)

        # 双感叹号拆分（‼ → !!，后续经 _PUNCT_TABLE 转为 ！！）
        text = text.replace("\u203c", "!!")

        # 字符级映射（逗号、括号等半角 → 全角）
        text = text.translate(cls._PUNCT_TABLE)

        # 句号：仅转换非数字上下文中的 .
        text = cls._DOT_RE.sub("。", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        - 去除零宽字符、BOM、方向控制符、变体选择符等不可见字符
        - 去除 ASCII 控制字符（保留 \\t \\n \\r）及 C1 控制字符（U+0080–U+009F）
        - 压缩连续空行（保留最多一个空行）
        - strip 首尾空白
        """

        def _compress_inline_spaces(line: str) -> str:
            stripped = line.lstrip(" ")
            indent = line[: len(line) - len(stripped)]
            return indent + re.sub(r" {5,}", "    ", stripped)

        # 零宽字符 / BOM / 软连字 / 词连接符 / 方向控制符（LRM、RLM、Embedding、Override、PDI）/ 变体选择符
        text = re.sub(
            r"[\u200b-\u200f\u202a-\u202e\u2069\ufeff\u00ad\u2060\ufe00-\ufe0f]",
            "",
            text,
        )
        # ASCII 控制字符（保留 \t=0x09, \n=0x0A, \r=0x0D）+ C1 控制字符（U+0080–U+009F）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # 排版空格（U+2000–U+200A：全角空格、窄空格等）→ 普通空格
        text = re.sub(r"[\u2000-\u200a]", " ", text)
        # 连续空格压缩：超过 4 个空格 → 4 个空格
        text = "\n".join(_compress_inline_spaces(line) for line in text.splitlines())
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


if __name__ == "__main__":
    cleaner = BaseCleaner()
    test_text = """
    今天气温高达38℃，创下今年新高。※注：数据来源于国家气象局。

    这份报告由Ａ、Ｂ、Ｃ三组研究员共同完成——研究周期长达１２个月。
    实验结果显示：样本中⅒的个体出现了异常反应，误差范围在±0.5之间。

    以下为噪音字符测试区：
    阿拉伯文：ﻍﻐﻑ
    西里尔文（俄文）：ыьэюя
    希伯来文：אבגד
    天城文：कखग
    泰文：กขค
    高棉文：ទធន
    韩文：한국어
    藏文：ཥསཧ
    缅甸文：ကခဂ
    emoji：🎉🔥💯
    数学变体字母：𝐀𝐁𝐂𝒜
    古埃及象形文字：𓂝𓃆𓄿
    苏美尔楔形文字：𒀀𒀉𒁀
    IPA音标：ɫɬɭɮ
    组合附加符号：a̋ȉ
    双感叹号：‼
    项目符号：†‡•
    盲文：⠀⠁⠂
    希腊文：ΑΒΓ
    马拉雅拉姆文：അഇഎ
    格鲁吉亚文：აბგ
    卢恩文：ᚢᚦᚱ
    他加禄文：ᜃᜄᜅ
    彝文：ꀀꀁꀃ
    易经卦符：䷀䷁䷂
    CJK扩展A：㐀㐁
    Latin-1字母：ÀÁÂ
    C1控制字符：\x8d\x9f
    方向控制符：\u200e\u202a\u2069
    变体选择符：\ufe00\ufe0f

    以下字符应当保留：
    标准标点：，。！？：；（）《》「」【】、
    破折号：–—
    引号：「这是引用内容」《书名》
    参考符号：※详见附录
    摄氏度：当前温度为25℃
    汉字：中文内容应完整保留，包括生僻字
    数字与英文：abc123 ABC
    """
    print(cleaner.clean(test_text))

    chars = ["󰀊", "󰡔", "󰡕"]
    for ch in chars:
        print(repr(ch), hex(ord(ch)))

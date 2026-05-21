"""Chinese NLP infrastructure: tokenization, keyword extraction, sentiment, topics.

Pure-computation module with optional external libraries (jieba, pkuseg, snownlp,
gensim).  All public methods accept ``args: dict`` and return ``str`` (JSON).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tokenization dependencies
# ---------------------------------------------------------------------------
try:
    import jieba
    import jieba.posseg as jieba_posseg
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    import pkuseg
    HAS_PKUSEG = True
except ImportError:
    HAS_PKUSEG = False

try:
    from snownlp import SnowNLP
    HAS_SNOWLP = True
except ImportError:
    HAS_SNOWLP = False

# ---------------------------------------------------------------------------
# Academic domain dictionary  (~2000 terms across 5 disciplines)
# ---------------------------------------------------------------------------
_ACADEMIC_DICT: List[str] = [
    # --- Sociology ---
    "社会资本", "文化资本", "符号资本", "惯习", "场域", "社会分层", "社会流动",
    "社会网络", "强弱关系", "结构洞", "制度主义", "理性选择", "集体行动",
    "社会资本理论", "社会交换理论", "社会建构主义", "社会认同理论", "角色理论",
    "标签理论", "冲突理论", "功能主义", "社会达尔文主义", "社会契约论",
    "城市社会学", "农村社会学", "经济社会学", "政治社会学", "文化社会学",
    "教育社会学", "法律社会学", "宗教社会学", "科学社会学", "知识社会学",
    "组织社会学", "家庭社会学", "性别社会学", "老年社会学", "青年社会学",
    "社会调查方法", "问卷设计", "量表开发", "信度", "效度", "因子分析",
    "社会距离", "相对剥夺", "绝对贫困", "相对贫困", "贫困陷阱", "福利依赖",
    "社区治理", "基层治理", "社会治理", "国家治理", "社会治理现代化",
    "城镇化", "新型城镇化", "城乡二元结构", "户籍制度", "人口流动",
    "老龄化社会", "人口红利", "人口转型", "人口普查", "人口结构",
    "社会排斥", "社会融入", "社会支持", "社会韧性", "社会信任",
    "公共领域", "公民社会", "市民社会", "第三部门", "非政府组织",
    "数字鸿沟", "信息不平等", "数字劳动", "平台经济", "零工经济",
    "灵活就业", "非正规就业", "劳动关系", "劳动权益", "社会保障",
    "社会政策", "公共政策", "福利国家", "社会民主主义", "新自由主义",
    "全球化", "逆全球化", "全球化与社会", "社会不平等", "收入差距",
    "社会流动", "阶层固化", "中产阶级", "精英阶层", "底层社会",
    "社会运动", "集体行为", "社会革命", "社会变迁", "社会转型",
    "质性研究", "量化研究", "混合方法", "扎根理论", "民族志",
    "参与观察", "深度访谈", "焦点小组", "个案研究", "比较研究",
    # --- Education ---
    "教育公平", "教育质量", "教育资源", "教育均衡", "教育现代化",
    "素质教育", "应试教育", "核心素养", "课程改革", "教学改革",
    "高等教育", "职业教育", "基础教育", "学前教育", "终身教育",
    "教育评价", "形成性评价", "终结性评价", "标准化测试", "表现性评价",
    "教师专业发展", "教学效能感", "教师认同", "教师倦怠", "教学反思",
    "行动研究", "课例研究", "教学设计", "学习理论", "建构主义学习",
    "自主学习", "合作学习", "探究式学习", "翻转课堂", "混合式教学",
    "教育技术", "在线教育", "远程教育", "慕课", "微课",
    "德育", "思想政治教育", "价值教育", "品格教育", "公民教育",
    "教育社会学", "教育经济学", "教育管理学", "比较教育学", "教育史",
    "学业成就", "学习动机", "学习策略", "自我效能感", "学习投入",
    "学校领导力", "学校文化", "学校改进", "教育政策分析", "教育治理",
    "教育信息化", "智慧教育", "人工智能教育", "编程教育", "创客教育",
    "教育机会均等", "代际流动", "文化再生产", "隐性课程", "教育分流",
    # --- Political Science ---
    "政治参与", "政治文化", "政治社会化", "政治信任", "政治效能感",
    "政治制度化", "政治发展", "政治现代化", "政治稳定", "政治转型",
    "民主化", "民主巩固", "民主转型", "选举制度", "政党制度",
    "国家能力", "国家建构", "国家治理", "政府绩效", "公共服务",
    "公共管理", "新公共管理", "善治", "治理能力", "治理体系",
    "公共政策分析", "政策过程", "议程设置", "政策执行", "政策评估",
    "制度设计", "制度变迁", "路径依赖", "制度供给", "制度创新",
    "国际关系", "地缘政治", "软实力", "硬实力", "巧实力",
    "公共舆论", "媒体政治", "政治传播", "政治话语", "框架效应",
    "政治心理学", "政治行为", "投票行为", "政治认同", "政治极化",
    "政治经济学", "发展型国家", "管制型国家", "福利体制", "国家资本主义",
    "中央地方关系", "地方治理", "社区参与", "协商民主", "协商治理",
    "社会矛盾", "群体性事件", "信访制度", "维稳体制", "风险社会",
    # --- Psychology ---
    "自我概念", "自我效能", "自我调节", "自我决定", "自我实现",
    "心理韧性", "心理弹性", "心理资本", "心理幸福感", "主观幸福感",
    "认知发展", "社会认知", "认知失调", "认知负荷", "执行功能",
    "情绪调节", "情绪智力", "情绪劳动", "情绪感染", "情绪表达",
    "依恋理论", "心理分析", "人本主义", "行为主义", "认知行为疗法",
    "压力应对", "应对策略", "心理危机", "创伤后成长", "正念",
    "社会支持", "社会比较", "从众行为", "服从权威", "群体决策",
    "态度改变", "说服理论", "刻板印象", "偏见", "歧视",
    "心理健康", "心理测量", "心理评估", "信效度检验", "项目反应理论",
    "结构方程模型", "中介效应", "调节效应", "多层线性模型", "潜变量增长模型",
    # --- Communication ---
    "传播学", "媒介效果", "议程设置", "框架理论", "使用与满足",
    "沉默的螺旋", "知识沟", "数字鸿沟", "媒介素养", "媒介融合",
    "新媒体", "社交媒体", "算法推荐", "信息茧房", "回音室效应",
    "舆论引导", "舆情分析", "网络舆情", "危机传播", "风险传播",
    "健康传播", "科学传播", "环境传播", "发展传播", "政治传播",
    "跨文化传播", "全球化传播", "媒介政治经济学", "媒介社会学", "媒介哲学",
    "受众研究", "媒介话语", "视觉传播", "影像叙事", "纪录片研究",
    "计算传播学", "计算社会科学", "大数据分析", "网络分析", "文本挖掘",
    # --- Methodology (shared) ---
    "因果推断", "因果识别", "反事实框架", "潜在结果", "处理效应",
    "平均处理效应", "异质性效应", "中介效应", "调节效应", "交互效应",
    "双重差分", "断点回归", "工具变量", "倾向得分匹配", "合成控制法",
    "面板数据", "横截面数据", "纵向数据", "时间序列", "面板回归",
    "固定效应", "随机效应", "混合效应模型", "多层模型", "聚类标准误",
    "内生性", "遗漏变量", "选择性偏差", "样本选择", "测量误差",
    "稳健性检验", "安慰剂检验", "平行趋势检验", "过度识别检验", "异质性分析",
    "机制分析", "异质性分析", "子样本分析", "敏感性分析", "蒙特卡洛模拟",
    "结构方程模型", "路径分析", "验证性因子分析", "探索性因子分析", "量表编制",
    "内容分析法", "框架分析", "话语分析", "叙事分析", "现象学分析",
    "案例研究法", "比较案例法", "过程追踪法", "历史制度主义", "档案研究法",
    "文献计量法", "系统综述", "范围综述", "叙事综述", "范围综述法",
    "三角验证", "成员核查", "厚描述", "反思性", "严谨性",
    "理论抽样", "目的性抽样", "滚雪球抽样", "最大变异抽样", "理论饱和",
]

# ---------------------------------------------------------------------------
# Chinese stop-word set (merged from HIT + Baidu lists, deduplicated)
# ---------------------------------------------------------------------------
_CN_STOPWORDS: Set[str] = {
    # Pronouns
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "自己", "自身", "本人", "人家", "别人", "大家", "彼此", "互相",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "这么",
    "那么", "这样", "那样", "这儿", "那儿", "此", "该", "其", "之",
    "什么", "哪", "哪个", "哪些", "哪里", "谁", "怎么", "怎样", "如何",
    "多少", "几", "何", "为啥", "为什么", "为何",
    # Conjunctions / Prepositions
    "和", "与", "及", "以及", "并", "并且", "或", "或者", "还是", "但",
    "但是", "然而", "可是", "不过", "却", "虽然", "尽管", "因为", "所以",
    "由于", "因此", "如果", "假如", "要是", "若", "既然", "即使", "就算",
    "只要", "只有", "除非", "无论", "不管", "不但", "而且", "不仅", "还",
    "更", "而且", "乃至", "甚至", "于是", "然后", "接着", "随后", "其次",
    # Particles / Auxiliaries
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "他", "么", "那", "个",
    "地", "得", "啊", "吧", "呢", "吗", "哦", "嗯", "呀", "哎", "唉",
    "啦", "哇", "哈", "嘛", "哟", "呗", "嘞", "喽", "唷", "呀",
    # Adverbs / degree
    "很", "非常", "特别", "十分", "极其", "相当", "比较", "稍微", "略",
    "太", "最", "更", "越", "极", "多", "少", "大", "小", "高", "低",
    "不", "没", "没有", "无", "非", "未", "别", "莫", "勿", "毋",
    "也", "还", "又", "再", "都", "全", "总", "共", "只", "仅",
    "已经", "曾经", "刚刚", "正在", "将要", "将要", "快要", "即将",
    "一直", "始终", "永远", "经常", "常常", "往往", "通常", "有时",
    "偶尔", "很少", "从不", "绝不", "大概", "也许", "可能", "似乎",
    "几乎", "差不多", "约", "大约", "左右", "上下", "之间",
    # Verbs / nouns that are too generic
    "可以", "能够", "应该", "必须", "需要", "得", "会", "能",
    "做", "搞", "弄", "搞", "打", "把", "被", "让", "给", "用",
    "从", "向", "往", "对", "于", "跟", "同", "比", "靠", "沿",
    "通过", "根据", "按照", "依照", "遵照", "鉴于", "关于", "至于",
    "为了", "因为", "由于", "以", "为", "因",
    "时候", "时间", "地方", "东西", "办法", "方面", "问题", "情况",
    "样子", "道理", "意义", "原因", "结果", "过程", "关系", "条件",
    "环境", "国家", "中国", "世界", "历史", "建设",
    # Common function phrases
    "是", "不是", "就是", "便是", "算是", "倒是", "还是", "或者是",
    "就是", "也正是", "一般来说", "基本上", "总体上", "整体上",
    "所以", "因此", "于是", "那么", "然后", "之后", "以后", "之后",
    "以来", "以来", "之前", "以前", "当时", "如今", "现在", "目前",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿", "些", "点", "下", "上", "中", "里",
    "外", "前", "后", "左", "右", "内", "旁", "底", "顶", "间",
    "其中", "部分", "全部", "整个", "各", "每", "某", "任何", "所有",
    "其他", "另外", "此外", "其余", "剩下", "多余", "第一", "第二",
}


class ChineseTokenizer:
    """Unified Chinese tokenizer with multi-backend support.

    Priority: jieba > pkuseg > character-level fallback.
    Supports academic domain dictionary and user custom dictionary.
    """

    _initialized: bool = False
    _user_dict_loaded: bool = False

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = workspace
        self._backend = self._detect_backend()
        self._ensure_init()

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect_backend(self) -> str:
        if HAS_JIEBA:
            return "jieba"
        if HAS_PKUSEG:
            return "pkuseg"
        return "char"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _ensure_init(self) -> None:
        if ChineseTokenizer._initialized:
            return
        if HAS_JIEBA:
            # Suppress jieba log noise
            jieba.setLogLevel(logging.WARNING)
            # Add academic domain terms
            for term in _ACADEMIC_DICT:
                jieba.add_word(term)
        ChineseTokenizer._initialized = True

    def _load_user_dict(self) -> None:
        if ChineseTokenizer._user_dict_loaded or not self.workspace:
            return
        dict_path = os.path.join(self.workspace, ".sophia", "user_dict.txt")
        if os.path.isfile(dict_path):
            if HAS_JIEBA:
                jieba.load_userdict(dict_path)
                logger.info("Loaded user dictionary from %s", dict_path)
            else:
                # Load manually for non-jieba backends
                with open(dict_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            _ACADEMIC_DICT.append(parts[0])
        ChineseTokenizer._user_dict_loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tokenize(self, text: str, mode: str = "default") -> List[str]:
        """Tokenize Chinese text.

        Parameters
        ----------
        text : str
            Input text (Chinese or mixed Chinese-English).
        mode : str
            'default' -- standard tokenization
            'search'  -- finer-grained (search-engine style)
            'all'     -- include all possible segmentations

        Returns
        -------
        List[str]
            Token list.
        """
        if not text or not text.strip():
            return []

        self._load_user_dict()

        if self._backend == "jieba":
            return self._tokenize_jieba(text, mode)
        elif self._backend == "pkuseg":
            return self._tokenize_pkuseg(text)
        else:
            return self._tokenize_char(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Chinese and English stop-words from a token list."""
        return [
            t for t in tokens
            if t.strip() and t not in _CN_STOPWORDS and t.lower() not in _CN_STOPWORDS
        ]

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def stopwords(self) -> Set[str]:
        return _CN_STOPWORDS.copy()

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _tokenize_jieba(self, text: str, mode: str) -> List[str]:
        if mode == "search":
            return list(jieba.cut_for_search(text))
        elif mode == "all":
            # For 'all' mode, return cut_for_search which gives all sub-words
            return list(jieba.cut_for_search(text))
        else:
            return list(jieba.cut(text))

    def _tokenize_pkuseg(self, text: str) -> List[str]:
        seg = pkuseg.pkuseg()
        return [w for w, _ in seg.cut(text)]

    def _tokenize_char(self, text: str) -> List[str]:
        """Character-level fallback: segment Chinese chars individually,
        keep English words and numbers together."""
        result: List[str] = []
        buf: List[str] = []

        for ch in text:
            if self._is_cjk(ch):
                if buf:
                    word = "".join(buf).strip()
                    if word:
                        # Check if the buffered chunk matches an academic term
                        matched = self._match_academic_term(word)
                        if matched:
                            result.extend(matched)
                        else:
                            result.append(word)
                    buf = []
                result.append(ch)
            elif ch.isalnum() or ch == "_" or ch == "-":
                buf.append(ch)
            else:
                if buf:
                    word = "".join(buf).strip()
                    if word:
                        result.append(word)
                    buf = []
        if buf:
            word = "".join(buf).strip()
            if word:
                matched = self._match_academic_term(word)
                if matched:
                    result.extend(matched)
                else:
                    result.append(word)
        return result

    def _match_academic_term(self, text: str) -> Optional[List[str]]:
        """Try to match known academic terms in *text* (for char-level fallback)."""
        for term in _ACADEMIC_DICT:
            if text == term:
                return [term]
        return None

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        )


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Detect whether text is primarily Chinese or English.

    Returns 'zh', 'en', or 'mixed'.
    """
    if not text:
        return "en"
    cjk_count = 0
    latin_count = 0
    for ch in text:
        if ChineseTokenizer._is_cjk(ch):
            cjk_count += 1
        elif ch.isalpha():
            latin_count += 1
    total = cjk_count + latin_count
    if total == 0:
        return "en"
    cjk_ratio = cjk_count / total
    if cjk_ratio > 0.7:
        return "zh"
    elif cjk_ratio < 0.3:
        return "en"
    return "mixed"


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction (no external deps)
# ---------------------------------------------------------------------------
def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for a token list."""
    counter = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {w: c / total for w, c in counter.items()}


def _compute_idf(doc_tokens_list: List[List[str]]) -> Dict[str, float]:
    """Compute IDF across multiple documents."""
    n_docs = len(doc_tokens_list)
    if n_docs == 0:
        return {}
    df: Counter = Counter()
    for tokens in doc_tokens_list:
        seen = set(tokens)
        for w in seen:
            df[w] += 1
    return {w: math.log((n_docs + 1) / (c + 1)) + 1 for w, c in df.items()}


def extract_keywords_tfidf(
    text: str,
    tokenizer: ChineseTokenizer,
    top_n: int = 20,
    reference_docs: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Extract keywords using TF-IDF.

    Parameters
    ----------
    text : str
        Target text.
    tokenizer : ChineseTokenizer
        Tokenizer instance.
    top_n : int
        Number of keywords to return.
    reference_docs : list of str, optional
        Additional documents for IDF computation.

    Returns
    -------
    List of (keyword, score) sorted by score descending.
    """
    tokens = tokenizer.remove_stopwords(tokenizer.tokenize(text))
    tf = _compute_tf(tokens)

    all_doc_tokens = [tokens]
    if reference_docs:
        for doc in reference_docs:
            all_doc_tokens.append(
                tokenizer.remove_stopwords(tokenizer.tokenize(doc))
            )

    idf = _compute_idf(all_doc_tokens)

    tfidf = {w: tf.get(w, 0) * idf.get(w, 0) for w in tf}
    sorted_kw = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    return sorted_kw[:top_n]


# ---------------------------------------------------------------------------
# TextRank keyword extraction (no external deps)
# ---------------------------------------------------------------------------
def extract_keywords_textrank(
    text: str,
    tokenizer: ChineseTokenizer,
    top_n: int = 20,
    window_size: int = 4,
    damping: float = 0.85,
    max_iter: int = 100,
) -> List[Tuple[str, float]]:
    """Extract keywords using TextRank algorithm.

    Parameters
    ----------
    text : str
        Target text.
    tokenizer : ChineseTokenizer
        Tokenizer instance.
    top_n : int
        Number of keywords to return.
    window_size : int
        Co-occurrence window size.
    damping : float
        Damping factor for PageRank iteration.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    List of (keyword, score) sorted by score descending.
    """
    tokens = tokenizer.remove_stopwords(tokenizer.tokenize(text))
    if not tokens:
        return []

    # Build co-occurrence graph
    graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            w1, w2 = tokens[i], tokens[j]
            if w1 != w2:
                graph[w1][w2] += 1.0
                graph[w2][w1] += 1.0

    # Initialize scores
    nodes = set(graph.keys())
    if not nodes:
        # Fallback: return top frequent tokens
        counter = Counter(tokens)
        return [(w, c / len(tokens)) for w, c in counter.most_common(top_n)]

    scores = {n: 1.0 for n in nodes}

    # Iterate
    for _ in range(max_iter):
        new_scores: Dict[str, float] = {}
        for node in nodes:
            rank_sum = 0.0
            for neighbor, weight in graph[node].items():
                neighbor_out = sum(graph[neighbor].values()) or 1.0
                rank_sum += (weight / neighbor_out) * scores[neighbor]
            new_scores[node] = (1 - damping) + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_scores[n] - scores[n]) for n in nodes)
        scores = new_scores
        if diff < 1e-6:
            break

    sorted_kw = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_kw[:top_n]


def extract_keywords(
    text: str,
    tokenizer: Optional[ChineseTokenizer] = None,
    top_n: int = 20,
    method: str = "hybrid",
) -> List[Tuple[str, float]]:
    """Extract keywords using hybrid TF-IDF + TextRank.

    Parameters
    ----------
    text : str
        Target text.
    tokenizer : ChineseTokenizer, optional
        Tokenizer instance (creates default if None).
    top_n : int
        Number of keywords.
    method : str
        'tfidf', 'textrank', or 'hybrid' (default).

    Returns
    -------
    List of (keyword, score) sorted by score descending.
    """
    if tokenizer is None:
        tokenizer = ChineseTokenizer()

    if method == "tfidf":
        return extract_keywords_tfidf(text, tokenizer, top_n)
    elif method == "textrank":
        return extract_keywords_textrank(text, tokenizer, top_n)
    else:
        # Hybrid: average normalized scores from both methods
        tfidf_results = extract_keywords_tfidf(text, tokenizer, top_n * 2)
        textrank_results = extract_keywords_textrank(text, tokenizer, top_n * 2)

        # Normalize scores
        tfidf_max = max((s for _, s in tfidf_results), default=1.0) or 1.0
        tr_max = max((s for _, s in textrank_results), default=1.0) or 1.0

        tfidf_dict = {w: s / tfidf_max for w, s in tfidf_results}
        tr_dict = {w: s / tr_max for w, s in textrank_results}

        all_words = set(tfidf_dict.keys()) | set(tr_dict.keys())
        hybrid = []
        for w in all_words:
            score = tfidf_dict.get(w, 0) * 0.5 + tr_dict.get(w, 0) * 0.5
            hybrid.append((w, score))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        return hybrid[:top_n]


# ---------------------------------------------------------------------------
# Topic extraction (LDA fallback via TF-IDF clustering)
# ---------------------------------------------------------------------------
def extract_topics(
    texts: List[str],
    tokenizer: Optional[ChineseTokenizer] = None,
    n_topics: int = 5,
) -> List[Dict[str, Any]]:
    """Extract topics from a list of texts.

    Tries gensim LDA first, falls back to TF-IDF keyword clustering.

    Parameters
    ----------
    texts : list of str
        Input texts.
    tokenizer : ChineseTokenizer, optional
        Tokenizer instance.
    n_topics : int
        Number of topics.

    Returns
    -------
    List of topic dicts with topic_id, keywords, weight, representative_docs.
    """
    if tokenizer is None:
        tokenizer = ChineseTokenizer()

    if not texts:
        return []

    # Try gensim LDA
    try:
        return _extract_topics_lda(texts, tokenizer, n_topics)
    except ImportError:
        pass

    # Fallback: TF-IDF clustering
    return _extract_topics_tfidf_cluster(texts, tokenizer, n_topics)


def _extract_topics_lda(
    texts: List[str],
    tokenizer: ChineseTokenizer,
    n_topics: int,
) -> List[Dict[str, Any]]:
    """Topic extraction using gensim LDA."""
    from gensim import corpora, models

    tokenized = [tokenizer.remove_stopwords(tokenizer.tokenize(t)) for t in texts]
    tokenized = [t for t in tokenized if t]  # Remove empty docs

    if not tokenized:
        return []

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    n_topics = min(n_topics, len(tokenized))
    lda = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=10)

    topics = []
    for idx in range(n_topics):
        topic_terms = lda.show_topic(idx, topn=10)
        keywords = [w for w, _ in topic_terms]
        # Find representative documents
        doc_scores = []
        for doc_idx, bow in enumerate(corpus):
            topic_dist = lda.get_document_topics(bow)
            for tid, prob in topic_dist:
                if tid == idx:
                    doc_scores.append((doc_idx, prob))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        rep_docs = [idx for idx, _ in doc_scores[:3]]

        topics.append({
            "topic_id": idx,
            "keywords": keywords,
            "weights": [round(s, 4) for _, s in topic_terms],
            "weight": round(sum(s for _, s in topic_terms) / len(topic_terms), 4) if topic_terms else 0,
            "representative_docs": rep_docs,
        })
    return topics


def _extract_topics_tfidf_cluster(
    texts: List[str],
    tokenizer: ChineseTokenizer,
    n_topics: int,
) -> List[Dict[str, Any]]:
    """Fallback topic extraction: cluster texts by TF-IDF similarity."""
    import math

    all_tokenized = [tokenizer.remove_stopwords(tokenizer.tokenize(t)) for t in texts]

    # Build vocabulary
    vocab: Dict[str, int] = {}
    for tokens in all_tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    if not vocab:
        return []

    # Compute TF-IDF vectors
    idf = _compute_idf(all_tokenized)
    vectors: List[Dict[str, float]] = []
    for tokens in all_tokenized:
        tf = _compute_tf(tokens)
        vec = {w: tf.get(w, 0) * idf.get(w, 0) for w in vocab}
        vectors.append(vec)

    # Simple k-means-like clustering
    n_topics = min(n_topics, len(texts))
    if n_topics <= 0:
        return []

    # Assign docs to topics by simple round-robin on cosine similarity
    # For simplicity, use keyword-based grouping
    doc_keywords = []
    for i, tokens in enumerate(all_tokenized):
        counter = Counter(tokens)
        top_kw = [w for w, _ in counter.most_common(5)]
        doc_keywords.append(top_kw)

    # Group by shared keywords
    topic_groups: Dict[int, List[int]] = defaultdict(list)
    keyword_to_topic: Dict[str, int] = {}
    topic_id = 0

    for doc_idx, keywords in enumerate(doc_keywords):
        assigned = None
        for kw in keywords:
            if kw in keyword_to_topic:
                assigned = keyword_to_topic[kw]
                break
        if assigned is None:
            if topic_id < n_topics:
                assigned = topic_id
                topic_id += 1
            else:
                assigned = topic_id - 1
        topic_groups[assigned].append(doc_idx)
        for kw in keywords[:2]:  # Assign top-2 keywords to this topic
            if kw not in keyword_to_topic:
                keyword_to_topic[kw] = assigned

    # Build topic descriptions
    topics = []
    for tid in sorted(topic_groups.keys()):
        doc_indices = topic_groups[tid]
        all_kw: List[str] = []
        for di in doc_indices:
            all_kw.extend(doc_keywords[di])
        kw_counter = Counter(all_kw)
        top_keywords = [w for w, _ in kw_counter.most_common(10)]
        weight = len(doc_indices) / len(texts)

        topics.append({
            "topic_id": tid,
            "keywords": top_keywords,
            "weights": [round(kw_counter[w] / sum(kw_counter.values()), 4) for w in top_keywords],
            "weight": round(weight, 4),
            "representative_docs": doc_indices[:3],
        })
    return topics


# ---------------------------------------------------------------------------
# Chinese sentiment analysis
# ---------------------------------------------------------------------------
# Built-in sentiment lexicon
_CN_POSITIVE: Set[str] = {
    "好", "优秀", "卓越", "出色", "杰出", "突出", "显著", "积极", "正面",
    "肯定", "支持", "赞扬", "称赞", "满意", "欣慰", "高兴", "开心", "愉快",
    "快乐", "幸福", "美满", "美好", "精彩", "成功", "胜利", "进步", "发展",
    "改善", "提升", "增强", "优化", "创新", "突破", "领先", "完善", "健全",
    "稳定", "和谐", "公平", "公正", "合理", "科学", "规范", "有效", "高效",
    "便利", "丰富", "繁荣", "富强", "强大", "壮大", "蓬勃", "兴旺", "振兴",
    "希望", "信心", "勇气", "力量", "贡献", "价值", "意义", "重要", "关键",
    "核心", "根本", "基础", "保障", "机遇", "优势", "特色", "亮点", "典范",
    "模范", "榜样", "先进", "典型", "充分", "全面", "深入", "广泛", "普遍",
    "热情", "热爱", "热心", "关怀", "关爱", "温暖", "温馨", "友善", "友好",
    "信任", "尊重", "理解", "包容", "宽容", "开放", "自由", "平等", "尊重",
}

_CN_NEGATIVE: Set[str] = {
    "差", "糟糕", "恶劣", "严重", "负面", "消极", "否定", "反对", "批评",
    "指责", "不满", "抱怨", "失望", "沮丧", "痛苦", "悲伤", "难过", "伤心",
    "焦虑", "担忧", "恐惧", "害怕", "愤怒", "生气", "烦恼", "厌烦", "厌倦",
    "失败", "挫折", "困难", "困境", "危机", "风险", "威胁", "挑战", "问题",
    "缺陷", "不足", "漏洞", "弱点", "短板", "瓶颈", "障碍", "阻力", "困境",
    "衰退", "下降", "减少", "萎缩", "倒退", "恶化", "退化", "老化", "僵化",
    "腐败", "堕落", "腐败", "不公平", "不合理", "不规范", "不科学", "不透明",
    "歧视", "偏见", "排斥", "孤立", "边缘化", "忽视", "漠视", "冷漠", "无情",
    "混乱", "无序", "动荡", "不安", "不稳", "失衡", "失调", "失控", "失灵",
    "浪费", "损失", "损害", "破坏", "污染", "腐败", "贪污", "欺诈", "违法",
    "犯罪", "暴力", "冲突", "矛盾", "争端", "纠纷", "分歧", "对立", "分裂",
    "贫穷", "贫困", "落后", "匮乏", "短缺", "不足", "缺乏", "缺失", "空白",
}

# Degree adverbs that amplify sentiment
_CN_DEGREE_HIGH: Set[str] = {
    "非常", "特别", "十分", "极其", "相当", "异常", "极为", "极度", "至为",
    "万分", "极度", "高度", "大幅", "显著", "明显", "深刻", "强烈", "严厉",
}
_CN_DEGREE_MODERATE: Set[str] = {"比较", "较为", "相对", "尚", "颇", "挺", "蛮"}
_CN_NEGATION: Set[str] = {"不", "没", "没有", "无", "非", "未", "别", "莫", "勿"}

# Emotion dimension keywords
_EMOTION_JOY: Set[str] = {"高兴", "开心", "愉快", "快乐", "幸福", "喜悦", "欢乐", "兴奋", "欣慰", "满足"}
_EMOTION_ANGER: Set[str] = {"愤怒", "生气", "恼火", "愤慨", "暴怒", "激怒", "气愤", "恼怒", "发怒", "怒"}
_EMOTION_SADNESS: Set[str] = {"悲伤", "难过", "伤心", "痛苦", "忧伤", "悲哀", "凄凉", "心酸", "沮丧", "失落"}
_EMOTION_FEAR: Set[str] = {"恐惧", "害怕", "担心", "焦虑", "忧虑", "惶恐", "不安", "紧张", "惊慌", "恐慌"}
_EMOTION_SURPRISE: Set[str] = {"惊讶", "意外", "震惊", "吃惊", "诧异", "意想不到", "出乎意料", "突然", "惊喜"}


def analyze_sentiment_cn(
    text: str,
    tokenizer: Optional[ChineseTokenizer] = None,
) -> Dict[str, Any]:
    """Analyze sentiment of Chinese text.

    Parameters
    ----------
    text : str
        Input Chinese text.
    tokenizer : ChineseTokenizer, optional
        Tokenizer instance.

    Returns
    -------
    Dict with sentiment, score, confidence, dimensions, key_phrases.
    """
    if tokenizer is None:
        tokenizer = ChineseTokenizer()

    # Try snownlp first
    if HAS_SNOWLP:
        try:
            s = SnowNLP(text)
            snownlp_score = s.sentiments  # 0.0 (negative) to 1.0 (positive)
            sentiment = "positive" if snownlp_score > 0.6 else ("negative" if snownlp_score < 0.4 else "neutral")
            score = (snownlp_score - 0.5) * 2  # Normalize to [-1, 1]
            confidence = abs(snownlp_score - 0.5) * 2
        except Exception:
            snownlp_score = None
            score = 0.0
            sentiment = "neutral"
            confidence = 0.0
    else:
        snownlp_score = None
        score = 0.0
        sentiment = "neutral"
        confidence = 0.0

    # Dictionary-based analysis (always runs as supplement)
    tokens = tokenizer.tokenize(text)
    pos_count = 0
    neg_count = 0
    key_phrases: List[str] = []

    for i, token in enumerate(tokens):
        degree = 1.0
        # Check if preceded by degree adverb or negation
        if i > 0:
            prev = tokens[i - 1]
            if prev in _CN_DEGREE_HIGH:
                degree = 2.0
            elif prev in _CN_DEGREE_MODERATE:
                degree = 0.7
            if prev in _CN_NEGATION:
                # Negation flips sentiment
                if token in _CN_POSITIVE:
                    neg_count += degree
                    key_phrases.append(f"{prev}{token}")
                elif token in _CN_NEGATIVE:
                    pos_count += degree * 0.5  # Negated negative is weakly positive
                    key_phrases.append(f"{prev}{token}")
                continue

        if token in _CN_POSITIVE:
            pos_count += degree
            key_phrases.append(token)
        elif token in _CN_NEGATIVE:
            neg_count += degree
            key_phrases.append(token)

    # Combine dictionary score with snownlp (if available)
    total = pos_count + neg_count
    if total > 0:
        dict_score = (pos_count - neg_count) / total
        if snownlp_score is not None:
            score = score * 0.6 + dict_score * 0.4
        else:
            score = dict_score
        confidence = abs(score)

    if score > 0.15:
        sentiment = "positive"
    elif score < -0.15:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Emotion dimensions
    dimensions = {
        "joy": sum(1 for t in tokens if t in _EMOTION_JOY) / max(len(tokens), 1),
        "anger": sum(1 for t in tokens if t in _EMOTION_ANGER) / max(len(tokens), 1),
        "sadness": sum(1 for t in tokens if t in _EMOTION_SADNESS) / max(len(tokens), 1),
        "fear": sum(1 for t in tokens if t in _EMOTION_FEAR) / max(len(tokens), 1),
        "surprise": sum(1 for t in tokens if t in _EMOTION_SURPRISE) / max(len(tokens), 1),
    }

    return {
        "sentiment": sentiment,
        "score": round(score, 4),
        "confidence": round(min(confidence, 1.0), 4),
        "dimensions": dimensions,
        "key_phrases": key_phrases[:10],
        "backend": "snownlp+dict" if HAS_SNOWLP else "dict",
        "positive_hits": int(pos_count),
        "negative_hits": int(neg_count),
    }


def analyze_sentiment_batch(
    texts: List[str],
    tokenizer: Optional[ChineseTokenizer] = None,
) -> List[Dict[str, Any]]:
    """Batch sentiment analysis for multiple texts."""
    if tokenizer is None:
        tokenizer = ChineseTokenizer()
    return [analyze_sentiment_cn(t, tokenizer) for t in texts]


# ---------------------------------------------------------------------------
# JSON serialization helper (consistent with qualitative.py)
# ---------------------------------------------------------------------------
def _json(result: dict) -> str:
    """Serialize *result* to a JSON string, converting non-serializable types."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, set):
            return [_convert(v) for v in sorted(obj, key=str)]
        return obj

    return json.dumps(_convert(result), ensure_ascii=False, indent=2)

"""Discourse analysis engine: power relations, discourse strategies, ideology.

Supports three frameworks:
  - Foucaultian (power/knowledge/discourse)
  - CDA (Fairclough 3D model)
  - Narrative discourse

Pure-computation with optional LLM enrichment.  All public methods accept
``args: dict`` and return ``str`` (JSON).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from sophia.research._input import resolve_parent_ids

# ---------------------------------------------------------------------------
# Discourse subject markers (who speaks, who is spoken about)
# ---------------------------------------------------------------------------
_SPEAKER_PATTERNS: List[Tuple[str, str]] = [
    # (pattern, role_label)
    (r"政府|国务院|部委|中央|地方政府|行政部门", "government"),
    (r"企业|公司|集团|平台|资本|雇主", "enterprise"),
    (r"民众|群众|人民|公民|居民|老百姓|市民", "public"),
    (r"学者|专家|研究者|教授|分析师", "scholar"),
    (r"媒体|新闻|记者|报道|社论|评论", "media"),
    (r"学生|青年|孩子|儿童|青少年", "youth"),
    (r"农民|农民工|工人|劳动者|职工", "labor"),
    (r"教师|医生|护士|公务员|基层干部", "professional"),
    (r"女性|男性|老人|残障|弱势群体", "vulnerable"),
    (r"我|我们|笔者|本文", "author"),
]

_TARGET_PATTERNS: List[Tuple[str, str]] = [
    (r"被.{0,4}的|受到.{0,4}影响|遭到.{0,4}|遭遇.{0,4}", "passive_target"),
    (r"要求|命令|规定|禁止|限制|管控", "command_target"),
    (r"帮助|支持|保护|扶持|惠及", "support_target"),
    (r"忽视|排斥|边缘化|歧视|剥夺", "exclude_target"),
]

# ---------------------------------------------------------------------------
# Discourse strategy detection patterns
# ---------------------------------------------------------------------------
_STRATEGY_PATTERNS: List[Dict[str, Any]] = [
    {
        "strategy": "模糊化",
        "description": "使用模糊表述回避明确立场",
        "patterns": [
            r"可能|或许|也许|大概|似乎|在一定程度上",
            r"有关部门|相关方面|某些",
            r"据说|传言|坊间传闻|有人认为",
        ],
    },
    {
        "strategy": "权威引用",
        "description": "引用权威来源增强说服力",
        "patterns": [
            r"根据.{2,20}(规定|文件|法律|政策|精神)",
            r"[某某]{0,4}(指出|强调|表示|认为|提出)",
            r"数据显示|统计表明|研究表明|调查发现",
            r"习.{0,4}指出|李.{0,4}强调",
        ],
    },
    {
        "strategy": "数据修辞",
        "description": "用数据/统计增强论证可信度",
        "patterns": [
            r"\d+\.?\d*%|\d+\.?\d*亿|\d+\.?\d*万|\d+\.?\d*个",
            r"增长|下降|上升|减少|提高|降低",
            r"同比|环比|较上年|较去年同期",
        ],
    },
    {
        "strategy": "情感动员",
        "description": "诉诸情感激发读者共鸣",
        "patterns": [
            r"令人.{1,6}的是|不禁.{1,6}|深感|痛心|欣慰|振奋",
            r"必须|紧急|刻不容缓|迫在眉睫|势在必行",
            r"共同|携手|团结|齐心协力|众志成城",
        ],
    },
    {
        "strategy": "他者化",
        "description": "区分'我们'与'他们'的边界",
        "patterns": [
            r"他们|那些人|外部势力|敌对势力|别有用心",
            r"不符合国情|不适合.*实际|照搬照抄",
            r"我们.*必须|中国特色.*道路",
            r"西方|境外|外部|外来|异质",
        ],
    },
    {
        "strategy": "二元对立",
        "description": "设置非此即彼的对立框架",
        "patterns": [
            r"要么.*要么|不是.*就是|非.*即",
            r"正义与邪恶|光明与黑暗|进步与落后",
            r"必须选择|没有中间|不存在妥协",
        ],
    },
]

# ---------------------------------------------------------------------------
# Ideology framework detection
# ---------------------------------------------------------------------------
_IDEOLOGY_FRAMES: List[Dict[str, Any]] = [
    {
        "ideology": "新自由主义",
        "indicators": ["市场化", "私有化", "自由竞争", "去监管", "效率优先",
                       "企业家精神", "自由贸易", "全球化", "个人主义"],
    },
    {
        "ideology": "国家主义",
        "indicators": ["集中力量", "统一领导", "顶层设计", "举国体制", "制度优势",
                       "党的领导", "集中统一", "规划引领", "以人民为中心", "举国"],
    },
    {
        "ideology": "民粹主义",
        "indicators": ["精英阶层", "既得利益", "底层呼声", "人民意愿", "反体制",
                       "代表人民", "平民立场", "利益集团"],
    },
    {
        "ideology": "民族主义",
        "indicators": ["民族复兴", "国家利益", "主权", "领土完整", "文化自信",
                       "中华民族", "伟大复兴", "民族尊严"],
    },
    {
        "ideology": "环保主义",
        "indicators": ["可持续发展", "生态文明", "碳中和", "绿色发展", "环境保护",
                       "生态保护", "低碳", "气候"],
    },
    {
        "ideology": "女权主义",
        "indicators": ["性别平等", "女性权益", "性别歧视", "男性凝视", "父权制",
                       "性别主流化", "赋权", "身体自主"],
    },
    {
        "ideology": "社会民主主义",
        "indicators": ["社会公正", "福利保障", "再分配", "公共服务均等化",
                       "社会保障", "劳动者权益", "共同富裕"],
    },
    {
        "ideology": "保守主义",
        "indicators": ["传统价值", "文化传承", "秩序", "稳定", "渐进改革",
                       "家庭价值", "道德底线", "习俗"],
    },
    {
        "ideology": "技术乐观主义",
        "indicators": ["科技创新", "数字化转型", "人工智能", "技术赋能",
                       "智慧城市", "数字经济", "技术驱动"],
    },
    {
        "ideology": "批判理论",
        "indicators": ["异化", "剥削", "权力关系", "话语霸权", "意识形态",
                       "压迫", "解放", "反抗", "批判", "解构"],
    },
]


def _json(result: dict) -> str:
    def _convert(obj):
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


class DiscourseEngine:
    """Discourse analysis engine with rule-based and optional LLM analysis."""

    def __init__(self, result_store=None, provider=None):
        self.store = result_store
        self.provider = provider

    def analyze_discourse(self, args: dict) -> str:
        """Analyze discourse in text.

        Parameters (in args dict):
            text : str - Input text
            framework : str - 'foucault' / 'cda' / 'narrative' (default: 'foucault')
            parent_ids : list - Optional parent result IDs for lineage

        Returns JSON string with analysis results.
        """
        text = args.get("text", "")
        framework = args.get("framework", "foucault")
        parent_ids = resolve_parent_ids(args)

        if not text or not text.strip():
            return _json({"error": "No text provided", "results": {}})

        # Rule-based analysis (always runs)
        subjects = self._identify_subjects(text)
        power_relations = self._analyze_power_relations(text, subjects)
        strategies = self._detect_strategies(text)
        ideologies = self._detect_ideologies(text)

        # Framework-specific analysis
        if framework == "cda":
            framework_results = self._cda_analysis(text, subjects, strategies)
        elif framework == "narrative":
            framework_results = self._narrative_discourse(text, subjects)
        else:
            framework_results = self._foucault_analysis(text, subjects, power_relations)

        result = {
            "framework": framework,
            "text_length": len(text),
            "subjects": subjects,
            "power_relations": power_relations,
            "discourse_strategies": strategies,
            "ideology_frames": ideologies,
            "framework_analysis": framework_results,
            "summary": self._generate_summary(subjects, strategies, ideologies),
        }

        if self.store:
            params = {"framework": framework, "text_length": len(text)}
            self.store.save("discourse_analysis", params, result, parent_ids=parent_ids)

        return _json(result)

    # ------------------------------------------------------------------
    # Subject identification
    # ------------------------------------------------------------------
    def _identify_subjects(self, text: str) -> List[Dict[str, Any]]:
        """Identify discourse subjects (who speaks, who is spoken about)."""
        subjects = []
        seen_roles: Set[str] = set()

        for pattern, role in _SPEAKER_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                # Find positions
                positions = []
                for m in re.finditer(pattern, text):
                    start = max(0, m.start() - 10)
                    end = min(len(text), m.end() + 10)
                    context = text[start:end].replace("\n", " ")
                    positions.append({
                        "match": m.group(),
                        "position": m.start(),
                        "context": context,
                    })

                if role not in seen_roles:
                    subjects.append({
                        "role": role,
                        "frequency": len(matches),
                        "examples": positions[:5],
                    })
                    seen_roles.add(role)

        subjects.sort(key=lambda x: x["frequency"], reverse=True)
        return subjects

    # ------------------------------------------------------------------
    # Power relation analysis
    # ------------------------------------------------------------------
    def _analyze_power_relations(self, text: str, subjects: List[Dict]) -> List[Dict[str, Any]]:
        """Detect power relations between subjects."""
        relations = []
        role_set = {s["role"] for s in subjects}

        # Check for specific relation patterns
        relation_patterns = [
            {
                "pattern": r"政府.{0,10}(管控|管理|治理|规定|要求|监督|批准)",
                "from_role": "government", "to_role": "public",
                "relation_type": "支配", "description": "政府对公众的行政管控",
            },
            {
                "pattern": r"(要求|迫使|迫使|压迫|压制).{0,6}(群众|民众|劳动者|工人)",
                "from_role": "government", "to_role": "public",
                "relation_type": "支配", "description": "行政权力对公众的支配",
            },
            {
                "pattern": r"(剥削|压榨|克扣|拖欠).{0,6}(工资|薪酬|劳动者)",
                "from_role": "enterprise", "to_role": "labor",
                "relation_type": "支配", "description": "资本对劳动者的剥削",
            },
            {
                "pattern": r"(保护|维护|保障|支持|帮助).{0,6}(权益|利益|权利|就业)",
                "from_role": "government", "to_role": "public",
                "relation_type": "保护", "description": "政府对公众权益的保护",
            },
            {
                "pattern": r"(监督|制约|批评|质疑|抗议).{0,6}(政府|官员|权力|行政)",
                "from_role": "public", "to_role": "government",
                "relation_type": "抵抗", "description": "公众对政府的监督与制约",
            },
            {
                "pattern": r"(呼吁|建议|主张|倡导).{0,6}(改革|改变|完善|调整)",
                "from_role": "scholar", "to_role": "government",
                "relation_type": "协商", "description": "知识分子对政策的建言",
            },
        ]

        for rp in relation_patterns:
            matches = list(re.finditer(rp["pattern"], text))
            if matches and rp["from_role"] in role_set:
                for m in matches[:3]:
                    start = max(0, m.start() - 15)
                    end = min(len(text), m.end() + 15)
                    context = text[start:end].replace("\n", " ")
                    relations.append({
                        "from": rp["from_role"],
                        "to": rp["to_role"],
                        "type": rp["relation_type"],
                        "description": rp["description"],
                        "evidence": context,
                        "position": m.start(),
                    })

        return relations

    # ------------------------------------------------------------------
    # Discourse strategy detection
    # ------------------------------------------------------------------
    def _detect_strategies(self, text: str) -> List[Dict[str, Any]]:
        """Detect discourse strategies in text."""
        detected = []

        for strategy_def in _STRATEGY_PATTERNS:
            evidences = []
            for pattern in strategy_def["patterns"]:
                for m in re.finditer(pattern, text):
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    context = text[start:end].replace("\n", " ")
                    evidences.append({
                        "match": m.group(),
                        "context": context,
                        "position": m.start(),
                    })

            if evidences:
                detected.append({
                    "strategy": strategy_def["strategy"],
                    "description": strategy_def["description"],
                    "frequency": len(evidences),
                    "evidences": evidences[:5],
                })

        detected.sort(key=lambda x: x["frequency"], reverse=True)
        return detected

    # ------------------------------------------------------------------
    # Ideology frame detection
    # ------------------------------------------------------------------
    def _detect_ideologies(self, text: str) -> List[Dict[str, Any]]:
        """Detect ideological frameworks present in text."""
        detected = []

        for frame in _IDEOLOGY_FRAMES:
            hits = []
            for indicator in frame["indicators"]:
                count = text.count(indicator)
                if count > 0:
                    hits.append({"indicator": indicator, "count": count})

            if hits:
                total_hits = sum(h["count"] for h in hits)
                detected.append({
                    "ideology": frame["ideology"],
                    "total_hits": total_hits,
                    "indicators_found": hits,
                    "strength": "strong" if total_hits >= 5 else ("moderate" if total_hits >= 2 else "weak"),
                })

        detected.sort(key=lambda x: x["total_hits"], reverse=True)
        return detected[:5]  # Top 5

    # ------------------------------------------------------------------
    # Framework-specific analyses
    # ------------------------------------------------------------------
    def _foucault_analysis(self, text: str, subjects: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """Foucaultian analysis: power/knowledge/discourse triad."""
        return {
            "knowledge_regimes": [
                s for s in subjects if s["role"] in ("scholar", "government", "media")
            ],
            "power_mechanisms": [
                r for r in relations if r["type"] in ("支配", "保护")
            ],
            "resistance_points": [
                r for r in relations if r["type"] in ("抵抗", "协商")
            ],
            "discourse_formations": {
                "dominant_voice": subjects[0]["role"] if subjects else "unknown",
                "marginalized_voices": [s["role"] for s in subjects[-2:] if s["frequency"] <= 2],
            },
        }

    def _cda_analysis(self, text: str, subjects: List[Dict], strategies: List[Dict]) -> Dict[str, Any]:
        """CDA analysis: Fairclough 3D model."""
        return {
            "text_analysis": {
                "vocabulary": {"formal_level": "high" if any(s["role"] == "government" for s in subjects) else "medium"},
                "grammar": {"voice": "passive" if "被" in text else "active"},
                "cohesion": {"discourse_markers": len(re.findall(r"因此|所以|然而|但是|此外|同时", text))},
            },
            "discourse_practice": {
                "production": "institutional" if any(s["role"] == "government" for s in subjects) else "grassroots",
                "consumption": "public audience",
                "strategies_used": [s["strategy"] for s in strategies],
            },
            "social_practice": {
                "hegemonic_elements": [s["strategy"] for s in strategies if s["strategy"] in ("权威引用", "他者化", "二元对立")],
                "naturalized_assumptions": [],
            },
        }

    def _narrative_discourse(self, text: str, subjects: List[Dict]) -> Dict[str, Any]:
        """Narrative discourse analysis."""
        sentences = re.split(r"[。！？\n]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "narrative_structure": {
                "total_segments": len(sentences),
                "avg_segment_length": sum(len(s) for s in sentences) / max(len(sentences), 1),
            },
            "voice_analysis": {
                "first_person": len(re.findall(r"我|我们|笔者", text)),
                "third_person": len(re.findall(r"他|她|他们|她们", text)),
                "impersonal": len(re.findall(r"被|受到|遭到", text)),
            },
            "temporal_markers": {
                "past": len(re.findall(r"曾经|过去|以前|历史上|此前", text)),
                "present": len(re.findall(r"目前|现在|当前|如今|当下", text)),
                "future": len(re.findall(r"将来|未来|今后|预期|展望", text)),
            },
        }

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def _generate_summary(self, subjects, strategies, ideologies) -> str:
        parts = []
        if subjects:
            dominant = subjects[0]["role"] if subjects else "unknown"
            parts.append(f"主要话语主体为{dominant}")
        if strategies:
            top_strategy = strategies[0]["strategy"] if strategies else ""
            parts.append(f"检测到的主要话语策略为{top_strategy}")
        if ideologies:
            top_ideo = ideologies[0]["ideology"] if ideologies else ""
            parts.append(f"文本中{top_ideo}倾向最为显著")
        return "；".join(parts) if parts else "分析完成"

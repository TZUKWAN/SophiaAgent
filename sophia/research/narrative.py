"""Narrative analysis engine: structure, turning points, identity construction.

Supports three modes:
  - structure: Labov 6-element narrative model
  - turning_point: key turning points in timeline
  - identity: role/identity construction analysis

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


# ---------------------------------------------------------------------------
# Labov narrative elements markers
# ---------------------------------------------------------------------------
_LABOV_MARKERS: Dict[str, List[str]] = {
    "abstract": ["总的来说", "概括来说", "简而言之", "一句话", "这个故事是",
                 "in short", "to summarize", "basically"],
    "orientation": ["那时候", "当时", "在.*之前", "背景是", "那个时候",
                    "地点.*是", "有.*人", "当时的情况",
                    "at that time", "the setting", "where we were"],
    "complication": ["但是", "然而", "突然", "没想到", "出乎意料", "不幸的是",
                     "困难", "问题", "冲突", "矛盾", "危机",
                     "but then", "however", "suddenly", "unexpectedly"],
    "evaluation": ["我觉得", "我认为", "最.*的是", "重要的是", "关键是",
                   "令人.*的是", "这让我", "感受到", "深刻",
                   "I think", "I feel", "what matters", "the point is"],
    "resolution": ["最后", "最终", "结果", "后来", "终于", "解决了",
                   "克服了", "走出了", "成功", "finally", "in the end", "resolved"],
    "coda": ["所以", "因此", "这告诉我", "从那以后", "现在回想",
             "经验是", "教训是", "now I realize", "the lesson"],
}

# ---------------------------------------------------------------------------
# Self/other construction markers
# ---------------------------------------------------------------------------
_SELF_MARKERS = [
    "我", "我们", "自己", "本人", "笔者", "个人",
    "我的", "我们的", "自己的",
]
_OTHER_MARKERS = [
    "他们", "她们", "别人", "人家", "对方", "社会",
    "他们的", "她们的", "别人的",
]

# ---------------------------------------------------------------------------
# Identity/role construction patterns
# ---------------------------------------------------------------------------
_IDENTITY_PATTERNS: List[Dict[str, Any]] = [
    {
        "role": "专业身份",
        "patterns": [
            r"作为.{1,6}(教师|医生|学者|律师|工程师|管理者|领导)",
            r"我是一名.{0,8}(教师|医生|学者|律师|工程师|管理者|青年教师)",
            r"我是一名来自.{0,6}的.{0,4}(教师|医生|学者|律师|工程师)",
            r"从事.{1,8}工作",
            r"在.*领域.*工作",
        ],
    },
    {
        "role": "家庭身份",
        "patterns": [
            r"作为.{0,4}(母亲|父亲|父母|子女|丈夫|妻子)",
            r"我.*孩子.*家长",
            r"家里.*需要.*照顾",
        ],
    },
    {
        "role": "社会身份",
        "patterns": [
            r"作为.{1,6}(女性|男性|青年|老年人|农村人|城市人)",
            r"来自.{0,4}(农村|城市|乡村|小镇)",
            r"我们.*群体|我们.*阶层",
            r"出身.*家庭",
        ],
    },
    {
        "role": "边缘身份",
        "patterns": [
            r"被.{0,4}排斥|被.{0,4}歧视|被.{0,4}边缘化",
            r"不被.{0,4}认可|不被.{0,4}理解",
            r"弱势|底层|困难群体",
        ],
    },
]

# ---------------------------------------------------------------------------
# Turning point indicators
# ---------------------------------------------------------------------------
_TURNING_POINT_MARKERS = [
    "转折点", "从那以后", "那一刻", "突然意识到", "改变了",
    "转折", "转变", "蜕变", "觉醒", "顿悟",
    "之前.*之后", "从此", "那时候开始",
    "最重要的变化", "关键事件",
    "人生.*转折", "命运.*改变",
]


class NarrativeEngine:
    """Narrative analysis engine with structure, turning point, and identity modes."""

    def __init__(self, result_store=None, provider=None):
        self.store = result_store
        self.provider = provider

    def analyze_narrative(self, args: dict) -> str:
        """Analyze narrative in text.

        Parameters (in args dict):
            text : str - Input text (interview transcript, narrative, etc.)
            mode : str - 'structure' / 'turning_point' / 'identity' (default: 'structure')
            parent_ids : list - Optional parent result IDs

        Returns JSON string with narrative analysis.
        """
        text = args.get("text", "")
        mode = args.get("mode", "structure")
        parent_ids = resolve_parent_ids(args)

        if not text or not text.strip():
            return _json({"error": "No text provided", "results": {}})

        if mode == "structure":
            result = self._analyze_structure(text)
        elif mode == "turning_point":
            result = self._analyze_turning_points(text)
        elif mode == "identity":
            result = self._analyze_identity(text)
        else:
            result = self._analyze_structure(text)

        result["mode"] = mode
        result["text_length"] = len(text)

        if self.store:
            params = {"mode": mode, "text_length": len(text)}
            self.store.save("narrative_analysis", params, result, parent_ids=parent_ids)

        return _json(result)

    # ------------------------------------------------------------------
    # Structure analysis (Labov 6-element model)
    # ------------------------------------------------------------------
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Identify Labov's 6 narrative elements in text."""
        sentences = re.split(r"(?<=[。！？\n])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        elements: List[Dict[str, Any]] = []
        element_counts: Dict[str, int] = Counter()

        for i, sent in enumerate(sentences):
            if not sent:
                continue
            for element_type, markers in _LABOV_MARKERS.items():
                for marker in markers:
                    if re.search(marker, sent):
                        elements.append({
                            "element": element_type,
                            "sentence_index": i,
                            "text": sent[:100],
                            "marker": marker,
                        })
                        element_counts[element_type] += 1
                        break  # One match per sentence per element type

        # Build timeline
        timeline = []
        for elem in elements:
            timeline.append({
                "index": elem["sentence_index"],
                "element": elem["element"],
                "text": elem["text"],
            })
        timeline.sort(key=lambda x: x["index"])

        # Calculate completeness (how many of 6 elements are present)
        present_elements = set(e["element"] for e in elements)
        completeness = len(present_elements) / 6.0

        # Coherence score based on element variety and order
        coherence = self._calculate_coherence(elements)

        return {
            "narrative_elements": elements,
            "timeline": timeline,
            "element_counts": dict(element_counts),
            "present_elements": list(present_elements),
            "missing_elements": [e for e in _LABOV_MARKERS if e not in present_elements],
            "completeness": round(completeness, 2),
            "coherence_score": round(coherence, 2),
            "total_segments": len(sentences),
        }

    # ------------------------------------------------------------------
    # Turning point analysis
    # ------------------------------------------------------------------
    def _analyze_turning_points(self, text: str) -> Dict[str, Any]:
        """Identify key turning points in narrative."""
        sentences = re.split(r"(?<=[。！？\n])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        turning_points = []
        timeline = []

        for i, sent in enumerate(sentences):
            if not sent:
                continue

            is_turning = False
            matched_markers = []

            for marker in _TURNING_POINT_MARKERS:
                if re.search(marker, sent):
                    is_turning = True
                    matched_markers.append(marker)

            # Also detect strong contrast patterns
            if re.search(r"之前.*[但却]现在|以前.*如今|曾经.*如今", sent):
                is_turning = True
                matched_markers.append("前后对比")

            # Sentiment shift detection (simple)
            has_negative = any(w in sent for w in ["困难", "痛苦", "挫折", "失败", "迷茫"])
            has_positive = any(w in sent for w in ["成功", "突破", "成长", "希望", "改变"])

            event_type = "neutral"
            sentiment_shift = None
            if has_negative and has_positive:
                event_type = "转折-克服"
                sentiment_shift = "negative_to_positive"
            elif has_negative:
                event_type = "挫折"
            elif has_positive:
                event_type = "积极转变"

            if is_turning:
                turning_points.append({
                    "index": i,
                    "text": sent[:150],
                    "markers": matched_markers,
                    "event_type": event_type,
                    "sentiment_shift": sentiment_shift,
                })

            timeline.append({
                "index": i,
                "text": sent[:80],
                "event_type": event_type,
            })

        return {
            "turning_points": turning_points,
            "timeline": timeline,
            "total_turning_points": len(turning_points),
            "narrative_arc": self._infer_arc(turning_points),
        }

    # ------------------------------------------------------------------
    # Identity construction analysis
    # ------------------------------------------------------------------
    def _analyze_identity(self, text: str) -> Dict[str, Any]:
        """Analyze self/other construction and role positioning."""
        # Self vs other construction
        self_count = sum(len(re.findall(re.escape(m), text)) for m in _SELF_MARKERS)
        other_count = sum(len(re.findall(re.escape(m), text)) for m in _OTHER_MARKERS)

        # Extract self-descriptions and other-descriptions
        self_descriptions = self._extract_descriptions(text, _SELF_MARKERS)
        other_descriptions = self._extract_descriptions(text, _OTHER_MARKERS)

        # Identity role patterns
        identity_roles = []
        for pattern_def in _IDENTITY_PATTERNS:
            evidences = []
            for pattern in pattern_def["patterns"]:
                for m in re.finditer(pattern, text):
                    start = max(0, m.start() - 10)
                    end = min(len(text), m.end() + 30)
                    evidences.append({
                        "match": m.group(),
                        "context": text[start:end],
                        "position": m.start(),
                    })
            if evidences:
                identity_roles.append({
                    "role_type": pattern_def["role"],
                    "frequency": len(evidences),
                    "evidences": evidences[:5],
                })

        # Role transitions (e.g., "从X变成了Y")
        transitions = []
        for m in re.finditer(r"从.{2,10}(变成|变为|转变为|转型为|成长为).{2,10}", text):
            transitions.append({
                "text": m.group(),
                "position": m.start(),
            })

        total_refs = self_count + other_count or 1
        return {
            "characters": [
                {
                    "type": "self",
                    "reference_count": self_count,
                    "proportion": round(self_count / total_refs, 2),
                    "descriptions": self_descriptions[:10],
                },
                {
                    "type": "other",
                    "reference_count": other_count,
                    "proportion": round(other_count / total_refs, 2),
                    "descriptions": other_descriptions[:10],
                },
            ],
            "identity_roles": identity_roles,
            "role_transitions": transitions,
            "self_other_ratio": round(self_count / max(other_count, 1), 2),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _calculate_coherence(self, elements: List[Dict]) -> float:
        """Calculate narrative coherence score (0-1)."""
        if not elements:
            return 0.0

        # Check for element variety
        types = set(e["element"] for e in elements)
        variety_score = len(types) / 6.0

        # Check for logical order (orientation before complication before resolution)
        order_map = {"abstract": 0, "orientation": 1, "complication": 2,
                     "evaluation": 3, "resolution": 4, "coda": 5}
        ordered = sorted(elements, key=lambda x: x["sentence_index"])
        order_violations = 0
        for i in range(1, len(ordered)):
            prev_order = order_map.get(ordered[i - 1]["element"], 0)
            curr_order = order_map.get(ordered[i]["element"], 0)
            if curr_order < prev_order - 1:  # Allow some flexibility
                order_violations += 1
        order_score = 1.0 - (order_violations / max(len(ordered), 1))

        return (variety_score * 0.6 + order_score * 0.4)

    def _infer_arc(self, turning_points: List[Dict]) -> str:
        """Infer the narrative arc from turning points."""
        if not turning_points:
            return "linear"
        types = [tp["event_type"] for tp in turning_points]
        if any("克服" in t for t in types):
            return "overcoming"
        elif any("挫折" in t for t in types):
            return "struggle"
        elif any("积极" in t for t in types):
            return "growth"
        return "mixed"

    def _extract_descriptions(self, text: str, markers: List[str]) -> List[str]:
        """Extract descriptions following self/other markers."""
        descriptions = []
        for marker in markers:
            for m in re.finditer(re.escape(marker) + r".{0,30}", text):
                desc = m.group()
                if len(desc) > len(marker) + 1:
                    descriptions.append(desc)
        return descriptions[:10]

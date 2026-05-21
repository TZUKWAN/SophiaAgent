"""Template registry: loads, manages, and recommends discipline-specific templates.

All templates are JSON files stored under discipline subdirectories.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DISCIPLINE_KEYWORDS: Dict[str, List[str]] = {
    "history": ["历史", "史料", "考证", "档案", "古籍", "方志", "口述史", "历史研究", "史学"],
    "literature": ["文学", "文化", "文本", "小说", "诗歌", "批评", "叙事", "符号", "后现代", "理论"],
    "education": ["教育", "教学", "学习", "课程", "教师", "学生", "学校", "课堂", "培养", "素质"],
    "sociology": ["社会", "田野", "民族志", "社区", "阶层", "群体", "文化", "观察", "访谈", "质性"],
    "politics_law": ["政治", "法律", "政策", "治理", "法治", "行政", "司法", "立法", "宪法", "比较"],
    "psychology": ["心理", "认知", "情绪", "实验", "量表", "脑", "行为", "人格", "精神", "测量"],
}


class TemplateRegistry:
    """Registry for discipline-specific academic writing templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            self._dir = Path(__file__).parent
        else:
            self._dir = Path(templates_dir)
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """Load all template JSON files from discipline subdirectories."""
        for subdir in self._dir.iterdir():
            if not subdir.is_dir():
                continue
            discipline = subdir.name
            for json_file in subdir.glob("*.json"):
                template_id = f"{discipline}/{json_file.stem}"
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        self._templates[template_id] = json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load template %s: %s", template_id, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by its ID (format: 'discipline/filename')."""
        return self._templates.get(template_id)

    def list_templates(self, discipline: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all templates, optionally filtered by discipline."""
        results = []
        for tid, tmpl in self._templates.items():
            parts = tid.split("/", 1)
            if discipline and parts[0] != discipline:
                continue
            results.append({
                "template_id": tid,
                "discipline": parts[0] if parts else "",
                "name": tmpl.get("name", tid),
                "description": tmpl.get("description", ""),
                "tags": tmpl.get("tags", []),
            })
        return results

    def list_disciplines(self) -> List[str]:
        """List all available disciplines."""
        disciplines = set()
        for tid in self._templates:
            parts = tid.split("/", 1)
            if parts:
                disciplines.add(parts[0])
        return sorted(disciplines)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def recommend_templates(self, research_question: str, discipline: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recommend templates based on research question keywords.

        Parameters
        ----------
        research_question : str
            User's research question or topic.
        discipline : str, optional
            If provided, only search within this discipline.

        Returns
        -------
        List of recommended templates with match score and reason.
        """
        rq = research_question.lower()
        scored: List[Tuple[float, str, str]] = []  # (score, template_id, reason)

        # If discipline is given, boost that discipline's templates
        for tid, tmpl in self._templates.items():
            parts = tid.split("/", 1)
            tmpl_discipline = parts[0] if parts else ""

            if discipline and tmpl_discipline != discipline:
                continue

            score = 0.0
            reasons = []

            # Tag matching
            tags = [t.lower() for t in tmpl.get("tags", [])]
            for tag in tags:
                if tag in rq:
                    score += 3.0
                    reasons.append(f"匹配标签 '{tag}'")

            # Description matching
            desc = tmpl.get("description", "").lower()
            desc_words = desc.split()
            for word in desc_words:
                if len(word) >= 2 and word in rq:
                    score += 1.0
                    if len(reasons) < 3:
                        reasons.append(f"匹配描述 '{word}'")

            # Outline matching
            outline = tmpl.get("outline", [])
            for item in outline:
                item_text = str(item).lower()
                for word in item_text.split():
                    if len(word) >= 2 and word in rq:
                        score += 0.5

            # Discipline keyword boost
            disc_keywords = _DISCIPLINE_KEYWORDS.get(tmpl_discipline, [])
            for kw in disc_keywords:
                if kw in rq:
                    score += 2.0
                    if len(reasons) < 3:
                        reasons.append(f"属于 '{tmpl_discipline}' 领域")
                    break

            # Discipline filter boost
            if discipline and tmpl_discipline == discipline:
                score += 5.0

            if score > 0:
                scored.append((score, tid, "；".join(reasons[:3]) if reasons else "关键词匹配"))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, tid, reason in scored[:10]:
            tmpl = self._templates[tid]
            results.append({
                "template_id": tid,
                "name": tmpl.get("name", tid),
                "description": tmpl.get("description", ""),
                "discipline": tid.split("/", 1)[0] if "/" in tid else "",
                "match_score": round(score, 1),
                "reason": reason,
            })
        return results

    # ------------------------------------------------------------------
    # Outline helpers
    # ------------------------------------------------------------------
    def get_outline(self, template_id: str) -> List[str]:
        """Get the outline of a template."""
        tmpl = self._templates.get(template_id)
        if not tmpl:
            return []
        return tmpl.get("outline", [])

    def get_section_prompt(self, template_id: str, section_name: str) -> str:
        """Get writing prompt for a specific section."""
        tmpl = self._templates.get(template_id)
        if not tmpl:
            return ""
        prompts = tmpl.get("section_prompts", {})
        return prompts.get(section_name, "")

    def get_checklist(self, template_id: str) -> List[str]:
        """Get the quality checklist for a template."""
        tmpl = self._templates.get(template_id)
        if not tmpl:
            return []
        return tmpl.get("checklist", [])

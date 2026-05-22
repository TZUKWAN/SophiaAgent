"""Journal matching and submission guide (Phase J).

J-1: Journal database with 100+ CSSCI journals
J-2: Submission guide generator
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JournalDatabase:
    """CSSCI journal database with matching and submission guidance."""

    def __init__(self, data_path: Optional[str] = None):
        self._journals: List[Dict[str, Any]] = []
        self._cas_zones: Dict[str, Dict] = {}
        self._load_data(data_path)
        self._load_cas_zones()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_path: Optional[str] = None) -> None:
        if data_path is None:
            base = Path(__file__).parent / "data"
            data_path = base / "journals.json"
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                self._journals = json.load(f)
        except FileNotFoundError:
            logger.warning("Journal data not found at %s", data_path)
            self._journals = []
        except Exception as exc:
            logger.warning("Failed to load journal data: %s", exc)
            self._journals = []

    def _load_cas_zones(self, cas_path: Optional[str] = None) -> None:
        if cas_path is None:
            base = Path(__file__).parent / "data"
            cas_path = base / "cas_zones.json"
        try:
            with open(cas_path, "r", encoding="utf-8") as f:
                self._cas_zones = json.load(f)
        except FileNotFoundError:
            logger.warning("CAS zones data not found at %s; using built-in defaults", cas_path)
            self._cas_zones = {
                "Nature": {"zone": 1, "issn": "0028-0836", "category": "综合性期刊", "top_journal": True, "impact_factor": 64.8},
                "Science": {"zone": 1, "issn": "0036-8075", "category": "综合性期刊", "top_journal": True, "impact_factor": 56.9},
                "Cell": {"zone": 1, "issn": "0092-8674", "category": "生物学", "top_journal": True, "impact_factor": 45.5},
            }
        except Exception as exc:
            logger.warning("Failed to load CAS zones data: %s", exc)
            self._cas_zones = {}

    # ------------------------------------------------------------------
    # J-1: Search and match
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search journals by name, discipline, or scope."""
        query_lower = query.lower()
        results = []
        for journal in self._journals:
            score = 0
            text = " ".join(
                filter(
                    None,
                    [
                        journal.get("name_cn", ""),
                        journal.get("name_en", ""),
                        journal.get("discipline", ""),
                        journal.get("scope", ""),
                        " ".join(journal.get("keywords", [])),
                    ],
                )
            ).lower()
            if query_lower in text:
                score += 1
            # Exact discipline match scores higher
            if query_lower == journal.get("discipline", "").lower():
                score += 3
            if score > 0:
                results.append((score, journal))
        results.sort(key=lambda x: x[0], reverse=True)
        return [j for _, j in results[:limit]]

    def match(self, args: dict) -> Dict[str, Any]:
        """Match journals to a paper abstract/title.

        Args:
            title: str
            abstract: str
            keywords: List[str]
            discipline: str
            method_type: str (optional)
            top_n: int (default 10)
        """
        title = args.get("title", "")
        abstract = args.get("abstract", "")
        keywords = args.get("keywords", [])
        discipline = args.get("discipline", "")
        method_type = args.get("method_type", "")
        top_n = args.get("top_n", 10)

        query_text = " ".join(filter(None, [title, abstract] + keywords)).lower()
        discipline_lower = discipline.lower()

        scored = []
        for journal in self._journals:
            score = 0.0

            # Discipline match (highest weight)
            journal_discipline = journal.get("discipline", "").lower()
            if discipline_lower and discipline_lower == journal_discipline:
                score += 30
            elif discipline_lower and discipline_lower in journal_discipline:
                score += 20

            # Keyword overlap with journal scope/keywords
            journal_scope = journal.get("scope", "").lower()
            journal_keywords = " ".join(journal.get("keywords", [])).lower()
            journal_text = journal_scope + " " + journal_keywords
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in journal_text:
                    score += 5
                if kw_lower in journal_scope:
                    score += 3

            # Abstract content overlap
            abstract_words = set(query_text.split())
            journal_words = set(journal_text.split())
            overlap = len(abstract_words & journal_words)
            score += overlap * 0.5

            # Method preference match
            if method_type:
                preferred_methods = journal.get("preferred_methods", [])
                if any(method_type.lower() in pm.lower() for pm in preferred_methods):
                    score += 10

            # CSSCI level bonus
            level = journal.get("cssci_level", "")
            if level == "来源刊":
                score += 3
            elif level == "扩展刊":
                score += 1

            if score > 0:
                scored.append((score, journal))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_n]

        return {
            "query": {
                "title": title,
                "discipline": discipline,
                "keywords": keywords,
            },
            "matches": [
                {
                    "journal_id": j.get("id"),
                    "name_cn": j.get("name_cn"),
                    "name_en": j.get("name_en"),
                    "discipline": j.get("discipline"),
                    "publisher": j.get("publisher"),
                    "cssci_level": j.get("cssci_level"),
                    "match_score": round(s, 1),
                    "scope_snippet": j.get("scope", "")[:120] + "...",
                }
                for s, j in top
            ],
            "total_checked": len(self._journals),
        }

    def list_disciplines(self) -> List[str]:
        """List all unique disciplines in the database."""
        disciplines = set()
        for journal in self._journals:
            d = journal.get("discipline", "")
            if d:
                disciplines.add(d)
        return sorted(disciplines)

    def list_journals(self, discipline: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """List journals, optionally filtered by discipline."""
        results = []
        for journal in self._journals:
            if not discipline or journal.get("discipline", "") == discipline:
                results.append({
                    "id": journal.get("id"),
                    "name_cn": journal.get("name_cn"),
                    "name_en": journal.get("name_en"),
                    "discipline": journal.get("discipline"),
                    "cssci_level": journal.get("cssci_level"),
                })
        return results[:limit]

    # ------------------------------------------------------------------
    # J-2: Submission guide
    # ------------------------------------------------------------------

    def get_submission_guide(self, args: dict) -> Dict[str, Any]:
        """Generate submission guide for a journal.

        Args:
            journal_id: str or journal_name: str
        """
        journal_id = args.get("journal_id", "")
        journal_name = args.get("journal_name", "")

        journal = None
        if journal_id:
            journal = next((j for j in self._journals if j.get("id") == journal_id), None)
        if journal is None and journal_name:
            journal = next(
                (j for j in self._journals if journal_name in j.get("name_cn", "")),
                None,
            )

        if journal is None:
            return {
                "error": "Journal not found",
                "suggestion": "Use journal_match or journal_search to find the correct journal ID or name.",
            }

        # Build format checklist
        format_reqs = journal.get("format_requirements", {})
        checklist = []
        for key, value in format_reqs.items():
            checklist.append(f"{key}: {value}")

        # Build word count checklist
        word_count = journal.get("word_count_limit", {})
        for key, value in word_count.items():
            checklist.append(f"{key}字数限制: {value}")

        return {
            "journal_info": {
                "id": journal.get("id"),
                "name_cn": journal.get("name_cn"),
                "name_en": journal.get("name_en"),
                "discipline": journal.get("discipline"),
                "publisher": journal.get("publisher"),
                "frequency": journal.get("frequency"),
                "cssci_level": journal.get("cssci_level"),
                "scope": journal.get("scope"),
            },
            "format_checklist": checklist,
            "common_rejection_reasons": journal.get("common_rejection_reasons", []),
            "writing_tips": journal.get("writing_tips", []),
            "preferred_methods": journal.get("preferred_methods", []),
            "similar_journals": journal.get("similar_journals", []),
            "submission_contact": journal.get("submission_contact", ""),
            "online_system": journal.get("online_system", ""),
        }

    def get_journal_detail(self, journal_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a single journal."""
        for journal in self._journals:
            if journal.get("id") == journal_id:
                return journal
        return None

    # ------------------------------------------------------------------
    # ISSN / name lookup
    # ------------------------------------------------------------------

    def find_by_issn(self, issn: str) -> Optional[Dict[str, Any]]:
        """Find a journal by exact ISSN or eISSN match."""
        issn = issn.strip()
        for journal in self._journals:
            if journal.get("issn") == issn or journal.get("eissn") == issn:
                return journal
        return None

    def find_by_name(self, name: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Find journals by name.

        Args:
            name: The name string to search for.
            fuzzy: If True, partial matching against name_cn or name_en.
                   If False, exact match only.
        """
        name_lower = name.lower()
        results = []
        for journal in self._journals:
            name_cn = journal.get("name_cn", "")
            name_en = journal.get("name_en", "")
            if fuzzy:
                if name_lower in name_cn.lower() or name_lower in name_en.lower():
                    results.append(journal)
            else:
                if name == name_cn or name == name_en:
                    results.append(journal)
        return results

    # ------------------------------------------------------------------
    # CAS zone lookup
    # ------------------------------------------------------------------

    def get_cas_zone(self, journal_name_or_issn: str) -> Optional[Dict]:
        """Query the CAS (Chinese Academy of Sciences) zone for a journal.

        Looks up by journal name first, then falls back to ISSN matching
        across all entries in _cas_zones.

        Returns a dict like:
            {"zone": 1, "category": "计算机科学", "top_journal": True}
        or None if not found.
        """
        query = journal_name_or_issn.strip()

        # Direct name key lookup
        if query in self._cas_zones:
            return self._cas_zones[query]

        # Case-insensitive name lookup
        query_lower = query.lower()
        for name, info in self._cas_zones.items():
            if name.lower() == query_lower:
                return info

        # ISSN / eISSN lookup across all entries
        for name, info in self._cas_zones.items():
            if info.get("issn") == query or info.get("eissn") == query:
                return info

        return None

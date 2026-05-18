"""Language quality checker for academic papers.

Verifies:
- Academic style and tone
- Grammar and spelling (rule-based)
- Redundancy and wordiness
"""

import re
from typing import Any, Dict, List, Set


class LanguageChecker:
    """Check language quality of academic writing."""

    # Non-academic words/phrases that should be flagged
    INFORMAL_WORDS = {
        "really", "very", "pretty", "quite", "rather", "fairly",
        "thing", "stuff", "lots", "big", "small", "good", "bad",
        "get", "got", "getting", "like", "kind of", "sort of",
        "etc.", "and so on", "and so forth",
    }

    # Academic weak phrases
    WEAK_PHRASES = [
        "it is interesting to note that",
        "it should be noted that",
        "it is important to mention",
        "in this day and age",
        "due to the fact that",
        "in order to",
        "for the purpose of",
        "in spite of the fact that",
        "at this point in time",
        "in the event that",
    ]

    # Chinese banned words for academic body text
    BANNED_CONNECTORS = {
        "首先", "其次", "再次", "最后", "第一", "第二", "第三",
        "其一", "其二", "其三", "一是", "二是", "三是",
    }
    BANNED_HYPE_WORDS = {
        "重构", "重建", "填补空白", "颠覆", "开创性", "里程碑",
        "划时代", "前所未有", "重大突破", "革命性", "独创",
    }
    BANNED_RHETORICAL = {"如何", "何以", "为何", "为什么", "怎能", "岂能", "何尝"}
    FORCED_CONTRAST_RE = re.compile(r"不是[^，。！？\n]{0,20}而是|并非[^，。！？\n]{0,20}而是|与其说[^，。！？\n]{0,20}不如说")
    BULLET_LIST_RE = re.compile(r"(?m)^\s*[-*•]\s+")
    NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.、)）]\s+")

    # Repeated words pattern (same word within 10 words)
    REPEAT_WINDOW = 10

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run language checks."""
        findings = []
        score = 100.0

        text = self._extract_full_text(doc)

        findings.extend(self._check_academic_style(text))
        findings.extend(self._check_weak_phrases(text))
        findings.extend(self._check_redundancy(text))
        findings.extend(self._check_chinese_banned_words(text))
        findings.extend(self._check_forced_contrast(text))
        findings.extend(self._check_bullet_lists(text))
        findings.extend(self._check_rhetorical_questions(text))

        for f in findings:
            sev = f.get("severity", "minor")
            if sev == "major":
                score -= 5
            elif sev == "minor":
                score -= 2

        score = max(0.0, min(100.0, score))

        return {
            "dimension": "language",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
        }

    def _check_academic_style(self, text: str) -> List[Dict]:
        """Check for informal language."""
        issues = []
        sentences = re.split(r'[.!?。！？]', text)

        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if word in self.INFORMAL_WORDS:
                    issues.append({
                        "type": "informal_language",
                        "severity": "minor",
                        "location": "text",
                        "detail": f"Informal word '{word}' detected: ...{sentence.strip()[-80:]}...",
                        "suggestion": f"Consider replacing '{word}' with a more formal alternative.",
                    })

        return issues[:10]  # Limit to avoid flooding

    def _check_weak_phrases(self, text: str) -> List[Dict]:
        """Check for weak/vague academic phrases."""
        issues = []
        lower_text = text.lower()

        for phrase in self.WEAK_PHRASES:
            count = lower_text.count(phrase)
            if count > 0:
                issues.append({
                    "type": "weak_phrase",
                    "severity": "minor",
                    "location": "text",
                    "detail": f"Weak phrase '{phrase}' appears {count} time(s).",
                    "suggestion": "Use more direct and specific language.",
                })

        return issues

    def _check_redundancy(self, text: str) -> List[Dict]:
        """Check for repeated words and redundant expressions."""
        issues = []
        sentences = re.split(r'[.!?。！？]', text)

        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for i in range(len(words)):
                for j in range(i + 1, min(i + self.REPEAT_WINDOW, len(words))):
                    if words[i] == words[j] and len(words[i]) > 3:
                        issues.append({
                            "type": "repeated_word",
                            "severity": "minor",
                            "location": "text",
                            "detail": f"Word '{words[i]}' repeated within short distance.",
                            "suggestion": "Vary vocabulary or restructure the sentence.",
                        })
                        break  # Only flag once per word per sentence

        return issues[:10]

    def _check_chinese_banned_words(self, text: str) -> List[Dict]:
        """Check for banned Chinese words in academic body text."""
        issues = []
        for word in self.BANNED_CONNECTORS:
            if word in text:
                issues.append({
                    "type": "banned_connector",
                    "severity": "major",
                    "location": "text",
                    "detail": f"机械连接词「{word}」出现在正文中。",
                    "suggestion": "删除该词，改为自然段落过渡。",
                })
        for word in self.BANNED_HYPE_WORDS:
            if word in text:
                issues.append({
                    "type": "banned_hype_word",
                    "severity": "major",
                    "location": "text",
                    "detail": f"夸张吹嘘词「{word}」出现在正文中。",
                    "suggestion": "使用平实学术表达替代。",
                })
        return issues

    def _check_forced_contrast(self, text: str) -> List[Dict]:
        """Check for forced contrast patterns like '不是...而是...'."""
        issues = []
        for m in self.FORCED_CONTRAST_RE.finditer(text):
            issues.append({
                "type": "forced_contrast",
                "severity": "major",
                "location": "text",
                "detail": f"强制转折句式「{m.group()}」出现在正文中。",
                "suggestion": "改为直接判断，避免'不是...而是...'式表达。",
            })
        return issues

    def _check_bullet_lists(self, text: str) -> List[Dict]:
        """Check for bullet points or numbered lists in body text."""
        issues = []
        bullet_count = len(self.BULLET_LIST_RE.findall(text))
        numbered_count = len(self.NUMBERED_LIST_RE.findall(text))
        if bullet_count > 0:
            issues.append({
                "type": "bullet_list_in_body",
                "severity": "major",
                "location": "text",
                "detail": f"检测到 {bullet_count} 处无序列表（项目符号）出现在正文中。",
                "suggestion": "将列表内容改写为连续段落化文本。",
            })
        if numbered_count > 0:
            issues.append({
                "type": "numbered_list_in_body",
                "severity": "major",
                "location": "text",
                "detail": f"检测到 {numbered_count} 处有序列表（编号）出现在正文中。",
                "suggestion": "将列表内容改写为连续段落化文本。",
            })
        return issues

    def _check_rhetorical_questions(self, text: str) -> List[Dict]:
        """Check for rhetorical questions in body text."""
        issues = []
        for word in self.BANNED_RHETORICAL:
            # Look for the word followed by a question mark within 15 chars
            pattern = re.compile(re.escape(word) + r"[^，。！？\n]{0,15}[?？]")
            for m in pattern.finditer(text):
                issues.append({
                    "type": "rhetorical_question",
                    "severity": "major",
                    "location": "text",
                    "detail": f"反问句「{m.group()}」出现在正文中。",
                    "suggestion": "改为直接陈述句。",
                })
        return issues[:5]  # Limit to avoid flooding

    @staticmethod
    def _extract_full_text(doc: Dict) -> str:
        parts = []
        if doc.get("abstract"):
            parts.append(doc["abstract"])
        for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x)):
            s = doc["sections"][key]
            if s.get("content"):
                parts.append(s["content"])
        return "\n".join(parts)

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No language quality issues detected."
        return f"Found {len(findings)} language issues."

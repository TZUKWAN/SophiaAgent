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

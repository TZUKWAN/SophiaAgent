"""Citation checker for academic papers.

Verifies:
- Citation format correctness (APA / GB-T-7714)
- In-text citations match reference list
- Phantom citation detection
"""

import re
from typing import Any, Dict, List, Set, Tuple


class CitationChecker:
    """Check citation format, completeness, and validity."""

    # Patterns for in-text citations
    APA_PAREN_CITE = re.compile(r'\(([A-Z][a-zA-Z\-]+(?:\s+&\s+[A-Z][a-zA-Z\-]+)?,\s*\d{4}[a-z]?)\)')
    APA_NARRATIVE_CITE = re.compile(r'([A-Z][a-zA-Z\-]+(?:\s+&\s+[A-Z][a-zA-Z\-]+)?)\s*\(\s*(\d{4}[a-z]?)\s*\)')
    GB_NUM_CITE = re.compile(r'\[(\d+)\]')
    GB_AUTHOR_YEAR = re.compile(r'（([^（）]+\d{4}[^（）]*)）')

    def check(self, doc: Dict[str, Any], citation_style: str = "apa7") -> Dict[str, Any]:
        """Run citation checks."""
        findings = []
        score = 100.0

        refs = doc.get("references", [])
        full_text = self._extract_full_text(doc)

        # 1. In-text vs reference list matching
        mismatches = self._check_list_matching(full_text, refs, citation_style)
        findings.extend(mismatches)
        score -= len(mismatches) * 10

        # 2. Format validation
        format_issues = self._check_format(refs, citation_style)
        findings.extend(format_issues)
        score -= len(format_issues) * 3

        # 3. Phantom detection
        phantom_issues = self._detect_phantoms(full_text, refs, citation_style)
        findings.extend(phantom_issues)
        score -= len(phantom_issues) * 15

        score = max(0.0, min(100.0, score))

        return {
            "dimension": "citations",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
            "stats": {
                "references_count": len(refs),
                "in_text_citations": len(self._extract_in_text_citations(full_text, citation_style)),
            },
        }

    def _check_list_matching(self, text: str, refs: List[str], style: str) -> List[Dict]:
        """Check that every in-text citation has a matching reference."""
        issues = []
        in_text = self._extract_in_text_citations(text, style)

        # Build reference index
        ref_index = set()
        for ref in refs:
            # Extract author and year from reference
            author_year = self._extract_author_year_from_ref(ref)
            if author_year:
                ref_index.add(author_year)

        for cite in in_text:
            if not self._matches_reference(cite, ref_index, style):
                issues.append({
                    "type": "citation_not_in_list",
                    "severity": "major",
                    "location": "text",
                    "detail": f"In-text citation '{cite}' does not match any entry in the reference list.",
                    "suggestion": "Add the missing reference or correct the citation.",
                })

        return issues

    def _check_format(self, refs: List[str], style: str) -> List[Dict]:
        """Check reference format compliance."""
        issues = []
        for i, ref in enumerate(refs):
            if style == "apa7":
                issues.extend(self._check_apa_format(ref, i + 1))
            elif style == "gb-t-7714-2015":
                issues.extend(self._check_gb_format(ref, i + 1))
        return issues

    def _check_apa_format(self, ref: str, idx: int) -> List[Dict]:
        """Check APA 7th edition format."""
        issues = []
        # Author (Year). Title. Journal, Vol(Issue), pages. DOI
        if not re.search(r'\(\d{4}[a-z]?\)', ref):
            issues.append({
                "type": "missing_year",
                "severity": "minor",
                "location": f"Reference [{idx}]",
                "detail": f"APA format requires year in parentheses: {ref[:80]}",
                "suggestion": "Add publication year in parentheses (e.g., Smith (2020)).",
            })
        if not re.search(r'[A-Z][a-z]+', ref):
            issues.append({
                "type": "missing_author",
                "severity": "major",
                "location": f"Reference [{idx}]",
                "detail": f"Reference appears to lack author names: {ref[:80]}",
                "suggestion": "Add author names at the beginning of the reference.",
            })
        return issues

    def _check_gb_format(self, ref: str, idx: int) -> List[Dict]:
        """Check GB/T 7714-2015 format."""
        issues = []
        # [序号] 作者. 题名[J]. 刊名, 年, 卷(期): 页码.
        if not re.search(r'\[\w+\]', ref) and not re.search(r'[\[（]\d{4}[\]）]', ref):
            issues.append({
                "type": "missing_year",
                "severity": "minor",
                "location": f"Reference [{idx}]",
                "detail": f"GB/T 7714 format requires year: {ref[:80]}",
                "suggestion": "Add publication year to the reference.",
            })
        return issues

    def _detect_phantoms(self, text: str, refs: List[str], style: str) -> List[Dict]:
        """Detect potentially fabricated references."""
        issues = []
        for i, ref in enumerate(refs):
            # Check for common fabrication patterns
            if len(ref) < 20:
                issues.append({
                    "type": "truncated_reference",
                    "severity": "major",
                    "location": f"Reference [{i + 1}]",
                    "detail": f"Reference is unusually short: {ref}",
                    "suggestion": "Complete the reference with all required fields.",
                })

            # Check for placeholder text
            placeholders = ["placeholder", "xxx", "待补充", "待定", "unknown", "???"]
            if any(p.lower() in ref.lower() for p in placeholders):
                issues.append({
                    "type": "placeholder_reference",
                    "severity": "fatal",
                    "location": f"Reference [{i + 1}]",
                    "detail": f"Reference contains placeholder text: {ref}",
                    "suggestion": "Replace placeholder with complete citation information.",
                })

        return issues

    def _extract_in_text_citations(self, text: str, style: str) -> List[str]:
        """Extract in-text citation strings."""
        citations = []
        if style == "apa7":
            for m in self.APA_PAREN_CITE.findall(text):
                citations.append(m)
            for m in self.APA_NARRATIVE_CITE.findall(text):
                citations.append(f"{m[0]} ({m[1]})")
        else:
            for m in self.GB_NUM_CITE.findall(text):
                citations.append(f"[{m}]")
            for m in self.GB_AUTHOR_YEAR.findall(text):
                citations.append(m)
        return citations

    def _extract_author_year_from_ref(self, ref: str) -> Tuple[str, str]:
        """Extract (author, year) tuple from a reference string."""
        year_match = re.search(r'(\d{4})', ref)
        year = year_match.group(1) if year_match else ""
        # Extract first author surname
        author_match = re.search(r'^([A-Z][a-zA-Z\-]+)', ref)
        if not author_match:
            author_match = re.search(r'^([^,\.\s]+)', ref)
        author = author_match.group(1) if author_match else ""
        return (author.lower(), year)

    def _matches_reference(self, cite: str, ref_index: Set[Tuple[str, str]], style: str) -> bool:
        """Check if an in-text citation matches any reference."""
        # Extract author and year from citation
        year_match = re.search(r'(\d{4}[a-z]?)', cite)
        year = year_match.group(1) if year_match else ""
        author_match = re.search(r'^([A-Z][a-zA-Z\-]+)', cite)
        author = author_match.group(1).lower() if author_match else ""

        for ref_author, ref_year in ref_index:
            if ref_year == year and (author in ref_author or ref_author in author):
                return True
        return False

    @staticmethod
    def _extract_full_text(doc: Dict) -> str:
        parts = []
        for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x)):
            s = doc["sections"][key]
            if s.get("content"):
                parts.append(s["content"])
        return "\n".join(parts)

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No citation issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major."

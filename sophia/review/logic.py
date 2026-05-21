"""Logic consistency checker for academic papers.

Verifies:
- Methodology matches research question
- Evidence supports conclusions
- Hypothesis-method-conclusion chain is intact
- Argument chain extraction and visualization
- Fallacy detection
- Evidence sufficiency assessment
- Argument structure scoring
"""

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class LogicChecker:
    """Check logical consistency of paper arguments."""

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run full logic checks with enhanced scoring."""
        findings = []

        findings.extend(self._check_methodology_match(doc))
        findings.extend(self._check_evidence_support(doc))
        findings.extend(self._check_conclusion_chain(doc))
        findings.extend(self._detect_fallacies(doc))
        findings.extend(self._check_evidence_sufficiency(doc))

        # Use new scoring system
        score_result = self._score_argument_structure(doc, findings)

        return {
            "dimension": "logic",
            "score": float(round(score_result["total_score"], 1)),
            "pass": score_result["total_score"] >= 70,
            "findings": findings,
            "summary": self._summary(findings),
            "score_breakdown": score_result["breakdown"],
            "grade": score_result["grade"],
        }

    # ------------------------------------------------------------------
    # F-1: Argument chain extraction
    # ------------------------------------------------------------------

    def extract_argument_chains(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract premise->reasoning->conclusion chains from text.

        Uses keyword-based heuristic extraction (LLM path can be added later).
        """
        text = self._extract_full_text(doc)
        sentences = self._split_sentences(text)
        chains = []

        # Heuristic: look for argument indicator patterns
        # Premise indicators: "因为", "由于", "鉴于", "基于", "鉴于..."
        # Conclusion indicators: "因此", "所以", "从而", "表明", "说明"
        # Evidence indicators: "数据显示", "研究表明", "调查发现"
        premise_markers = ["因为", "由于", "鉴于", "基于", "考虑到", "根据", "依据", "鉴于"]
        conclusion_markers = ["因此", "所以", "从而", "表明", "说明", "证明", "证实了", "意味着", "可以得出"]
        evidence_markers = ["数据显示", "研究表明", "调查发现", "实验表明", "统计结果显示", "访谈结果显示", "文献显示", "实证结果表明", "根据"]

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue

            # Try to find a sentence that contains both premise and conclusion markers
            has_premise = any(m in sent for m in premise_markers)
            has_conclusion = any(m in sent for m in conclusion_markers)
            has_evidence = any(m in sent for m in evidence_markers)

            if has_premise and has_conclusion:
                # Split by conclusion marker
                for marker in conclusion_markers:
                    if marker in sent:
                        parts = sent.split(marker, 1)
                        if len(parts) == 2:
                            premise = parts[0].strip()
                            conclusion = parts[1].strip()
                            chains.append({
                                "chain_id": len(chains),
                                "premise": premise[:200],
                                "reasoning": f"通过 {marker} 推导",
                                "conclusion": conclusion[:200],
                                "evidence": has_evidence,
                                "weak": False,
                                "sentence_index": i,
                            })
                            break

            elif has_conclusion and has_evidence:
                chains.append({
                    "chain_id": len(chains),
                    "premise": "",
                    "reasoning": f"基于证据推导",
                    "conclusion": sent[:200],
                    "evidence": True,
                    "weak": True,  # No explicit premise
                    "sentence_index": i,
                })

            # Cross-sentence: conclusion in current sentence, premise in previous 1-2 sentences
            elif has_conclusion and not has_premise:
                for j in range(max(0, i - 2), i):
                    prev_sent = sentences[j].strip()
                    if len(prev_sent) < 5:
                        continue
                    prev_has_premise = any(m in prev_sent for m in premise_markers)
                    prev_has_evidence = any(m in prev_sent for m in evidence_markers)
                    if prev_has_premise:
                        # Find which conclusion marker triggered
                        for marker in conclusion_markers:
                            if marker in sent:
                                conclusion_part = sent.split(marker, 1)[-1].strip() if marker in sent else sent
                                chains.append({
                                    "chain_id": len(chains),
                                    "premise": prev_sent[:200],
                                    "reasoning": f"通过 {marker} 推导（跨句）",
                                    "conclusion": conclusion_part[:200],
                                    "evidence": has_evidence or prev_has_evidence,
                                    "weak": False,
                                    "sentence_index": i,
                                })
                                break
                        break

        return chains

    def visualize_argument_chains(
        self,
        chains: List[Dict[str, Any]],
        format: str = "mermaid",
    ) -> str:
        """Generate visual representation of argument chains.

        Supports: 'mermaid', 'dot' (GraphViz).
        """
        if format == "mermaid":
            lines = ["graph LR"]
            for chain in chains:
                cid = chain["chain_id"]
                premise = self._escape_mermaid(chain.get("premise", "前提"))[:50]
                conclusion = self._escape_mermaid(chain.get("conclusion", "结论"))[:50]
                reasoning = self._escape_mermaid(chain.get("reasoning", "推理"))[:30]

                node_p = f"P{cid}[{premise}]"
                node_r = f"R{cid}({reasoning})"
                node_c = f"C{cid}[{conclusion}]"

                weak = chain.get("weak", False)
                style = "-.->" if weak else "-->"

                lines.append(f"    {node_p} {style} {node_r}")
                lines.append(f"    {node_r} {style} {node_c}")

                if chain.get("evidence"):
                    lines.append(f"    E{cid}((证据)) -.-> {node_r}")

            return "\n".join(lines)
        elif format == "dot":
            lines = ["digraph Arguments {"]
            for chain in chains:
                cid = chain["chain_id"]
                premise = self._escape_dot(chain.get("premise", "前提"))[:50]
                conclusion = self._escape_dot(chain.get("conclusion", "结论"))[:50]
                reasoning = self._escape_dot(chain.get("reasoning", "推理"))[:30]

                lines.append(f'  p{cid} [label="{premise}", shape=box];')
                lines.append(f'  r{cid} [label="{reasoning}", shape=diamond];')
                lines.append(f'  c{cid} [label="{conclusion}", shape=ellipse];')

                weak = chain.get("weak", False)
                style = "dashed" if weak else "solid"
                lines.append(f'  p{cid} -> r{cid} [style={style}];')
                lines.append(f'  r{cid} -> c{cid} [style={style}];')

                if chain.get("evidence"):
                    lines.append(f'  e{cid} [label="证据", shape=circle];')
                    lines.append(f'  e{cid} -> r{cid} [style=dotted];')
            lines.append("}")
            return "\n".join(lines)
        else:
            return json.dumps(chains, ensure_ascii=False, indent=2)

    @staticmethod
    def _escape_mermaid(text: str) -> str:
        return text.replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")

    @staticmethod
    def _escape_dot(text: str) -> str:
        return text.replace('"', '\\"').replace("\n", " ")

    # ------------------------------------------------------------------
    # F-2: Fallacy detection
    # ------------------------------------------------------------------

    _FALLACY_PATTERNS: List[Dict[str, Any]] = [
        {
            "type": "滑坡谬误",
            "en": "slippery_slope",
            "patterns": [r"一旦.*就.*会.*导致.*最终.*", r"如果.*那么.*就会.*进而.*", r"一发不可收拾", r"后果不堪设想"],
            "description": "从一个小前提推导出极端后果，忽略了中间可能存在的干预因素。",
        },
        {
            "type": "稻草人谬误",
            "en": "straw_man",
            "patterns": [r"有人认为.*但实际上.*", r"表面上.*实际上.*", r"看似.*实则.*"],
            "description": "歪曲对方的观点使其更容易攻击。",
        },
        {
            "type": "虚假因果",
            "en": "false_causation",
            "patterns": [r"因为.*所以.*", r".*导致.*", r".*造成.*", r".*引起.*", r".*使得.*"],
            "description": "将时间上的先后关系误认为因果关系。",
            "context_check": True,  # Need context to confirm false causation
        },
        {
            "type": "诉诸权威",
            "en": "appeal_to_authority",
            "patterns": [r"根据.{0,10}(指出|认为|表示|强调).{0,50}(所以|因此|表明)", r"正如.*所说.*", r".*权威.*认为.*"],
            "description": "以权威身份替代论证本身。",
        },
        {
            "type": "以偏概全",
            "en": "hasty_generalization",
            "patterns": [r"都.*是.*", r"所有.*都.*", r"无一例外.*", r" invariably ", r" always ", r" every "],
            "description": "基于有限的样本得出普遍性的结论。",
        },
        {
            "type": "循环论证",
            "en": "circular_reasoning",
            "patterns": [r".*就是.*因为.*", r".*之所以.*是因为.*就是.*"],
            "description": "结论被包含在前提之中。",
        },
        {
            "type": "虚假两难",
            "en": "false_dilemma",
            "patterns": [r"要么.*要么.*", r"不是.*就是.*", r"非.*即.*", r"要么.*否则.*"],
            "description": "将复杂问题简化为非此即彼的二元选择。",
        },
        {
            "type": "诉诸情感",
            "en": "appeal_to_emotion",
            "patterns": [r"令人痛心.*", r"令人愤慨.*", r"令人担忧.*", r"震惊.*", r"愤怒.*"],
            "description": "用情感反应替代理性论证。",
        },
        {
            "type": "相关当因果",
            "en": "correlation_causation",
            "patterns": [r"随着.*的增加.*也.*增加", r".*与.*呈正相关.*说明.*", r".*与.*呈负相关.*说明.*"],
            "description": "将相关性误认为因果性。",
        },
        {
            "type": "采樱桃谬误",
            "en": "cherry_picking",
            "patterns": [r"正如.*所示.*", r".*数据.*支持.*", r".*例子.*证明.*"],
            "description": "只选择支持自己观点的证据，忽略反面证据。",
            "context_check": True,
        },
        {
            "type": "诉诸传统",
            "en": "appeal_to_tradition",
            "patterns": [r"历来.*", r"自古以来.*", r"传统上.*", r"一直以来.*"],
            "description": "以'历来如此'作为论据。",
        },
        {
            "type": "幸存者偏差",
            "en": "survivorship_bias",
            "patterns": [r"成功者.*", r"杰出.*案例.*", r"优秀.*代表.*"],
            "description": "只关注成功案例而忽略失败案例。",
        },
    ]

    def _detect_fallacies(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect common logical fallacies in text."""
        text = self._extract_full_text(doc)
        sentences = self._split_sentences(text)
        findings = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10:
                continue

            for fallacy in self._FALLACY_PATTERNS:
                for pattern in fallacy["patterns"]:
                    if re.search(pattern, sent):
                        # For some fallacies, do additional context check
                        if fallacy.get("context_check"):
                            # Skip if the sentence also contains qualifying language
                            if any(q in sent for q in ["可能", "也许", "不一定", "有待验证", "进一步研究"]):
                                continue

                        findings.append({
                            "type": "fallacy",
                            "subtype": fallacy["type"],
                            "subtype_en": fallacy["en"],
                            "severity": "minor",
                            "location": f"sentence_{i}",
                            "detail": f"检测到可能的 {fallacy['type']}：{sent[:100]}",
                            "explanation": fallacy["description"],
                            "suggestion": f"检查是否确实存在 {fallacy['type']}，考虑补充反例或修正论证。",
                        })
                        break  # One match per fallacy type per sentence

        return findings

    # ------------------------------------------------------------------
    # F-3: Evidence sufficiency assessment
    # ------------------------------------------------------------------

    def _check_evidence_sufficiency(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if core arguments have sufficient evidence."""
        text = self._extract_full_text(doc)
        sections = doc.get("sections", {})

        # Find core arguments (sentences with conclusion markers)
        conclusion_markers = ["因此", "所以", "表明", "说明", "证明", "证实了", "结论是", "我们认为", "研究发现"]
        argument_sentences = []
        sentences = self._split_sentences(text)

        for i, sent in enumerate(sentences):
            for marker in conclusion_markers:
                if marker in sent and len(sent) > 8:
                    argument_sentences.append((i, sent, marker))
                    break

        # Check evidence types in nearby sentences
        evidence_types = {
            "数据": ["数据", "统计", "百分比", "显著", "p值", "回归系数"],
            "引文": ["根据", "研究表明", "学者指出", "已有研究"],
            "案例": ["例如", "以.*为例", "个案", "案例分析"],
            "逻辑推理": ["由此可见", "同理", "既然.*那么"],
        }

        findings = []
        for idx, sent, marker in argument_sentences[:20]:  # Check top 20 arguments
            # Look at context (±2 sentences)
            start = max(0, idx - 2)
            end = min(len(sentences), idx + 3)
            context = " ".join(sentences[start:end])

            found_types = []
            for etype, keywords in evidence_types.items():
                if any(kw in context for kw in keywords):
                    found_types.append(etype)

            is_naked = len(found_types) == 0
            sufficiency_score = min(100, len(found_types) * 25)

            if is_naked:
                findings.append({
                    "type": "naked_argument",
                    "severity": "major",
                    "location": f"sentence_{idx}",
                    "detail": f"论点缺乏证据支撑：{sent[:120]}",
                    "suggestion": "为该论点补充数据、引文、案例或逻辑推理证据。",
                    "sufficiency_score": 0,
                    "is_naked": True,
                })
            elif sufficiency_score < 50:
                findings.append({
                    "type": "weak_evidence",
                    "severity": "minor",
                    "location": f"sentence_{idx}",
                    "detail": f"论点证据类型单一（{', '.join(found_types)}）：{sent[:120]}",
                    "suggestion": "尝试补充更多类型的证据（数据+引文+案例）以增强说服力。",
                    "sufficiency_score": sufficiency_score,
                    "is_naked": False,
                })

        return findings

    # ------------------------------------------------------------------
    # F-4: Argument structure scoring
    # ------------------------------------------------------------------

    def _score_argument_structure(
        self,
        doc: Dict[str, Any],
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score argument structure across 5 dimensions (each 0-20, total 100)."""
        text = self._extract_full_text(doc)

        # Dimension 1: 论点明确性 (20)
        conclusion_markers = ["因此", "所以", "表明", "说明", "证明", "结论是", "我们认为", "研究发现"]
        argument_count = sum(1 for m in conclusion_markers if m in text)
        clarity_score = min(20, argument_count * 2) if argument_count < 10 else 20

        # Dimension 2: 证据充分性 (20)
        evidence_finding = [f for f in findings if f["type"] in ("naked_argument", "weak_evidence")]
        naked_count = sum(1 for f in evidence_finding if f.get("is_naked"))
        weak_count = sum(1 for f in evidence_finding if not f.get("is_naked", True))
        evidence_score = max(0, 20 - naked_count * 5 - weak_count * 2)

        # Dimension 3: 推理连贯性 (20)
        chains = self.extract_argument_chains(doc)
        weak_chains = sum(1 for c in chains if c.get("weak"))
        coherence_score = min(20, len(chains) * 2) if weak_chains == 0 else max(0, min(20, len(chains) * 2) - weak_chains * 3)

        # Dimension 4: 反驳考虑 (20)
        counter_keywords = ["但是", "然而", "不过", "另一方面", "尽管如此", "存在局限性", "有待进一步"]
        counter_count = sum(text.count(k) for k in counter_keywords)
        counter_score = min(20, counter_count * 3) if counter_count < 7 else 20

        # Dimension 5: 结论谨慎性 (20)
        cautious_markers = ["可能", "也许", "一定程度上", "倾向于", "有待", "初步", "探索性"]
        strong_markers = ["证明了", "证实了", "毫无疑问", "必然", "绝对"]
        cautious_count = sum(text.count(k) for k in cautious_markers)
        strong_count = sum(text.count(k) for k in strong_markers)
        caution_score = min(20, cautious_count * 2) if strong_count < 3 else max(0, 20 - strong_count * 3)

        total_score = clarity_score + evidence_score + coherence_score + counter_score + caution_score

        if total_score >= 90:
            grade = "A"
        elif total_score >= 70:
            grade = "B"
        elif total_score >= 50:
            grade = "C"
        else:
            grade = "D"

        breakdown = {
            "clarity": {"score": clarity_score, "max": 20, "label": "论点明确性"},
            "evidence": {"score": evidence_score, "max": 20, "label": "证据充分性"},
            "coherence": {"score": coherence_score, "max": 20, "label": "推理连贯性"},
            "counter": {"score": counter_score, "max": 20, "label": "反驳考虑"},
            "caution": {"score": caution_score, "max": 20, "label": "结论谨慎性"},
        }

        return {
            "total_score": total_score,
            "grade": grade,
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Original checks (preserved)
    # ------------------------------------------------------------------

    def _check_methodology_match(self, doc: Dict) -> List[Dict]:
        """Check if methodology matches research question."""
        issues = []
        text = self._extract_full_text(doc)
        lower_text = text.lower()

        is_causal = any(k in lower_text for k in ("effect", "impact", "因果", "影响", "效应"))
        is_descriptive = any(k in lower_text for k in ("describe", "distribution", "描述", "分布"))
        is_relational = any(k in lower_text for k in ("correlation", "relationship", "相关", "关系"))

        has_experiment = any(k in lower_text for k in ("experiment", "randomized", "实验", "随机"))
        has_did = any(k in lower_text for k in ("difference-in-differences", "双重差分", "did", "双重差分法"))
        has_iv = any(k in lower_text for k in ("instrumental variable", "工具变量", "iv"))
        has_regression = any(k in lower_text for k in ("regression", "回归", "ols"))
        has_correlation = any(k in lower_text for k in ("correlation", "pearson", "spearman", "相关分析"))

        if is_causal and not (has_experiment or has_did or has_iv or has_regression):
            issues.append({
                "type": "methodology_mismatch",
                "severity": "major",
                "location": "Methods section",
                "detail": "Research question implies causal inference, but no causal method is mentioned.",
                "suggestion": "Consider using a causal inference method appropriate for your data and design.",
            })

        if is_relational and not (has_correlation or has_regression):
            issues.append({
                "type": "methodology_mismatch",
                "severity": "minor",
                "location": "Methods section",
                "detail": "Research question asks about relationships, but no correlation or regression analysis is mentioned.",
                "suggestion": "Consider adding correlation analysis or regression to examine relationships.",
            })

        return issues

    def _check_evidence_support(self, doc: Dict) -> List[Dict]:
        """Check if conclusions are supported by evidence in Results."""
        issues = []
        sections = doc.get("sections", {})

        results_text = ""
        discussion_text = ""
        for key, sec in sections.items():
            title_lower = sec.get("title", "").lower()
            content = sec.get("content", "")
            if "result" in title_lower or "结果" in title_lower:
                results_text += "\n" + content
            if "discussion" in title_lower or "讨论" in title_lower:
                discussion_text += "\n" + content

        if not results_text and discussion_text:
            issues.append({
                "type": "missing_evidence",
                "severity": "fatal",
                "location": "Results section",
                "detail": "Discussion section exists but Results section is empty or missing.",
                "suggestion": "Add a Results section with empirical findings before the Discussion.",
            })

        if results_text and discussion_text:
            strong_claims = re.findall(r'\b(prove|demonstrate|confirm|establish|表明|证明|证实)\b', discussion_text, re.I)
            if len(strong_claims) > 3:
                issues.append({
                    "type": "overstated_claims",
                    "severity": "major",
                    "location": "Discussion section",
                    "detail": f"Discussion uses strong causal language ({len(strong_claims)} instances).",
                    "suggestion": "Use more cautious language unless strong causal evidence exists.",
                })

        return issues

    def _check_conclusion_chain(self, doc: Dict) -> List[Dict]:
        """Check hypothesis-method-conclusion chain."""
        issues = []
        text = self._extract_full_text(doc)
        lower_text = text.lower()

        has_hypothesis = any(k in lower_text for k in ("hypothesis", "假设", "h1", "h0"))
        has_method = any(k in lower_text for k in ("method", "方法", "analysis", "分析"))
        has_result = any(k in lower_text for k in ("result", "结果", "finding", "发现"))
        has_conclusion = any(k in lower_text for k in ("conclusion", "结论", "讨论"))

        if has_hypothesis and not has_result:
            issues.append({
                "type": "broken_chain",
                "severity": "fatal",
                "location": "overall",
                "detail": "Hypotheses are stated but no results are reported.",
                "suggestion": "Add a Results section that tests each hypothesis.",
            })

        if has_result and not has_conclusion:
            issues.append({
                "type": "broken_chain",
                "severity": "major",
                "location": "overall",
                "detail": "Results are reported but no Conclusion or Discussion section.",
                "suggestion": "Add a Conclusion or Discussion section.",
            })

        if has_method and not has_result:
            issues.append({
                "type": "broken_chain",
                "severity": "fatal",
                "location": "overall",
                "detail": "Methods are described but no results are presented.",
                "suggestion": "Add a Results section.",
            })

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_full_text(doc: Dict) -> str:
        parts = []
        if doc.get("abstract"):
            parts.append(doc["abstract"])
        for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            s = doc["sections"][key]
            if s.get("content"):
                parts.append(s["content"])
        return "\n".join(parts)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences (Chinese-aware)."""
        # Split by Chinese sentence-ending punctuation and periods
        import re
        # Normalize line breaks
        text = text.replace("\n", " ")
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[。！？.?!])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No logical consistency issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        minor = sum(1 for f in findings if f.get("severity") == "minor")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major, {minor} minor."

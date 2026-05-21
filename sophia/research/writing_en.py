"""Academic English writing engine for SophiaAgent.

Provides: polish, readability analysis, glossary management,
cover letter generation, and review response generation.
All methods accept dict args and return JSON strings.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & data paths
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

_STYLE_PROFILES = {
    "social_science": {
        "sentence_length_target": (20, 30),
        "passive_ratio_target": (0.15, 0.35),
        "paragraph_length_target": (100, 250),
        "flesch_kincaid_target": (15, 25),
        "ttr_target": (0.45, 0.65),
        "awl_target": (0.08, 0.20),
        "connectors": [
            "however", "in contrast", "moreover", "furthermore", "therefore",
            "consequently", "nevertheless", "nonetheless", "in addition",
            "similarly", "conversely", "accordingly", "thus", "hence",
        ],
    },
    "humanities": {
        "sentence_length_target": (25, 40),
        "passive_ratio_target": (0.10, 0.30),
        "paragraph_length_target": (120, 300),
        "flesch_kincaid_target": (18, 30),
        "ttr_target": (0.50, 0.70),
        "awl_target": (0.06, 0.18),
        "connectors": [
            "however", "in contrast", "moreover", "furthermore", "therefore",
            "consequently", "nevertheless", "nonetheless", "in addition",
            "similarly", "conversely", "accordingly", "thus", "hence",
            "notwithstanding", "albeit", "whereas",
        ],
    },
    "education": {
        "sentence_length_target": (18, 28),
        "passive_ratio_target": (0.15, 0.30),
        "paragraph_length_target": (100, 220),
        "flesch_kincaid_target": (14, 22),
        "ttr_target": (0.45, 0.60),
        "awl_target": (0.08, 0.18),
        "connectors": [
            "however", "in contrast", "moreover", "furthermore", "therefore",
            "consequently", "nevertheless", "in addition", "similarly",
            "conversely", "accordingly", "thus",
        ],
    },
    "public_policy": {
        "sentence_length_target": (18, 28),
        "passive_ratio_target": (0.20, 0.40),
        "paragraph_length_target": (100, 220),
        "flesch_kincaid_target": (14, 22),
        "ttr_target": (0.40, 0.55),
        "awl_target": (0.10, 0.22),
        "connectors": [
            "however", "in contrast", "moreover", "furthermore", "therefore",
            "consequently", "nevertheless", "in addition", "similarly",
            "conversely", "accordingly", "thus", "hence",
        ],
    },
}

# Academic Word List (simplified core subset)
_AWL_WORDS = {
    "analysis", "analyse", "analyzed", "analyzing", "approach", "approached",
    "area", "assessment", "assume", "assumed", "authority", "available",
    "benefit", "benefited", "concept", "conclusion", "conclude", "concluded",
    "condition", "consequence", "consequently", "consider", "considered",
    "consist", "consistent", "constitute", "context", "contract", "create",
    "created", "creation", "culture", "cultural", "data", "define", "defined",
    "definition", "derive", "derived", "design", "designed", "despite",
    "determine", "determined", "develop", "developed", "development",
    "device", "devise", "devised", "different", "differentiate", "dimension",
    "distinct", "distinctive", "dominate", "dominated", "economic", "economy",
    "environment", "environmental", "equate", "equated", "equivalent",
    "establish", "established", "estimate", "estimated", "evaluate",
    "evaluated", "evident", "evidence", "export", "exported", "factor",
    "feature", "featured", "federal", "fee", "final", "finally", "finance",
    "financial", "formula", "formulate", "formulated", "function",
    "functional", "functioned", "fund", "funded", "identify", "identified",
    "identify", "income", "indicate", "indicated", "individual", "initial",
    "initially", "instance", "institute", "institution", "invest",
    "invested", "investigate", "investigated", "issue", "issued", "item",
    "job", "journal", "labour", "legal", "legislation", "levy", "levied",
    "locate", "located", "location", "logic", "logical", "major", "manual",
    "margin", "mature", "matured", "media", "method", "methodology",
    "migrate", "migrated", "military", "minimal", "minimize", "minimized",
    "minimum", "ministry", "minor", "mode", "modify", "modified", "monitor",
    "monitored", "network", "normal", "norm", "notion", "objective",
    "obtain", "obtained", "occupy", "occupied", "occur", "occurred",
    "ongoing", "option", "output", "overall", "parallel", "parameter",
    "participate", "participated", "partner", "partnership", "passive",
    "period", "persist", "persisted", "phase", "phenomenon", "philosophy",
    "physical", "plus", "policy", "portion", "potential", "practitioner",
    "precede", "preceded", "precise", "predict", "predicted", "predominant",
    "preliminary", "presume", "presumed", "previous", "previously", "primary",
    "prime", "principal", "principle", "prior", "priority", "proceed",
    "proceeded", "process", "processed", "professional", "prohibit",
    "prohibited", "project", "projected", "promote", "promoted", "proportion",
    "prospect", "protocol", "psychology", "publication", "publish",
    "published", "purchase", "purchased", "pursue", "pursued", "range",
    "ranged", "ratio", "react", "reaction", "recover", "recovered", "refine",
    "refined", "regime", "region", "regional", "register", "registered",
    "regulate", "regulated", "reinforce", "reinforced", "reject", "rejected",
    "relax", "relaxed", "release", "released", "relevant", "reluctance",
    "rely", "relied", "remove", "removed", "require", "required",
    "research", "reside", "resided", "resolve", "resolved", "resource",
    "respond", "responded", "response", "restore", "restored", "restrain",
    "restrained", "restrict", "restricted", "retain", "retained", "reveal",
    "revealed", "revenue", "reverse", "reversed", "revise", "revised",
    "revolution", "rigid", "role", "route", "scenario", "schedule",
    "scheduled", "scheme", "scope", "section", "sector", "secure", "secured",
    "seek", "sought", "select", "selected", "sequence", "series", "sex",
    "shift", "shifted", "significant", "significance", "similar", "simulate",
    "simulated", "site", "so-called", "sole", "somewhat", "source", "specific",
    "specify", "specified", "sphere", "stable", "stability", "statistic",
    "status", "straightforward", "strategy", "stress", "stressed", "structure",
    "structured", "style", "submit", "submitted", "subordinate", "subsequent",
    "subsequently", "subsidy", "substitute", "substituted", "successor",
    "sufficient", "sum", "summary", "supplement", "supplemented", "survey",
    "survive", "survived", "suspect", "suspected", "sustain", "sustained",
    "symbol", "symbolic", "symptom", "synthetic", "system", "systematic",
    "tactic", "tape", "target", "targeted", "task", "team", "technical",
    "technique", "technology", "temporary", "tense", "tension", "term",
    "terminate", "terminated", "text", "theme", "theory", "thereby",
    "thesis", "topic", "trace", "traced", "tradition", "transfer",
    "transferred", "transform", "transformed", "transition", "transmit",
    "transmitted", "transport", "transported", "trend", "trigger",
    "triggered", "ultimate", "ultimately", "undergo", "underwent",
    "underlie", "underlay", "undermine", "undermined", "undertake",
    "undertook", "uniform", "unify", "unified", "unique", "unit", "unite",
    "united", "universal", "universally", "unlike", "unlikely", "utilize",
    "utilized", "valid", "validity", "vary", "varied", "vehicle", "version",
    "via", "violate", "violated", "virtual", "virtually", "visible",
    "vision", "visual", "volume", "voluntary", "welfare", "whereas", "whereby",
    "widespread", "willing", "withdraw", "withdrew", "yield", "yielded",
    "zone",
}

# Passive-voice detection: common past-participle forms
_PAST_PARTICIPLES = {
    "achieved", "adopted", "analyzed", "applied", "argued", "assessed",
    "assumed", "based", "calculated", "carried", "categorized", "caused",
    "changed", "chosen", "classified", "collected", "compared", "completed",
    "computed", "conducted", "confirmed", "considered", "constructed",
    "consulted", "contained", "controlled", "converted", "correlated",
    "created", "defined", "demonstrated", "derived", "described",
    "designated", "designed", "detected", "determined", "developed",
    "deviated", "differentiated", "discovered", "discussed", "distributed",
    "divided", "documented", "employed", "enabled", "encountered",
    "encouraged", "enhanced", "ensured", "established", "estimated",
    "evaluated", "examined", "excluded", "executed", "exhibited",
    "expanded", "expected", "experienced", "explained", "explored",
    "expressed", "extended", "extracted", "facilitated", "found",
    "formed", "formulated", "generated", "given", "grouped", "guided",
    "identified", "implemented", "implied", "improved", "included",
    "increased", "indicated", "inferred", "influenced", "initiated",
    "integrated", "intended", "interpreted", "introduced", "investigated",
    "involved", "isolated", "justified", "labeled", "led", "limited",
    "linked", "located", "made", "maintained", "managed", "matched",
    "measured", "mentioned", "modified", "monitored", "noted", "observed",
    "obtained", "offered", "operated", "organized", "outlined", "perceived",
    "performed", "placed", "planned", "predicted", "prepared", "presented",
    "preserved", "produced", "promoted", "proposed", "provided", "published",
    "purchased", "raised", "randomized", "rated", "reached", "received",
    "recognized", "recommended", "recorded", "recruited", "reduced",
    "referred", "refined", "reflected", "refused", "regarded", "regulated",
    "rejected", "related", "released", "relied", "removed", "repeated",
    "replaced", "replicated", "reported", "represented", "required",
    "researched", "resolved", "responded", "restored", "restricted",
    "resulted", "retained", "retrieved", "revealed", "reviewed", "revised",
    "satisfied", "scheduled", "selected", "separated", "served", "set",
    "settled", "shared", "shown", "signaled", "simulated", "situated",
    "solved", "sorted", "specified", "spent", "spread", "stated",
    "stimulated", "studied", "submitted", "succeeded", "suffered",
    "suggested", "summarized", "supervised", "supported", "supposed",
    "suppressed", "surveyed", "sustained", "taken", "targeted", "tested",
    "titled", "traced", "trained", "transformed", "treated", "triggered",
    "undertaken", "updated", "used", "utilized", "validated", "valued",
    "verified", "viewed", "visited", "weighted", "written",
}

_AUXILIARIES = {"am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "get", "got",
                "getting"}

# Vocab upgrade map (colloquial -> academic)
_VOCAB_UPGRADE = {
    "get": "obtain", "got": "obtained", "getting": "obtaining",
    "give": "provide", "gives": "provides", "gave": "provided",
    "show": "demonstrate", "shows": "demonstrates", "showed": "demonstrated",
    "use": "utilize", "uses": "utilizes", "used": "utilized",
    "make": "produce", "makes": "produces", "made": "produced",
    "do": "perform", "does": "performs", "did": "performed",
    "say": "state", "says": "states", "said": "stated",
    "think": "argue", "thinks": "argues", "thought": "argued",
    "help": "facilitate", "helps": "facilitates", "helped": "facilitated",
    "need": "require", "needs": "requires", "needed": "required",
    "want": "seek", "wants": "seeks", "wanted": "sought",
    "keep": "maintain", "keeps": "maintains", "kept": "maintained",
    "put": "place", "puts": "places", "placed": "placed",
    "tell": "report", "tells": "reports", "told": "reported",
    "find": "identify", "finds": "identifies", "found": "identified",
    "look at": "examine", "looks at": "examines", "looked at": "examined",
    "deal with": "address", "deals with": "addresses", "dealt with": "addressed",
    "carry out": "conduct", "carries out": "conducts", "carried out": "conducted",
    "set up": "establish", "sets up": "establishes", "set up": "established",
    "point out": "indicate", "points out": "indicates", "pointed out": "indicated",
    "talk about": "discuss", "talks about": "discusses", "talked about": "discussed",
    "work out": "resolve", "works out": "resolves", "worked out": "resolved",
    "come up with": "propose", "comes up with": "proposes", "came up with": "proposed",
    "put forward": "advance", "puts forward": "advances", "put forward": "advanced",
    "take into account": "consider", "takes into account": "considers", "took into account": "considered",
    "make sure": "ensure", "makes sure": "ensures", "made sure": "ensured",
    "find out": "determine", "finds out": "determines", "found out": "determined",
    "go up": "increase", "goes up": "increases", "went up": "increased",
    "go down": "decrease", "goes down": "decreases", "went down": "decreased",
    "cut down": "reduce", "cuts down": "reduces", "cut down": "reduced",
    "lots of": "numerous", "a lot of": "numerous", "a lot": "significantly",
    "big": "substantial", "good": "favorable", "bad": "adverse",
    "really": "significantly", "very": "highly", "pretty": "rather",
    "quite": "rather", "rather": "rather", "fairly": "rather",
    "things": "factors", "thing": "factor", "stuff": "material",
    "people": "individuals", "something": "a factor", "someone": "an individual",
    "begin": "commence", "begins": "commences", "began": "commenced",
    "start": "commence", "starts": "commences", "started": "commenced",
    "end": "conclude", "ends": "concludes", "ended": "concluded",
    "change": "alter", "changes": "alters", "changed": "altered",
    "happen": "occur", "happens": "occurs", "happened": "occurred",
    "mean": "signify", "means": "signifies", "meant": "signified",
    "try": "attempt", "tries": "attempts", "tried": "attempted",
    "check": "examine", "checks": "examines", "checked": "examined",
    "see": "observe", "sees": "observes", "saw": "observed",
}

# Redundant phrases
_REDUNDANCIES = {
    "advance planning": "planning",
    "advance warning": "warning",
    "actual fact": "fact",
    "add an additional": "add",
    "all-time record": "record",
    "alternative choice": "alternative",
    "and etc.": "etc.",
    "annual anniversary": "anniversary",
    "as a matter of fact": "in fact",
    "at the present time": "currently",
    "basic fundamentals": "fundamentals",
    "best ever": "best",
    "brief summary": "summary",
    "cancel out": "cancel",
    "careful scrutiny": "scrutiny",
    "close proximity": "proximity",
    "collaborate together": "collaborate",
    "combine together": "combine",
    "completely eliminate": "eliminate",
    "consensus of opinion": "consensus",
    "continue on": "continue",
    "cooperate together": "cooperate",
    "current trend": "trend",
    "depreciate in value": "depreciate",
    "descend down": "descend",
    "different kinds": "kinds",
    "each and every": "each",
    "end result": "result",
    "enter in": "enter",
    "exact same": "same",
    "few in number": "few",
    "final outcome": "outcome",
    "first and foremost": "first",
    "first of all": "first",
    "follow after": "follow",
    "free gift": "gift",
    "future plans": "plans",
    "gather together": "gather",
    "general consensus": "consensus",
    "grow in size": "grow",
    "honest truth": "truth",
    "in order to": "to",
    "in spite of the fact that": "although",
    "in the event that": "if",
    "in the final analysis": "ultimately",
    "in the nature of": "like",
    "in the vicinity of": "near",
    "join together": "join",
    "knowledgeable expert": "expert",
    "lag behind": "lag",
    "later on": "later",
    "lift up": "lift",
    "major breakthrough": "breakthrough",
    "meet together": "meet",
    "merge together": "merge",
    "mix together": "mix",
    "mutual cooperation": "cooperation",
    "new innovation": "innovation",
    "new invention": "invention",
    "null and void": "void",
    "old adage": "adage",
    "oral conversation": "conversation",
    "original source": "source",
    "over again": "again",
    "past experience": "experience",
    "past history": "history",
    "period of time": "period",
    "personal opinion": "opinion",
    "plan ahead": "plan",
    "plan in advance": "plan",
    "please rsvp": "rsvp",
    "postpone until later": "postpone",
    "present time": "present",
    "proceed forward": "proceed",
    "reason is because": "reason is that",
    "reason why": "reason",
    "recur again": "recur",
    "refer back": "refer",
    "reflect back": "reflect",
    "repeat again": "repeat",
    "revert back": "revert",
    "same exact": "same",
    "serious danger": "danger",
    "share in common": "share",
    "shorter in length": "shorter",
    "still remains": "remains",
    "sudden impulse": "impulse",
    "sum total": "total",
    "surrounded on all sides": "surrounded",
    "the fact that": "that",
    "true facts": "facts",
    "unexpected surprise": "surprise",
    "unintended mistake": "mistake",
    "usual custom": "custom",
    "various different": "various",
    "whether or not": "whether",
    "with regard to": "regarding",
    "with respect to": "regarding",
    "with the exception of": "except",
    "written down": "written",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chinglish_patterns() -> List[Dict[str, str]]:
    path = os.path.join(_DATA_DIR, "chinglish_patterns.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _sentences(text: str) -> List[str]:
    """Split text into sentences (naive but functional)."""
    # Preserve abbreviations roughly
    text = re.sub(r'([A-Z]\.)+', lambda m: m.group(0).replace('.', ''), text)
    splits = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in splits if s.strip()]


def _paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _flesch_kincaid(text: str, sentences: List[str], words: List[str]) -> float:
    """Compute Flesch-Kincaid Grade Level."""
    if not sentences or not words:
        return 0.0
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = sum(_count_syllables(w) for w in words) / len(words)
    return 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59


def _count_syllables(word: str) -> int:
    """Rough syllable count."""
    word = word.lower().strip(".,;:!?'")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(1, count)


def _sentence_type(sentence: str) -> str:
    """Classify sentence as simple, compound, complex, compound-complex."""
    s_lower = sentence.lower()
    # Count dependent clause markers
    dependent_markers = ["because", "since", "although", "though", "while", "whereas",
                         "if", "unless", "until", "when", "after", "before", "even though"]
    dependent = sum(1 for m in dependent_markers if m in s_lower)
    # Coordinating conjunctions (not within first word)
    has_coord_conj = any(f" {cc} " in s_lower for cc in ["and", "but", "or", "yet", "so", "for", "nor"])
    # Semicolon indicates compound
    has_semicolon = ";" in sentence
    has_conj = has_coord_conj or has_semicolon

    if dependent == 0 and not has_conj:
        return "simple"
    if dependent == 0 and has_conj:
        return "compound"
    if dependent > 0 and not has_conj:
        return "complex"
    return "compound-complex"


def _is_passive(sentence: str) -> bool:
    """Heuristic passive-voice detection."""
    words = _tokenize(sentence)
    for i, w in enumerate(words):
        if w in _AUXILIARIES and i + 1 < len(words) and words[i + 1] in _PAST_PARTICIPLES:
            return True
    return False


def _extract_noun_phrases(text: str) -> Counter:
    """Extract simple noun phrases (adj + noun or noun + noun)."""
    tokens = _tokenize(text)
    # Very naive: bigrams and trigrams where last word is a noun-like token
    # We approximate nouns as words not in a small stop list and > 3 chars
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "we", "our",
        "us", "i", "me", "my", "he", "him", "his", "she", "her", "you", "your",
    }
    phrases = Counter()
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            phrase = tokens[i:i + n]
            if phrase[-1] in stop_words or len(phrase[-1]) <= 3:
                continue
            if any(w in stop_words for w in phrase[:-1]):
                continue
            phrases[" ".join(phrase)] += 1
    return phrases


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AcademicEnglishEngine:
    """Rule-based academic English writing assistant with optional LLM enhancement."""

    def __init__(self, provider=None):
        self.provider = provider
        self._patterns = _load_chinglish_patterns()

    # =====================================================================
    # C-1: Polish
    # =====================================================================

    def polish(self, text: str, style: str = "social_science", llm: bool = True) -> Dict[str, Any]:
        """Polish academic English with rule-based + optional LLM path.

        Returns dict with: diff, clean_version, stats.
        """
        if style not in _STYLE_PROFILES:
            style = "social_science"

        edits: List[Dict[str, str]] = []
        revised = text

        # 1. Vocab upgrade
        revised, vocab_edits = self._apply_vocab_upgrade(revised)
        edits.extend(vocab_edits)

        # 2. Redundancy removal
        revised, red_edits = self._apply_redundancy_removal(revised)
        edits.extend(red_edits)

        # 3. Chinglish detection
        revised, ching_edits = self._apply_chinglish_detection(revised)
        edits.extend(ching_edits)

        # 4. Academic connectors (suggest missing ones)
        connector_suggestions = self._suggest_connectors(revised, style)

        # 5. Sentence structure adjustments (suggest only, don't auto-rewrite)
        structure_suggestions = self._suggest_sentence_structure(revised, style)

        # LLM path (optional)
        llm_revised = None
        if llm and self.provider is not None:
            try:
                llm_revised = self._llm_polish(text, style)
            except Exception as exc:
                logger.warning("LLM polish failed: %s", exc)

        # Build diff view
        diff = self._build_diff(text, revised, edits)

        result = {
            "style": style,
            "original": text,
            "revised": revised,
            "diff": diff,
            "clean_version": revised,
            "edits": edits,
            "connector_suggestions": connector_suggestions,
            "structure_suggestions": structure_suggestions,
            "stats": {
                "total_edits": len(edits),
                "vocab_upgrades": len([e for e in edits if e.get("category") == "vocab_upgrade"]),
                "redundancy_removals": len([e for e in edits if e.get("category") == "redundancy"]),
                "chinglish_fixes": len([e for e in edits if e.get("category") == "chinglish"]),
            },
        }
        if llm_revised is not None:
            result["llm_revised"] = llm_revised
            result["llm_available"] = True
        else:
            result["llm_available"] = False
        return result

    def _apply_vocab_upgrade(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        edits = []
        revised = text
        # Sort keys by length desc to avoid partial replacements
        for colloquial, academic in sorted(_VOCAB_UPGRADE.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(r'\b' + re.escape(colloquial) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(revised):
                original = match.group(0)
                # Preserve case roughly
                if original[0].isupper():
                    replacement = academic.capitalize()
                else:
                    replacement = academic
                edits.append({
                    "original": original,
                    "revised": replacement,
                    "reason": f"Colloquial -> academic: '{colloquial}' -> '{academic}'",
                    "category": "vocab_upgrade",
                })
            revised = pattern.sub(lambda m: academic.capitalize() if m.group(0)[0].isupper() else academic, revised)
        return revised, edits

    def _apply_redundancy_removal(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        edits = []
        revised = text
        for redundant, concise in sorted(_REDUNDANCIES.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(r'\b' + re.escape(redundant) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(revised):
                edits.append({
                    "original": match.group(0),
                    "revised": concise,
                    "reason": f"Redundancy removal: '{redundant}' -> '{concise}'",
                    "category": "redundancy",
                })
            revised = pattern.sub(concise, revised)
        return revised, edits

    def _apply_chinglish_detection(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        edits = []
        revised = text
        for rule in self._patterns:
            pat = rule.get("pattern", "")
            fix = rule.get("fix", "")
            reason = rule.get("reason", "")
            category = rule.get("category", "chinglish")
            if not pat or not fix:
                continue
            # Simple phrase match (not regex for safety)
            pattern = re.compile(r'\b' + re.escape(pat) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(revised):
                edits.append({
                    "original": match.group(0),
                    "revised": fix,
                    "reason": reason,
                    "category": category,
                })
            # For multi-word replacements, take first option if pipe-separated
            replacement = fix.split("|")[0].strip()
            revised = pattern.sub(replacement, revised)
        return revised, edits

    def _suggest_connectors(self, text: str, style: str) -> List[Dict[str, str]]:
        profile = _STYLE_PROFILES[style]
        present = set()
        text_lower = text.lower()
        for conn in profile["connectors"]:
            if conn in text_lower:
                present.add(conn)
        missing = [c for c in profile["connectors"] if c not in present]
        suggestions = []
        # Suggest top 5 missing connectors
        for conn in missing[:5]:
            suggestions.append({
                "connector": conn,
                "suggestion": f"Consider using '{conn}' to improve logical flow.",
            })
        return suggestions

    def _suggest_sentence_structure(self, text: str, style: str) -> List[Dict[str, str]]:
        sents = _sentences(text)
        suggestions = []
        for i, sent in enumerate(sents):
            word_count = len(_tokenize(sent))
            target = _STYLE_PROFILES[style]["sentence_length_target"]
            if word_count > target[1] + 10:
                suggestions.append({
                    "sentence_index": i,
                    "sentence_preview": sent[:80] + "..." if len(sent) > 80 else sent,
                    "issue": "overlong_sentence",
                    "suggestion": f"Sentence is {word_count} words (target: {target[0]}-{target[1]}). Consider splitting.",
                })
            elif word_count < target[0] - 5:
                suggestions.append({
                    "sentence_index": i,
                    "sentence_preview": sent[:80] + "..." if len(sent) > 80 else sent,
                    "issue": "too_short_sentence",
                    "suggestion": f"Sentence is only {word_count} words. Consider expanding or combining.",
                })
        return suggestions

    def _build_diff(self, original: str, revised: str, edits: List[Dict[str, str]]) -> str:
        """Build a simple line-by-line diff."""
        orig_lines = original.splitlines()
        rev_lines = revised.splitlines()
        diff_lines = []
        for i, (o, r) in enumerate(zip(orig_lines, rev_lines)):
            if o != r:
                diff_lines.append(f"- [{i + 1}] {o}")
                diff_lines.append(f"+ [{i + 1}] {r}")
        return "\n".join(diff_lines)

    def _llm_polish(self, text: str, style: str) -> Optional[str]:
        if self.provider is None:
            return None
        prompt = (
            f"You are an academic English editor. Polish the following text for {style} style. "
            "Improve vocabulary, sentence structure, remove redundancy, and fix any Chinglish. "
            "Return ONLY the polished text, no explanations.\n\n"
            f"{text}"
        )
        try:
            response = self.provider.chat([{"role": "user", "content": prompt}], tools=None)
            return response.content or text
        except Exception:
            return None

    # =====================================================================
    # C-2: Readability Analysis
    # =====================================================================

    def analyze_readability(self, text: str, style: str = "social_science") -> Dict[str, Any]:
        """Analyze readability metrics with benchmark references."""
        if style not in _STYLE_PROFILES:
            style = "social_science"
        profile = _STYLE_PROFILES[style]

        sents = _sentences(text)
        words = _tokenize(text)
        paras = _paragraphs(text)

        if not sents:
            return {"error": "No sentences found in text."}

        sent_lengths = [len(_tokenize(s)) for s in sents]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths)
        sent_len_std = (sum((x - avg_sent_len) ** 2 for x in sent_lengths) / len(sent_lengths)) ** 0.5

        sent_types = Counter(_sentence_type(s) for s in sents)
        total_sents = len(sents)
        sent_type_ratios = {k: round(v / total_sents, 3) for k, v in sent_types.items()}

        passive_count = sum(1 for s in sents if _is_passive(s))
        passive_ratio = passive_count / total_sents

        para_lengths = [len(_tokenize(p)) for p in paras] if paras else [0]
        avg_para_len = sum(para_lengths) / len(para_lengths)

        fk = _flesch_kincaid(text, sents, words)

        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0.0

        awl_count = sum(1 for w in words if w.lower() in _AWL_WORDS)
        awl_coverage = awl_count / len(words) if words else 0.0

        def _metric(name: str, current: float, target: Tuple[float, float], unit: str = "") -> Dict[str, Any]:
            low, high = target
            status = "good" if low <= current <= high else ("high" if current > high else "low")
            suggestion = ""
            if status == "high":
                suggestion = f"Reduce {name} toward {high}{unit}."
            elif status == "low":
                suggestion = f"Increase {name} toward {low}{unit}."
            else:
                suggestion = f"{name} is within target range."
            return {
                "metric": name,
                "current": round(current, 3),
                "target_low": low,
                "target_high": high,
                "status": status,
                "suggestion": suggestion,
            }

        metrics = [
            _metric("avg_sentence_length", avg_sent_len, profile["sentence_length_target"], " words"),
            _metric("sentence_length_std", sent_len_std, (2, 8), " words"),
            _metric("passive_voice_ratio", passive_ratio, profile["passive_ratio_target"]),
            _metric("avg_paragraph_length", avg_para_len, profile["paragraph_length_target"], " words"),
            _metric("flesch_kincaid", fk, profile["flesch_kincaid_target"]),
            _metric("type_token_ratio", ttr, profile["ttr_target"]),
            _metric("awl_coverage", awl_coverage, profile["awl_target"]),
        ]

        return {
            "style": style,
            "total_sentences": total_sents,
            "total_words": len(words),
            "total_paragraphs": len(paras),
            "sentence_type_distribution": sent_type_ratios,
            "metrics": metrics,
        }

    def diversify_sentences(self, text: str) -> Dict[str, Any]:
        """Detect repeated sentence structures and suggest rewrites."""
        sents = _sentences(text)
        structures = []
        for s in sents:
            words = _tokenize(s)
            # Capture first 3 words as a structural signature
            signature = " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
            structures.append(signature)

        counter = Counter(structures)
        repeated = {sig: count for sig, count in counter.items() if count >= 3}

        suggestions = []
        for sig, count in repeated.items():
            indices = [i for i, s in enumerate(structures) if s == sig]
            suggestions.append({
                "structure": sig,
                "count": count,
                "sentence_indices": indices,
                "suggestion": (
                    f"Sentences starting with '{sig}' appear {count} times. "
                    "Vary sentence openings (e.g., use prepositional phrases, participles, or dependent clauses)."
                ),
            })

        return {
            "total_sentences": len(sents),
            "unique_structures": len(counter),
            "repeated_structures": suggestions,
        }

    # =====================================================================
    # C-3: Glossary Management
    # =====================================================================

    def build_glossary(self, text: str) -> List[Dict[str, Any]]:
        """Auto-extract key terms (noun phrases appearing 3+ times)."""
        phrases = _extract_noun_phrases(text)
        glossary = []
        for phrase, count in phrases.most_common():
            if count >= 3:
                glossary.append({
                    "term": phrase,
                    "frequency": count,
                    "suggested_definition": "",
                })
        return glossary

    def check_consistency(self, text: str, glossary: Optional[List[Dict[str, Any]]] = None, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check spelling consistency, Chinese-English alignment, and abbreviation definitions."""
        issues = []

        # 1. Spelling consistency: common variants
        variant_groups = [
            ["e-commerce", "ecommerce", "electronic commerce"],
            ["email", "e-mail", "electronic mail"],
            ["web site", "website", "web-site"],
            ["co-operation", "cooperation"],
            ["pre-school", "preschool"],
            ["post-modern", "postmodern"],
            ["multi-cultural", "multicultural"],
            ["non-profit", "nonprofit"],
            ["well-being", "wellbeing"],
            ["data base", "database"],
            ["open source", "open-source"],
            ["machine learning", "machine-learning"],
            ["artificial intelligence", "artificial-intelligence"],
            ["public policy", "public-policy"],
            ["social science", "social-science"],
            ["high school", "high-school"],
        ]
        text_lower = text.lower()
        for group in variant_groups:
            found = [v for v in group if v in text_lower]
            if len(found) > 1:
                issues.append({
                    "type": "spelling_inconsistency",
                    "variants_found": found,
                    "suggestion": f"Choose one form and use consistently (recommended: '{group[0]}').",
                })

        # 2. Chinese-English alignment: detect Chinese characters mixed with English
        chinese_chars = re.findall(r'[一-鿿]+', text)
        if chinese_chars:
            issues.append({
                "type": "chinese_english_alignment",
                "chinese_segments": chinese_chars[:10],
                "suggestion": "Ensure Chinese terms are properly translated or annotated with English equivalents.",
            })

        # 3. Abbreviation first-use definition
        # Pattern: ALLCAPS words of 2-5 letters
        abbrev_pattern = re.compile(r'\b[A-Z]{2,5}\b')
        abbreviations = set(abbrev_pattern.findall(text))
        # Exclude common words
        common_acronyms = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
                           "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
                           "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY",
                           "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO",
                           "USE", "USA", "UK", "EU", "UN", "GDP", "CEO", "CFO", "CTO"}
        abbreviations -= common_acronyms
        for abbr in abbreviations:
            # Check if defined earlier in text: pattern like "Abbreviation (ABBR)" or "ABBR (Abbreviation)"
            defined = bool(re.search(rf'\b{re.escape(abbr)}\s*\([^)]+\)', text) or
                           re.search(rf'\([^)]{{5,50}}\)\s*{re.escape(abbr)}\b', text))
            if not defined:
                issues.append({
                    "type": "abbreviation_undefined",
                    "abbreviation": abbr,
                    "suggestion": f"Define '{abbr}' on first use, e.g., '... structural equation modeling (SEM) ...'.",
                })

        # 4. Check against persistent glossary
        if workspace:
            glossary_path = os.path.join(workspace, ".sophia", "glossary.json")
            if os.path.exists(glossary_path):
                with open(glossary_path, "r", encoding="utf-8") as f:
                    saved_glossary = json.load(f)
                saved_terms = {entry.get("term", "").lower() for entry in saved_glossary}
                current_terms = {entry.get("term", "").lower() for entry in (glossary or [])}
                missing_from_glossary = saved_terms - current_terms
                if missing_from_glossary:
                    issues.append({
                        "type": "glossary_deviation",
                        "missing_terms": sorted(missing_from_glossary),
                        "suggestion": "Terms previously in glossary no longer appear in text.",
                    })

        return issues

    def save_glossary(self, workspace: str, glossary: List[Dict[str, Any]]) -> None:
        """Persist glossary to workspace/.sophia/glossary.json."""
        path = os.path.join(workspace, ".sophia", "glossary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)

    # =====================================================================
    # C-4: Cover Letter & Review Response
    # =====================================================================

    def generate_cover_letter(self, paper_meta: Dict[str, Any], journal: Dict[str, Any]) -> str:
        """Generate a cover letter for journal submission."""
        template_path = os.path.join(_TEMPLATES_DIR, "cover_letter.txt")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            template = self._default_cover_letter_template()

        editor = journal.get("editor_name", "Editor")
        journal_name = journal.get("name", "the Journal")
        scope = journal.get("scope", "")

        title = paper_meta.get("title", "")
        authors = paper_meta.get("authors", "")
        abstract = paper_meta.get("abstract", "")
        keywords = ", ".join(paper_meta.get("keywords", []))
        highlights = "\n".join(f"- {h}" for h in paper_meta.get("highlights", []))

        letter = template.format(
            editor_name=editor,
            journal_name=journal_name,
            journal_scope=scope,
            paper_title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            highlights=highlights,
        )
        return letter

    def _default_cover_letter_template(self) -> str:
        return (
            "Dear {editor_name},\n\n"
            "We wish to submit our manuscript entitled \"{paper_title}\" for consideration "
            "for publication in {journal_name}.\n\n"
            "{journal_scope}\n\n"
            "Our study makes the following contributions:\n"
            "{highlights}\n\n"
            "This work falls within the scope of {journal_name} because it addresses "
            "core themes central to the journal's readership.\n\n"
            "All authors have approved the manuscript and agree with its submission to "
            "{journal_name}. The manuscript has not been published previously and is not "
            "under consideration elsewhere.\n\n"
            "We appreciate your time and consideration.\n\n"
            "Sincerely,\n"
            "{authors}\n"
        )

    def generate_review_response(self, review_comments: List[Dict[str, Any]],
                                 author_revisions: List[Dict[str, Any]]) -> str:
        """Generate a structured response to reviewer comments."""
        template_path = os.path.join(_TEMPLATES_DIR, "review_response.txt")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            template = self._default_review_response_template()

        # Build response blocks
        blocks = []
        revision_map = {r.get("comment_id"): r for r in author_revisions}

        for comment in review_comments:
            cid = comment.get("comment_id", "")
            ctext = comment.get("comment_text", "")
            rev = revision_map.get(cid, {})
            response_text = rev.get("response", "We thank the reviewer for this comment.")
            changes = rev.get("changes", "")

            blocks.append(
                f"### Reviewer Comment {cid}\n\n"
                f"**Comment:** {ctext}\n\n"
                f"**Response:** {response_text}\n\n"
                f"**Changes made:** {changes}\n"
            )

        body = "\n\n".join(blocks)
        return template.format(response_blocks=body)

    def _default_review_response_template(self) -> str:
        return (
            "Dear Editor and Reviewers,\n\n"
            "We thank you for the constructive feedback on our manuscript. "
            "We have carefully addressed all comments and revised the manuscript accordingly. "
            "Below, we provide point-by-point responses to each comment.\n\n"
            "{response_blocks}\n\n"
            "We believe the revised manuscript is now significantly improved and hope it meets "
            "the standards for publication.\n\n"
            "Sincerely,\n"
            "The Authors\n"
        )


# ---------------------------------------------------------------------------
# Public wrapper functions (accept dict args, return JSON strings)
# ---------------------------------------------------------------------------

def en_polish(args: Dict[str, Any], workspace: str = "", provider=None) -> str:
    text = args.get("text", "")
    style = args.get("style", "social_science")
    use_llm = args.get("llm", True)
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine(provider=provider)
    result = engine.polish(text, style=style, llm=use_llm)
    return json.dumps(result, ensure_ascii=False)


def en_readability(args: Dict[str, Any], workspace: str = "") -> str:
    text = args.get("text", "")
    style = args.get("style", "social_science")
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    result = engine.analyze_readability(text, style=style)
    return json.dumps(result, ensure_ascii=False)


def en_diversify_sentences(args: Dict[str, Any], workspace: str = "") -> str:
    text = args.get("text", "")
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    result = engine.diversify_sentences(text)
    return json.dumps(result, ensure_ascii=False)


def en_glossary_build(args: Dict[str, Any], workspace: str = "") -> str:
    text = args.get("text", "")
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    glossary = engine.build_glossary(text)
    if workspace:
        engine.save_glossary(workspace, glossary)
    return json.dumps({
        "glossary": glossary,
        "total_terms": len(glossary),
        "saved_to": os.path.join(workspace, ".sophia", "glossary.json") if workspace else None,
    }, ensure_ascii=False)


def en_consistency_check(args: Dict[str, Any], workspace: str = "") -> str:
    text = args.get("text", "")
    glossary = args.get("glossary", None)
    if not text:
        return json.dumps({"error": "text is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    issues = engine.check_consistency(text, glossary=glossary, workspace=workspace)
    return json.dumps({
        "issues_found": len(issues),
        "issues": issues,
    }, ensure_ascii=False)


def en_cover_letter(args: Dict[str, Any], workspace: str = "") -> str:
    paper_meta = args.get("paper_meta", {})
    journal = args.get("journal", {})
    if not paper_meta or not journal:
        return json.dumps({"error": "paper_meta and journal are required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    letter = engine.generate_cover_letter(paper_meta, journal)
    return json.dumps({
        "cover_letter": letter,
        "word_count": len(letter.split()),
    }, ensure_ascii=False)


def en_review_response(args: Dict[str, Any], workspace: str = "") -> str:
    review_comments = args.get("review_comments", [])
    author_revisions = args.get("author_revisions", [])
    if not review_comments:
        return json.dumps({"error": "review_comments is required"}, ensure_ascii=False)
    engine = AcademicEnglishEngine()
    response = engine.generate_review_response(review_comments, author_revisions)
    return json.dumps({
        "review_response": response,
        "word_count": len(response.split()),
        "comments_addressed": len(review_comments),
    }, ensure_ascii=False)

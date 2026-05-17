"""Qualitative analysis engine: thematic, grounded theory, content, sentiment.

Pure-computation module with optional LLM provider integration.  All public
methods accept ``args: dict`` and return ``str`` (JSON).
"""

from __future__ import annotations

import json
import logging
import math
import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from sophia.research._input import resolve_parent_ids

# ---------------------------------------------------------------------------
# Optional sentiment dependency
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Stop-word lists (English minimal + extended)
# ---------------------------------------------------------------------------
_EN_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "if", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "up",
    "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "any", "also", "am", "s", "t", "don", "didn",
    "doesn", "isn", "wasn", "weren", "hasn", "haven", "hadn", "won",
    "wouldn", "couldn", "shouldn", "mustn", "ll", "ve", "re", "d", "m",
    "not", "get", "got", "much", "like", "well", "still", "even",
    "going", "go", "went", "one", "two", "make", "made", "know",
    "think", "say", "said", "see", "come", "take", "want", "give",
    "use", "find", "tell", "ask", "work", "seem", "feel", "try",
    "leave", "call", "keep", "let", "begin", "show", "hear", "play",
    "run", "move", "live", "believe", "hold", "bring", "happen",
    "really", "thing", "things", "something", "anything", "everything",
    "nothing", "many", "way", "back", "being", "because", "however",
}

# Simple positive / negative word lists for fallback sentiment
_POSITIVE_WORDS: Set[str] = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "happy", "glad", "pleased", "satisfied", "enjoy", "enjoyed",
    "best", "better", "beautiful", "nice", "awesome", "perfect",
    "outstanding", "superb", "brilliant", "positive", "success",
    "successful", "achieve", "achievement", "improve", "improved",
    "impressive", "remarkable", "strong", "strength", "benefit",
    "beneficial", "advantage", "helpful", "supportive", "effective",
    "efficient", "innovative", "creative", "valuable", "important",
    "significant", "exciting", "inspiring", "motivated", "confident",
    "optimistic", "enthusiastic", "delighted", "thrilled", "grateful",
    "thankful", "appreciate", "recommend", "recommended", "favorite",
    "pleasant", "comfortable", "reliable", "trust", "trustworthy",
}

_NEGATIVE_WORDS: Set[str] = {
    "bad", "terrible", "horrible", "awful", "worst", "worse", "hate",
    "sad", "angry", "disappointed", "frustrated", "annoyed", "upset",
    "poor", "negative", "fail", "failure", "failed", "wrong", "error",
    "problem", "issue", "difficult", "difficulty", "hard", "struggle",
    "struggling", "painful", "boring", "bored", "confused", "confusing",
    "useless", "worthless", "waste", "ugly", "disgusting", "annoying",
    "uncomfortable", "unhappy", "dissatisfied", "complaint", "complain",
    "unreliable", "slow", "expensive", "overpriced", "broken", "damage",
    "damaged", "weak", "weakness", "lack", "missing", "absent", "fear",
    "afraid", "worried", "worry", "anxious", "anxiety", "stress",
    "stressful", "tired", "exhausted", "hopeless", "helpless",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    return [w for w in text.split() if w.strip()]


def _remove_stopwords(tokens: List[str], stopwords: Set[str] = _EN_STOPWORDS) -> List[str]:
    """Remove stopwords from a token list."""
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def _extract_noun_phrases(text: str) -> List[str]:
    """Extract simple noun phrases (consecutive capitalized words or short
    word groups that look like noun phrases) without POS tagger.
    Fallback heuristic: bigrams and trigrams of non-stopwords."""
    tokens = _tokenize(text)
    clean = _remove_stopwords(tokens)
    phrases: List[str] = []
    # Bigrams
    for i in range(len(clean) - 1):
        phrases.append(f"{clean[i]} {clean[i+1]}")
    # Trigrams
    for i in range(len(clean) - 2):
        phrases.append(f"{clean[i]} {clean[i+1]} {clean[i+2]}")
    # Unigrams too
    phrases.extend(clean)
    return phrases


# ======================================================================
# QualitativeEngine
# ======================================================================

class QualitativeEngine:
    """Qualitative research methods.  Uses LLM for some methods, NLP for others."""

    def __init__(self, provider=None, store=None, guard=None):
        self.provider = provider
        self.store = store
        self.guard = guard

    # ------------------------------------------------------------------
    # ResultStore plumbing
    # ------------------------------------------------------------------

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky values (text lists, coder lists) with summary strings."""
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list):
                if len(v) > 80:
                    clean[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], (list, tuple)):
                    total = sum(len(row) if hasattr(row, "__len__") else 1 for row in v)
                    if total > 200:
                        clean[k] = f"<nested list outer={len(v)} total={total}>"
                    else:
                        clean[k] = v
                elif v and isinstance(v[0], str):
                    # Per-text content can be long; summarise if total chars huge
                    total_chars = sum(len(s) for s in v)
                    if total_chars > 4000:
                        clean[k] = f"<list of {len(v)} strings, total_chars={total_chars}>"
                    else:
                        clean[k] = v
                else:
                    clean[k] = v
            elif isinstance(v, dict):
                total = sum(len(x) if hasattr(x, "__len__") else 1 for x in v.values())
                if total > 200 or len(v) > 50:
                    clean[k] = f"<dict keys={len(v)} total={total}>"
                else:
                    clean[k] = v
            elif isinstance(v, str) and len(v) > 2000:
                clean[k] = f"<str len={len(v)}>"
            else:
                clean[k] = v
        return clean

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Persist a successful result to the store and embed result_id."""
        if "error" in result:
            return _json(result)
        if self.store is None:
            return _json(result)
        parents = resolve_parent_ids(args)
        sanitized = self._sanitize_params(args)
        rid = self.store.store(
            result,
            kind="result",
            tool=tool_name,
            params=sanitized,
            parents=parents,
        )
        result = {**result, "result_id": rid}
        return _json(result)

    # ------------------------------------------------------------------
    # Thematic analysis
    # ------------------------------------------------------------------

    def thematic(self, args: dict) -> str:
        """Automated thematic analysis.

        Parameters
        ----------
        args : dict
            texts : list of str
                Text segments to analyse.
            approach : str
                ``inductive`` (default) or ``deductive``.
            n_themes : int, optional
                Number of themes to extract.  Auto-detected if not given.
            existing_themes : list of str
                For deductive approach: predefined theme labels.
            language : str
                Language code (default ``en``).

        Returns
        -------
        str
            JSON with ``themes``, ``coded_segments``, ``theme_frequencies``.
        """
        texts: List[str] = args.get("texts", [])
        if not texts:
            return _json({"error": "No texts provided for thematic analysis."})

        approach: str = str(args.get("approach", "inductive")).lower()
        n_themes = args.get("n_themes")
        existing_themes: List[str] = args.get("existing_themes", [])
        language: str = str(args.get("language", "en"))

        review = args.get("review", True)
        # ---- LLM path ----
        if self.provider is not None:
            inner = self._thematic_llm(texts, approach, n_themes, existing_themes, language, review=review)
        else:
            # ---- Fallback: keyword frequency + co-occurrence clustering ----
            inner = self._thematic_keyword(texts, approach, n_themes, existing_themes)
        parsed = json.loads(inner)
        try:
            n_themes = len(parsed.get("themes", []))
            approach = parsed.get("approach", "inductive")
            parsed["apa"] = (
                f"A thematic analysis ({approach}) of {len(texts)} texts identified {n_themes} themes."
            )
        except Exception:
            pass
        return self._final(args, parsed, "research_thematic")

    # ------------------------------------------------------------------
    # Iterative LLM thematic analysis (4-pass)
    # ------------------------------------------------------------------

    def _prompt_open_code(self, text: str, language: str) -> str:
        return (
            f"You are a qualitative research assistant. Perform open coding on the following text. "
            f"Identify all relevant codes (concepts/ideas) present in the text. "
            f"Respond in JSON format with key 'codes' containing a list of code strings.\n\n"
            f"Text: {text[:500]}"
        )

    def _parse_codes(self, content: str) -> List[str]:
        try:
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                parsed = json.loads(m.group())
                return parsed.get("codes", [])
        except Exception:
            pass
        # Fallback: extract quoted strings or bullet items
        codes = re.findall(r'["\']([^"\']+)["\']', content)
        if not codes:
            codes = re.findall(r'[-*]\s*(.+)', content)
        return [c.strip() for c in codes if len(c.strip()) > 1]

    def _prompt_consolidate(self, codes: List[str], language: str) -> str:
        codes_json = json.dumps(codes, ensure_ascii=False)
        return (
            f"You are a qualitative research assistant. Consolidate the following open codes. "
            f"Merge semantically identical or highly similar codes into unified codes. "
            f"Respond in JSON format with key 'code_map' as a dict where each key is an original code "
            f"and each value is the consolidated code it maps to.\n\n"
            f"Codes: {codes_json}"
        )

    def _parse_code_map(self, content: str) -> Dict[str, str]:
        try:
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                parsed = json.loads(m.group())
                cm = parsed.get("code_map", {})
                if cm:
                    return {str(k): str(v) for k, v in cm.items()}
        except Exception:
            pass
        return {}

    def _prompt_themes(self, consolidated_codes: List[str], pass1_entries: List[dict],
                       code_map: Dict[str, str], language: str, n_themes: Optional[int],
                       existing_themes: List[str]) -> str:
        if existing_themes:
            theme_instr = f"Use these predefined themes: {json.dumps(existing_themes, ensure_ascii=False)}."
        elif n_themes:
            theme_instr = f"Generate exactly {n_themes} themes."
        else:
            theme_instr = "Generate the main themes that best capture the data."
        codes_json = json.dumps(sorted(set(consolidated_codes)), ensure_ascii=False)
        return (
            f"You are a qualitative research assistant. Based on the following consolidated codes, "
            f"generate themes for thematic analysis. {theme_instr} "
            f"For each theme provide: label, description, and keywords. "
            f"Respond in JSON format with key 'themes' as a list of objects with keys "
            f"'id', 'label', 'description', 'keywords' (list of strings).\n\n"
            f"Consolidated codes: {codes_json}"
        )

    def _parse_themes(self, content: str) -> List[dict]:
        try:
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                parsed = json.loads(m.group())
                themes = parsed.get("themes", [])
                if themes:
                    return themes
        except Exception:
            pass
        return []

    def _prompt_review(self, themes: List[dict]) -> str:
        themes_json = json.dumps(themes, ensure_ascii=False)
        return (
            f"You are a qualitative research assistant. Review the following themes. "
            f"Evaluate whether any themes should be split, merged, or renamed. "
            f"Respond in JSON format with keys: 'confidence' (0-1), 'suggestions' (list of strings), "
            f"'apply' (boolean: whether changes should be applied).\n\n"
            f"Themes: {themes_json}"
        )

    def _parse_review(self, content: str) -> dict:
        try:
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                parsed = json.loads(m.group())
                return {
                    "confidence": parsed.get("confidence", 0.5),
                    "suggestions": parsed.get("suggestions", []),
                    "apply": parsed.get("apply", False),
                }
        except Exception:
            pass
        return {"confidence": 0.5, "suggestions": [], "apply": False}

    def _apply_review(self, themes: List[dict], suggestions: dict) -> List[dict]:
        # For now, return themes as-is; future: implement automated merge/split
        return themes

    def _link_quotes(self, texts: List[str], pass1_entries: List[dict],
                     code_map: Dict[str, str], themes: List[dict]) -> List[dict]:
        """Link text segments to themes via consolidated codes."""
        theme_code_map: Dict[str, int] = {}
        for theme in themes:
            tid = theme.get("id", theme.get("label", ""))
            for kw in theme.get("keywords", []):
                theme_code_map[kw.lower()] = tid

        coded_segments = []
        for entry in pass1_entries:
            i = entry["text_index"]
            codes = entry["codes"]
            mapped_codes = [code_map.get(c, c) for c in codes]
            matched_themes = []
            for mc in mapped_codes:
                for theme in themes:
                    tid = theme.get("id", theme.get("label", ""))
                    if any(kw.lower() in mc.lower() for kw in theme.get("keywords", [])):
                        if tid not in matched_themes:
                            matched_themes.append(tid)
            if matched_themes:
                excerpt = texts[i][:200] + ("..." if len(texts[i]) > 200 else "")
                coded_segments.append({
                    "text_index": i,
                    "theme_ids": matched_themes,
                    "excerpt": excerpt,
                })
        return coded_segments

    def _export_codebook(self, themes: List[dict], code_map: Dict[str, str]) -> dict:
        codebook = {}
        for theme in themes:
            tid = theme.get("id", theme.get("label", ""))
            codebook[str(tid)] = {
                "label": theme.get("label", ""),
                "description": theme.get("description", ""),
                "keywords": theme.get("keywords", []),
                "source_codes": [k for k, v in code_map.items() if v == theme.get("label", "")],
            }
        return codebook

    def _thematic_llm(self, texts, approach, n_themes, existing_themes, language, review=True):
        """Iterative LLM thematic analysis (3-4 pass).

        Pass 1: Open coding per text.
        Pass 2: Code consolidation.
        Pass 3: Theme generation.
        Pass 4 (optional): Self-review.
        """
        if self.provider is None:
            return self._thematic_keyword(texts, approach, n_themes, existing_themes)

        # Pass 1: Open coding
        pass1_entries = []
        failures = []
        for i, text in enumerate(texts):
            prompt = self._prompt_open_code(text, language)
            codes = []
            for attempt in range(2):
                try:
                    resp = self.provider.chat([{"role": "user", "content": prompt}])
                    codes = self._parse_codes(resp.content or "")
                    if codes:
                        break
                except Exception as e:
                    if attempt == 1:
                        failures.append((i, str(e)))
            pass1_entries.append({"text_index": i, "codes": codes})

        if len(failures) > len(texts) * 0.5:
            logger.warning(f"LLM coding failed for {len(failures)}/{len(texts)} texts; falling back to keyword method")
            return self._thematic_keyword(texts, approach, n_themes, existing_themes)

        # Pass 2: Code consolidation
        all_codes = sorted(set(c for entry in pass1_entries for c in entry["codes"]))
        code_map = {}
        if all_codes:
            try:
                consol_prompt = self._prompt_consolidate(all_codes, language)
                consol_resp = self.provider.chat([{"role": "user", "content": consol_prompt}])
                code_map = self._parse_code_map(consol_resp.content or "")
            except Exception as e:
                logger.warning(f"Code consolidation failed: {e}; using identity mapping")
                code_map = {c: c for c in all_codes}
        if not code_map:
            code_map = {c: c for c in all_codes}

        consolidated_codes = sorted(set(code_map.values()))

        # Pass 3: Theme generation
        themes = []
        if consolidated_codes:
            try:
                theme_prompt = self._prompt_themes(
                    consolidated_codes, pass1_entries, code_map, language,
                    n_themes, existing_themes
                )
                theme_resp = self.provider.chat([{"role": "user", "content": theme_prompt}])
                themes = self._parse_themes(theme_resp.content or "")
            except Exception as e:
                logger.warning(f"Theme generation failed: {e}")

        if not themes:
            logger.warning("No themes generated by LLM; falling back to keyword method")
            return self._thematic_keyword(texts, approach, n_themes, existing_themes)

        # Assign integer IDs if missing
        for idx, theme in enumerate(themes):
            if "id" not in theme:
                theme["id"] = idx

        # Pass 4: Self-review (optional)
        if review:
            try:
                review_prompt = self._prompt_review(themes)
                review_resp = self.provider.chat([{"role": "user", "content": review_prompt}])
                review_result = self._parse_review(review_resp.content or "")
                if review_result.get("apply"):
                    themes = self._apply_review(themes, review_result)
            except Exception as e:
                logger.warning(f"Theme review failed: {e}")

        # Link quotes to themes
        coded_segments = self._link_quotes(texts, pass1_entries, code_map, themes)

        # Compute frequencies
        freq: Dict[str, int] = Counter()
        for seg in coded_segments:
            for tid in seg["theme_ids"]:
                freq[str(tid)] += 1

        result = {
            "themes": themes,
            "coded_segments": coded_segments,
            "theme_frequencies": dict(freq),
            "code_map": code_map,
            "codebook": self._export_codebook(themes, code_map),
            "pass1_failures": len(failures),
            "iterations": 4 if review else 3,
            "method": "llm_iterative",
        }
        return _json(result)

    def _thematic_keyword(self, texts, approach, n_themes, existing_themes):
        """Fallback thematic analysis using keyword frequency + co-occurrence."""
        # Tokenize all texts
        all_tokens: List[str] = []
        doc_tokens: List[List[str]] = []
        for t in texts:
            tokens = _remove_stopwords(_tokenize(t))
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)

        if not all_tokens:
            return _json({"error": "No meaningful tokens found after removing stopwords."})

        # Word frequencies across all texts
        word_freq = Counter(all_tokens)

        # Build co-occurrence within window
        window_size = 5
        co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tokens in doc_tokens:
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + window_size, len(tokens))):
                    w1, w2 = tokens[i], tokens[j]
                    if w1 != w2:
                        co_occurrence[w1][w2] += 1
                        co_occurrence[w2][w1] += 1

        # For deductive: use existing themes
        if approach == "deductive" and existing_themes:
            themes = []
            coded_segments = []
            theme_keywords: Dict[int, List[str]] = {}
            for idx, theme_label in enumerate(existing_themes):
                theme_words = _remove_stopwords(_tokenize(theme_label))
                theme_keywords[idx] = theme_words
                themes.append({
                    "id": idx,
                    "label": theme_label,
                    "description": f"Deductive theme: {theme_label}",
                    "keywords": theme_words,
                })
            for i, tokens in enumerate(doc_tokens):
                matched_themes = []
                for tidx, kws in theme_keywords.items():
                    if any(kw in tokens for kw in kws):
                        matched_themes.append(tidx)
                if matched_themes:
                    excerpt = texts[i][:200] + ("..." if len(texts[i]) > 200 else "")
                    coded_segments.append({
                        "text_index": i,
                        "theme_ids": matched_themes,
                        "excerpt": excerpt,
                    })
                else:
                    # Assign to closest theme by word overlap
                    best_theme = 0
                    best_overlap = 0
                    for tidx, kws in theme_keywords.items():
                        overlap = len(set(tokens) & set(kws))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_theme = tidx
                    if best_overlap > 0:
                        excerpt = texts[i][:200] + ("..." if len(texts[i]) > 200 else "")
                        coded_segments.append({
                            "text_index": i,
                            "theme_ids": [best_theme],
                            "excerpt": excerpt,
                        })

            freq: Dict[str, int] = Counter()
            for seg in coded_segments:
                for tid in seg["theme_ids"]:
                    freq[str(tid)] += 1
            return _json({
                "themes": themes,
                "coded_segments": coded_segments,
                "theme_frequencies": dict(freq),
                "approach": "deductive",
                "method": "keyword_matching",
            })

        # Inductive: cluster keywords into themes
        top_words = [w for w, _ in word_freq.most_common(min(100, len(word_freq)))]

        # Simple clustering: greedily build clusters by co-occurrence strength
        used: Set[str] = set()
        clusters: List[List[str]] = []

        for word in top_words:
            if word in used:
                continue
            # Start a new cluster with this word
            cluster = [word]
            used.add(word)
            # Add top co-occurring words
            co_words = sorted(
                co_occurrence[word].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for cw, count in co_words:
                if cw not in used and count >= 2:
                    cluster.append(cw)
                    used.add(cw)
                if len(cluster) >= 8:
                    break
            clusters.append(cluster)

        # Limit themes
        if n_themes is not None:
            clusters = clusters[:n_themes]
        elif len(clusters) > 8:
            clusters = clusters[:8]

        # Build theme objects
        themes = []
        for idx, cluster in enumerate(clusters):
            label = " / ".join(cluster[:3]).title()
            desc_words = ", ".join(cluster[:5])
            themes.append({
                "id": idx,
                "label": label,
                "description": f"Theme related to: {desc_words}",
                "keywords": cluster[:10],
            })

        # Code segments
        coded_segments = []
        for i, tokens in enumerate(doc_tokens):
            token_set = set(tokens)
            matched = []
            for theme in themes:
                overlap = len(token_set & set(theme["keywords"]))
                if overlap >= 1:
                    matched.append(theme["id"])
            if matched:
                excerpt = texts[i][:200] + ("..." if len(texts[i]) > 200 else "")
                coded_segments.append({
                    "text_index": i,
                    "theme_ids": matched,
                    "excerpt": excerpt,
                })

        freq = Counter()
        for seg in coded_segments:
            for tid in seg["theme_ids"]:
                freq[str(tid)] += 1

        return _json({
            "themes": themes,
            "coded_segments": coded_segments,
            "theme_frequencies": dict(freq),
            "approach": "inductive",
            "method": "keyword_co-occurrence",
        })

    # ------------------------------------------------------------------
    # Content analysis
    # ------------------------------------------------------------------

    def content(self, args: dict) -> str:
        """Content analysis (keyword frequency, concept network).

        Parameters
        ----------
        args : dict
            texts : list of str
            keywords : list of str, optional (extract if not given)
            min_freq : int (default 2)
            window : int (co-occurrence window, default 5)

        Returns
        -------
        str
            JSON with ``word_frequencies``, ``keyword_frequencies``,
            ``co_occurrence_matrix``, ``key_concepts``.
        """
        texts: List[str] = args.get("texts", [])
        if not texts:
            return _json({"error": "No texts provided for content analysis."})

        keywords: Optional[List[str]] = args.get("keywords")
        min_freq: int = int(args.get("min_freq", 2))
        window: int = int(args.get("window", 5))

        # Tokenize all texts
        all_tokens: List[str] = []
        doc_tokens: List[List[str]] = []
        for t in texts:
            tokens = _tokenize(t)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)

        # Word frequencies (all words, not just non-stopwords)
        word_freq: Counter = Counter(all_tokens)

        # Filter by min_freq
        filtered_freq = {w: c for w, c in word_freq.items() if c >= min_freq}

        # Keyword frequencies
        keyword_freq: Dict[str, int] = {}
        if keywords:
            kw_set = {k.lower() for k in keywords}
            keyword_freq = {k: word_freq.get(k.lower(), 0) for k in keywords}
        else:
            # Extract top keywords (non-stopword, min_freq)
            top_kws = []
            for w, c in word_freq.most_common(50):
                if w not in _EN_STOPWORDS and len(w) > 2 and c >= min_freq:
                    top_kws.append(w)
            keyword_freq = {w: word_freq[w] for w in top_kws[:20]}
            keywords = top_kws[:20]

        # Co-occurrence matrix within sliding window
        co_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        target_words = set(keyword_freq.keys()) if keyword_freq else set(filtered_freq.keys())
        target_words_lower = {w.lower() for w in target_words}

        for tokens in doc_tokens:
            tokens_lower = [t.lower() for t in tokens]
            for i in range(len(tokens_lower)):
                if tokens_lower[i] not in target_words_lower:
                    continue
                for j in range(i + 1, min(i + window, len(tokens_lower))):
                    if tokens_lower[j] not in target_words_lower:
                        continue
                    if tokens_lower[i] != tokens_lower[j]:
                        co_matrix[tokens_lower[i]][tokens_lower[j]] += 1
                        co_matrix[tokens_lower[j]][tokens_lower[i]] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        co_matrix_json: Dict[str, Dict[str, int]] = {}
        for w1 in co_matrix:
            co_matrix_json[w1] = dict(co_matrix[w1])

        # Key concepts: words with highest degree centrality in co-occurrence
        centrality: Dict[str, int] = {}
        for w in co_matrix:
            centrality[w] = sum(co_matrix[w].values())
        key_concepts = sorted(centrality, key=centrality.get, reverse=True)[:15]

        result = {
            "word_frequencies": dict(sorted(
                filtered_freq.items(), key=lambda x: x[1], reverse=True
            )[:100]),
            "keyword_frequencies": keyword_freq,
            "co_occurrence_matrix": co_matrix_json,
            "key_concepts": key_concepts,
            "n_documents": len(texts),
            "total_tokens": len(all_tokens),
            "unique_tokens": len(set(all_tokens)),
        }
        try:
            result["apa"] = (
                f"Content analysis of {len(texts)} documents identified {len(set(all_tokens))} unique tokens "
                f"and {len(key_concepts)} key concepts."
            )
        except Exception:
            pass
        return self._final(args, result, "research_content")

    # ------------------------------------------------------------------
    # Grounded theory coding
    # ------------------------------------------------------------------

    def grounded_code(self, args: dict) -> str:
        """Grounded theory coding workflow.

        Parameters
        ----------
        args : dict
            texts : list of str
            stage : str
                ``open``, ``axial``, or ``selective``.
            existing_codes : list of str (for axial / selective)
            axial_core : str (core category for axial)
            language : str

        Returns
        -------
        str
            JSON with codes, code_frequencies, categories, relationships,
            core_category depending on stage.
        """
        texts: List[str] = args.get("texts", [])
        if not texts:
            return _json({"error": "No texts provided for grounded theory coding."})

        stage: str = str(args.get("stage", "open")).lower()
        existing_codes: List[str] = args.get("existing_codes", [])
        axial_core: Optional[str] = args.get("axial_core")
        language: str = str(args.get("language", "en"))

        if stage == "open":
            inner = self._grounded_open(texts, language)
        elif stage == "axial":
            inner = self._grounded_axial(texts, existing_codes, axial_core, language)
        elif stage == "selective":
            inner = self._grounded_selective(texts, existing_codes, language)
        else:
            return _json({"error": f"Unknown stage '{stage}'. Use 'open', 'axial', or 'selective'."})

        parsed = json.loads(inner)
        try:
            stage = parsed.get("stage", "open")
            n_codes = len(parsed.get("codes", []))
            parsed["apa"] = (
                f"Grounded theory coding ({stage}) of {len(texts)} texts yielded {n_codes} codes."
            )
        except Exception:
            pass
        return self._final(args, parsed, "research_grounded_code")

    def _grounded_open(self, texts, language):
        """Open coding: extract initial codes from texts."""
        # ---- LLM path ----
        if self.provider is not None:
            return self._grounded_open_llm(texts, language)

        # ---- Fallback: keyword extraction ----
        all_codes: List[str] = []
        code_freq: Counter = Counter()

        for t in texts:
            # Extract unigrams and bigrams as potential codes
            tokens = _remove_stopwords(_tokenize(t))
            phrases = _extract_noun_phrases(t)

            # Count unigrams
            for token in tokens:
                code_freq[token] += 1

            # Count bigrams from phrases
            for phrase in phrases:
                code_freq[phrase] += 1

        # Select top codes (minimum frequency 2)
        codes = []
        for code, freq in code_freq.most_common(50):
            if freq >= 2 or len(code_freq) <= 10:
                codes.append({
                    "code": code,
                    "frequency": freq,
                    "type": "in_vivo" if " " in code else "descriptive",
                })

        if not codes:
            # If no codes meet threshold, take top 10 regardless
            for code, freq in code_freq.most_common(10):
                codes.append({
                    "code": code,
                    "frequency": freq,
                    "type": "in_vivo" if " " in code else "descriptive",
                })

        return _json({
            "stage": "open",
            "codes": codes,
            "code_frequencies": {c["code"]: c["frequency"] for c in codes},
            "n_texts": len(texts),
            "method": "keyword_extraction",
        })

    def _grounded_open_llm(self, texts, language):
        """Open coding with LLM."""
        prompt = (
            "You are a qualitative researcher performing open coding in grounded theory.\n"
            "For each text segment, generate concise codes that capture key concepts.\n\n"
            "Respond in JSON format:\n"
            '  "codes": [{"code": "...", "frequency": N, "type": "descriptive|in_vivo|..."}]\n'
            '  "coded_segments": [{"text_index": N, "codes": ["..."]}]\n\n'
            "Texts:\n"
        )
        for i, t in enumerate(texts):
            excerpt = t[:400] + ("..." if len(t) > 400 else "")
            prompt += f"[{i}] {excerpt}\n"

        try:
            response = self.provider.chat([{"role": "user", "content": prompt}])
            content = response.content or ""
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
                freq = Counter()
                for seg in parsed.get("coded_segments", []):
                    for c in seg.get("codes", []):
                        freq[c] += 1
                parsed["code_frequencies"] = dict(freq)
                parsed["stage"] = "open"
                parsed["method"] = "llm_coding"
                return _json(parsed)
        except Exception:
            pass

        return self._grounded_open(texts, language)

    def _grounded_axial(self, texts, existing_codes, axial_core, language):
        """Axial coding: organize codes into categories around a core category."""
        if not existing_codes:
            # Run open coding first to get codes
            open_result = json.loads(self._grounded_open(texts, language))
            existing_codes = [c["code"] for c in open_result.get("codes", [])]

        if not existing_codes:
            return _json({"error": "No codes available for axial coding."})

        # ---- LLM path ----
        if self.provider is not None:
            prompt = (
                "You are a qualitative researcher performing axial coding in grounded theory.\n"
                "Organize these codes into categories around a core phenomenon.\n\n"
                f"Codes: {json.dumps(existing_codes)}\n"
            )
            if axial_core:
                prompt += f"Core category: {axial_core}\n"

            prompt += (
                "\nRespond in JSON format:\n"
                '  "categories": [{"name": "...", "codes": [...], "description": "..."}]\n'
                '  "relationships": [{"from": "...", "to": "...", "type": "causal|contextual|..."}]\n'
                '  "core_category": "..."\n'
            )
            try:
                response = self.provider.chat([{"role": "user", "content": prompt}])
                content = response.content or ""
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    parsed["stage"] = "axial"
                    parsed["method"] = "llm_coding"
                    return _json(parsed)
            except Exception:
                pass

        # ---- Fallback: cluster codes by word overlap ----
        categories: Dict[str, List[str]] = defaultdict(list)
        used_codes: Set[str] = set()

        # Simple clustering: group codes that share words
        code_words: Dict[str, Set[str]] = {}
        for code in existing_codes:
            code_words[code] = set(_tokenize(code)) - _EN_STOPWORDS

        cluster_id = 0
        for code in existing_codes:
            if code in used_codes:
                continue
            cluster_name = code.title()
            categories[cluster_name] = [code]
            used_codes.add(code)

            # Find related codes
            for other in existing_codes:
                if other in used_codes:
                    continue
                overlap = len(code_words[code] & code_words[other])
                if overlap > 0 or (len(code_words[code]) == 0 and len(code_words[other]) == 0):
                    categories[cluster_name].append(other)
                    used_codes.add(other)
            cluster_id += 1

        # Determine core category
        core = axial_core or (existing_codes[0] if existing_codes else "unknown")

        # Build relationships
        cat_names = list(categories.keys())
        relationships = []
        for i in range(len(cat_names)):
            for j in range(i + 1, len(cat_names)):
                relationships.append({
                    "from": cat_names[i],
                    "to": cat_names[j],
                    "type": "contextual",
                })

        return _json({
            "stage": "axial",
            "categories": {name: codes for name, codes in categories.items()},
            "relationships": relationships,
            "core_category": core,
            "codes_used": existing_codes,
            "method": "word_overlap_clustering",
        })

    def _grounded_selective(self, texts, existing_codes, language):
        """Selective coding: identify core category and code around it."""
        if not existing_codes:
            open_result = json.loads(self._grounded_open(texts, language))
            existing_codes = [c["code"] for c in open_result.get("codes", [])]

        if not existing_codes:
            return _json({"error": "No codes available for selective coding."})

        # ---- LLM path ----
        if self.provider is not None:
            prompt = (
                "You are a qualitative researcher performing selective coding in grounded theory.\n"
                "Identify the core category and relate all other categories to it.\n\n"
                f"Codes: {json.dumps(existing_codes)}\n\n"
                "Respond in JSON format:\n"
                '  "core_category": "..."\n'
                '  "core_category_description": "..."\n'
                '  "subcategories": [{"name": "...", "relation_to_core": "..."}]\n'
                '  "theoretical_memo": "..."\n'
            )
            try:
                response = self.provider.chat([{"role": "user", "content": prompt}])
                content = response.content or ""
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    parsed["stage"] = "selective"
                    parsed["method"] = "llm_coding"
                    return _json(parsed)
            except Exception:
                pass

        # ---- Fallback: most frequent code is core ----
        code_freq: Counter = Counter()
        for t in texts:
            tokens = _remove_stopwords(_tokenize(t))
            for code in existing_codes:
                code_toks = _tokenize(code)
                if all(ct in tokens for ct in code_toks):
                    code_freq[code] += 1

        if not code_freq:
            # Fallback: use order from existing_codes
            for i, code in enumerate(existing_codes):
                code_freq[code] = len(existing_codes) - i

        core = code_freq.most_common(1)[0][0]
        subcategories = []
        for code, freq in code_freq.most_common():
            if code != core:
                subcategories.append({
                    "name": code,
                    "frequency": freq,
                    "relation_to_core": "related",
                })

        return _json({
            "stage": "selective",
            "core_category": core,
            "core_category_description": f"The central category '{core}' integrates all other categories.",
            "subcategories": subcategories[:20],
            "code_frequencies": dict(code_freq.most_common(30)),
            "method": "frequency_based",
        })

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def sentiment(self, args: dict) -> str:
        """Sentiment analysis for interview coding.

        Parameters
        ----------
        args : dict
            texts : list of str

        Returns
        -------
        str
            JSON with per-text sentiment, overall distribution, key words.
        """
        texts: List[str] = args.get("texts", [])
        if not texts:
            return _json({"error": "No texts provided for sentiment analysis."})

        results: List[Dict[str, Any]] = []
        positive_words_all: Counter = Counter()
        negative_words_all: Counter = Counter()

        if HAS_VADER:
            analyzer = SentimentIntensityAnalyzer()
            for i, text in enumerate(texts):
                scores = analyzer.polarity_scores(text)
                compound = scores["compound"]
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"

                # Extract sentiment-bearing words
                tokens = _tokenize(text)
                pos_words = [t for t in tokens if t in _POSITIVE_WORDS]
                neg_words = [t for t in tokens if t in _NEGATIVE_WORDS]
                positive_words_all.update(pos_words)
                negative_words_all.update(neg_words)

                results.append({
                    "text_index": i,
                    "label": label,
                    "compound": round(compound, 4),
                    "positive": round(scores["pos"], 4),
                    "negative": round(scores["neg"], 4),
                    "neutral": round(scores["neu"], 4),
                    "key_positive_words": pos_words[:5],
                    "key_negative_words": neg_words[:5],
                })
        else:
            # Simple lexicon fallback
            for i, text in enumerate(texts):
                tokens = _tokenize(text)
                pos_count = sum(1 for t in tokens if t in _POSITIVE_WORDS)
                neg_count = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
                total = pos_count + neg_count
                if total == 0:
                    label = "neutral"
                    compound = 0.0
                else:
                    compound = (pos_count - neg_count) / total
                    if compound >= 0.05:
                        label = "positive"
                    elif compound <= -0.05:
                        label = "negative"
                    else:
                        label = "neutral"

                pos_words = [t for t in tokens if t in _POSITIVE_WORDS]
                neg_words = [t for t in tokens if t in _NEGATIVE_WORDS]
                positive_words_all.update(pos_words)
                negative_words_all.update(neg_words)

                results.append({
                    "text_index": i,
                    "label": label,
                    "compound": round(compound, 4),
                    "positive_words_count": pos_count,
                    "negative_words_count": neg_count,
                    "key_positive_words": list(set(pos_words))[:5],
                    "key_negative_words": list(set(neg_words))[:5],
                    "method": "simple_lexicon",
                })

        # Overall distribution
        dist: Dict[str, int] = Counter(r["label"] for r in results)

        # Key positive / negative words across all texts
        top_positive = [w for w, _ in positive_words_all.most_common(15)]
        top_negative = [w for w, _ in negative_words_all.most_common(15)]

        # Average compound
        compounds = [r["compound"] for r in results]
        avg_compound = sum(compounds) / len(compounds) if compounds else 0.0

        result = {
            "sentiments": results,
            "overall_distribution": dict(dist),
            "average_compound": round(avg_compound, 4),
            "key_positive_words": top_positive,
            "key_negative_words": top_negative,
            "n_texts": len(texts),
            "method": "vader" if HAS_VADER else "simple_lexicon",
        }
        try:
            pos_pct = dict(dist).get("positive", 0) / len(texts) * 100 if texts else 0
            neg_pct = dict(dist).get("negative", 0) / len(texts) * 100 if texts else 0
            result["apa"] = (
                f"Sentiment analysis of {len(texts)} texts ({'VADER' if HAS_VADER else 'simple lexicon'}) "
                f"showed {pos_pct:.1f}% positive and {neg_pct:.1f}% negative sentiment "
                f"(mean compound score = {avg_compound:.3f})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_sentiment")

    # ------------------------------------------------------------------
    # Inter-coder reliability
    # ------------------------------------------------------------------

    def _krippendorff_alpha(self, coders_matrix: List[List[Any]], level: str = "nominal") -> dict:
        """Compute Krippendorff's α for any number of coders.

        Parameters
        ----------
        coders_matrix : list of list
            Rows = units, columns = coders. Missing values as None.
        level : str
            "nominal", "ordinal", "interval", or "ratio".

        Returns
        -------
        dict with alpha, n_units, n_coders, n_missing, interpretation.
        """
        import numpy as np

        matrix = np.array(coders_matrix, dtype=object)
        n_units, n_coders = matrix.shape

        # Collect all unique values (excluding None/missing)
        all_values = []
        for val in matrix.flat:
            if val is not None and val == val:  # not None and not NaN
                try:
                    all_values.append(float(val))
                except (ValueError, TypeError):
                    all_values.append(str(val))

        if not all_values:
            return {"alpha": None, "n_units": n_units, "n_coders": n_coders,
                    "n_missing": n_units * n_coders, "interpretation": "No valid data"}

        # Determine if numeric or categorical
        try:
            values_arr = np.array([float(v) for v in all_values])
            is_numeric = True
            unique_vals = sorted(set(values_arr))
        except (ValueError, TypeError):
            is_numeric = False
            unique_vals = sorted(set(str(v) for v in all_values))

        # Build coincidence matrix
        n = len(unique_vals)
        coincidence = np.zeros((n, n))

        for i in range(n_units):
            unit_values = []
            for j in range(n_coders):
                val = matrix[i, j]
                if val is not None and val == val:
                    try:
                        unit_values.append(float(val))
                    except (ValueError, TypeError):
                        unit_values.append(str(val))
            m_u = len(unit_values)
            if m_u < 2:
                continue
            for a_idx, a_val in enumerate(unique_vals):
                for b_idx, b_val in enumerate(unique_vals):
                    count_a = sum(1 for v in unit_values if (float(v) if is_numeric else str(v)) == a_val)
                    count_b = sum(1 for v in unit_values if (float(v) if is_numeric else str(v)) == b_val)
                    if a_idx == b_idx:
                        coincidence[a_idx, b_idx] += count_a * (count_b - 1)
                    else:
                        coincidence[a_idx, b_idx] += count_a * count_b

        coincidence = coincidence / 2.0
        total_pairs = np.sum(coincidence)
        if total_pairs == 0:
            return {"alpha": None, "n_units": n_units, "n_coders": n_coders,
                    "n_missing": n_units * n_coders - len(all_values),
                    "interpretation": "No pairable data"}

        # Observed disagreement
        def delta(a, b):
            if level == "nominal":
                return 0.0 if a == b else 1.0
            if level == "ordinal":
                # Rank-based squared difference
                rank_a = unique_vals.index(a)
                rank_b = unique_vals.index(b)
                return (rank_a - rank_b) ** 2
            if level in ("interval", "ratio"):
                return (float(a) - float(b)) ** 2
            return 0.0 if a == b else 1.0

        do = 0.0
        for i in range(n):
            for j in range(n):
                if coincidence[i, j] > 0:
                    do += coincidence[i, j] * delta(unique_vals[i], unique_vals[j])
        do = do / total_pairs

        # Expected disagreement
        row_sums = np.sum(coincidence, axis=1)
        de = 0.0
        for i in range(n):
            for j in range(n):
                pi = row_sums[i] / total_pairs
                pj = row_sums[j] / total_pairs
                de += pi * pj * delta(unique_vals[i], unique_vals[j])

        if de == 0:
            alpha = 1.0 if do == 0 else -np.inf
        else:
            alpha = 1.0 - do / de

        # Interpretation (Krippendorff 2004)
        if alpha < 0:
            interp = "unreliable"
        elif alpha < 0.67:
            interp = "questionable"
        elif alpha < 0.80:
            interp = "tentative"
        else:
            interp = "substantial"

        n_missing = sum(1 for val in matrix.flat if val is None or val != val)
        return {
            "alpha": round(float(alpha), 4) if np.isfinite(alpha) else None,
            "level": level,
            "n_units": n_units,
            "n_coders": n_coders,
            "n_missing": int(n_missing),
            "interpretation": interp,
        }

    def coding_reliability(self, args: dict) -> str:
        """Inter-coder reliability (Cohen's Kappa + Krippendorff's α).

        Parameters
        ----------
        args : dict
            coder1 : list of str/int
            coder2 : list of str/int
            coder3 : list of str/int (optional)
            coders : list of lists (optional, overrides individual coders)
            level : str (default "nominal") — for Krippendorff α

        Returns
        -------
        str
            JSON with ``kappa``, ``krippendorff_alpha``, ``agreement_rate``,
            ``confusion_matrix``, ``interpretation``.
        """
        # Support multiple coders via coders list or individual coderN args
        coders_list = args.get("coders")
        if coders_list and isinstance(coders_list, list):
            coders_matrix = coders_list
        else:
            coder1 = args.get("coder1")
            coder2 = args.get("coder2")
            if coder1 is None or coder2 is None:
                return _json({"error": "Both coder1 and coder2 must be provided, or use 'coders' list."})
            coders_matrix = [coder1, coder2]
            # Add optional coder3+
            for key in sorted(args.keys()):
                if key.startswith("coder") and key != "coder1" and key != "coder2":
                    try:
                        idx = int(key.replace("coder", ""))
                        if idx > 2:
                            coders_matrix.append(args[key])
                    except ValueError:
                        pass

        if not coders_matrix or len(coders_matrix) < 2:
            return _json({"error": "At least 2 coders are required."})

        n_items = len(coders_matrix[0])
        for i, c in enumerate(coders_matrix):
            if len(c) != n_items:
                return _json({"error": f"coder {i+1} length ({len(c)}) does not match first coder ({n_items})."})

        if n_items == 0:
            return _json({"error": "Coding lists must not be empty."})

        # Build transposed matrix: rows = units, columns = coders
        unit_coder_matrix = [[None] * len(coders_matrix) for _ in range(n_items)]
        for j, coder in enumerate(coders_matrix):
            for i, val in enumerate(coder):
                unit_coder_matrix[i][j] = val

        # Cohen's Kappa (only for exactly 2 coders)
        c1 = [str(x) for x in coders_matrix[0]]
        c2 = [str(x) for x in coders_matrix[1]]
        n = len(c1)
        categories = sorted(set(c1) | set(c2))
        cat_idx = {cat: i for i, cat in enumerate(categories)}
        k = len(categories)

        agreements = sum(1 for a, b in zip(c1, c2) if a == b)
        po = agreements / n
        c1_dist = Counter(c1)
        c2_dist = Counter(c2)
        pe = sum((c1_dist.get(cat, 0) / n) * (c2_dist.get(cat, 0) / n) for cat in categories)
        if pe == 1.0:
            kappa = 1.0
        else:
            kappa = (po - pe) / (1.0 - pe)

        # Confusion matrix
        matrix = [[0] * k for _ in range(k)]
        for a, b in zip(c1, c2):
            matrix[cat_idx[a]][cat_idx[b]] += 1

        # Cohen's Kappa interpretation
        if kappa < 0:
            kappa_interp = "poor"
        elif kappa < 0.20:
            kappa_interp = "slight"
        elif kappa < 0.40:
            kappa_interp = "fair"
        elif kappa < 0.60:
            kappa_interp = "moderate"
        elif kappa < 0.80:
            kappa_interp = "substantial"
        else:
            kappa_interp = "almost perfect"

        # Krippendorff's α
        level = str(args.get("level", "nominal")).lower()
        kripp = self._krippendorff_alpha(unit_coder_matrix, level=level)

        result = {
            "kappa": round(kappa, 4),
            "kappa_interpretation": kappa_interp,
            "krippendorff_alpha": kripp.get("alpha"),
            "krippendorff_level": kripp.get("level"),
            "krippendorff_n_coders": kripp.get("n_coders"),
            "krippendorff_n_units": kripp.get("n_units"),
            "krippendorff_interpretation": kripp.get("interpretation"),
            "agreement_rate": round(po, 4),
            "expected_agreement": round(pe, 4),
            "confusion_matrix": matrix,
            "categories": categories,
            "interpretation": kappa_interp,
            "n_items": n,
        }
        try:
            from sophia.research.apa import APAFormatter
            result["apa"] = APAFormatter.krippendorff(
                alpha=kripp.get("alpha"),
                n_coders=kripp.get("n_coders", 2),
                n_units=kripp.get("n_units", n),
                level=kripp.get("level", "nominal")
            )
        except Exception:
            pass
        return self._final(args, result, "research_coding_reliability")

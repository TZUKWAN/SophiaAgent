"""Chinese NLP tool registration for SophiaAgent.

Registers: chinese_tokenize, chinese_keywords, chinese_sentiment, chinese_topics.
"""

import json
import logging
from typing import Any, Dict

from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_chinese_nlp_tools(registry: ToolRegistry, tokenizer=None) -> None:
    """Register Chinese NLP tools.

    Parameters
    ----------
    registry : ToolRegistry
        Central tool registry.
    tokenizer : ChineseTokenizer, optional
        Shared tokenizer instance.
    """
    from sophia.research.chinese_nlp import (
        ChineseTokenizer,
        extract_keywords,
        analyze_sentiment_cn,
        analyze_sentiment_batch,
        extract_topics,
        detect_language,
    )

    _tok = tokenizer or ChineseTokenizer()

    # --- chinese_tokenize ---
    def _tokenize(args: dict) -> str:
        text = args.get("text", "")
        mode = args.get("mode", "default")
        tokens = _tok.tokenize(text, mode=mode)
        clean = _tok.remove_stopwords(tokens)
        return json.dumps({
            "tokens": tokens,
            "tokens_no_stopwords": clean,
            "token_count": len(tokens),
            "clean_count": len(clean),
            "language": detect_language(text),
            "backend": _tok.backend,
        }, ensure_ascii=False, indent=2)

    registry.register(
        "chinese_tokenize",
        "Tokenize Chinese text with optional stopword removal. Supports 'default', 'search', and 'all' modes.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Chinese text to tokenize"},
                "mode": {"type": "string", "enum": ["default", "search", "all"],
                         "description": "Tokenization mode (default: 'default')"},
            },
            "required": ["text"],
        },
        _tokenize,
    )

    # --- chinese_keywords ---
    def _keywords(args: dict) -> str:
        text = args.get("text", "")
        top_n = int(args.get("top_n", 20))
        method = args.get("method", "hybrid")
        results = extract_keywords(text, tokenizer=_tok, top_n=top_n, method=method)
        return json.dumps({
            "keywords": [{"word": w, "score": round(s, 4)} for w, s in results],
            "count": len(results),
            "method": method,
        }, ensure_ascii=False, indent=2)

    registry.register(
        "chinese_keywords",
        "Extract keywords from Chinese text using hybrid TF-IDF + TextRank.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Chinese text"},
                "top_n": {"type": "integer", "description": "Number of keywords (default: 20)"},
                "method": {"type": "string", "enum": ["tfidf", "textrank", "hybrid"],
                           "description": "Extraction method (default: 'hybrid')"},
            },
            "required": ["text"],
        },
        _keywords,
    )

    # --- chinese_sentiment ---
    def _sentiment(args: dict) -> str:
        text = args.get("text", "")
        result = analyze_sentiment_cn(text, tokenizer=_tok)
        return json.dumps(result, ensure_ascii=False, indent=2)

    registry.register(
        "chinese_sentiment",
        "Analyze sentiment of Chinese text. Returns sentiment label, score, confidence, emotion dimensions.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Chinese text to analyze"},
            },
            "required": ["text"],
        },
        _sentiment,
    )

    # --- chinese_topics ---
    def _topics(args: dict) -> str:
        texts = args.get("texts", [])
        n_topics = int(args.get("n_topics", 5))
        if isinstance(texts, str):
            texts = [texts]
        results = extract_topics(texts, tokenizer=_tok, n_topics=n_topics)
        return json.dumps({
            "topics": results,
            "n_topics": len(results),
        }, ensure_ascii=False, indent=2)

    registry.register(
        "chinese_topics",
        "Extract topics from a list of Chinese texts using LDA (if available) or TF-IDF clustering.",
        {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Chinese texts to analyze",
                },
                "n_topics": {"type": "integer", "description": "Number of topics (default: 5)"},
            },
            "required": ["texts"],
        },
        _topics,
    )

    # --- discourse analysis ---
    from sophia.research.discourse import DiscourseEngine

    _discourse_engine = DiscourseEngine()

    def _discourse(args: dict) -> str:
        return _discourse_engine.analyze_discourse(args)

    registry.register(
        "research_discourse_analysis",
        "Analyze discourse in text: identify subjects, power relations, discourse strategies, ideology frames. Supports foucault/cda/narrative frameworks.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "framework": {
                    "type": "string",
                    "enum": ["foucault", "cda", "narrative"],
                    "description": "Analysis framework (default: 'foucault')",
                },
            },
            "required": ["text"],
        },
        _discourse,
    )

    # --- narrative analysis ---
    from sophia.research.narrative import NarrativeEngine

    _narrative_engine = NarrativeEngine()

    def _narrative(args: dict) -> str:
        return _narrative_engine.analyze_narrative(args)

    registry.register(
        "research_narrative_analysis",
        "Analyze narrative structure, turning points, and identity construction in text. Supports structure/turning_point/identity modes.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Narrative text (interview transcript, story, etc.)"},
                "mode": {
                    "type": "string",
                    "enum": ["structure", "turning_point", "identity"],
                    "description": "Analysis mode (default: 'structure')",
                },
            },
            "required": ["text"],
        },
        _narrative,
    )

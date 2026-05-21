"""Tool registration for English academic writing support.

Registers: en_polish, en_readability, en_glossary_build, en_consistency_check,
en_cover_letter, en_review_response, citation_style_convert.
"""

import json
import logging
from typing import Any, Dict

from sophia.research.writing_en import (
    AcademicEnglishEngine,
    en_consistency_check,
    en_cover_letter,
    en_diversify_sentences,
    en_glossary_build,
    en_polish,
    en_readability,
    en_review_response,
)
from sophia.tools.citation import _convert_citation_style

logger = logging.getLogger(__name__)


def register_writing_en_tools(registry, workspace: str = "", provider=None):
    """Register all English academic writing tools."""
    from functools import partial

    registry.register(
        name="en_polish",
        description=(
            "Polish academic English text. Styles: social_science, humanities, "
            "education, public_policy. Performs vocab upgrade, redundancy removal, "
            "Chinglish detection, connector suggestions, and sentence structure advice. "
            "Optional LLM enhancement if provider is available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to polish"},
                "style": {
                    "type": "string",
                    "default": "social_science",
                    "enum": ["social_science", "humanities", "education", "public_policy"],
                },
                "llm": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to attempt LLM-enhanced polish",
                },
            },
            "required": ["text"],
        },
        handler=partial(en_polish, workspace=workspace, provider=provider),
    )

    registry.register(
        name="en_readability",
        description=(
            "Analyze readability of academic English text. Returns metrics: "
            "avg sentence length, sentence length std, sentence structure variety, "
            "passive voice ratio, paragraph avg length, Flesch-Kincaid score, "
            "TTR, AWL coverage, with benchmark references."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "style": {
                    "type": "string",
                    "default": "social_science",
                    "enum": ["social_science", "humanities", "education", "public_policy"],
                },
            },
            "required": ["text"],
        },
        handler=partial(en_readability, workspace=workspace),
    )

    registry.register(
        name="en_diversify_sentences",
        description=(
            "Detect repeated sentence structures and suggest rewrites "
            "to improve sentence variety."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"],
        },
        handler=partial(en_diversify_sentences, workspace=workspace),
    )

    registry.register(
        name="en_glossary_build",
        description=(
            "Auto-extract key terms (noun phrases appearing 3+ times) from text "
            "and persist to workspace/.sophia/glossary.json."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract terms from"},
            },
            "required": ["text"],
        },
        handler=partial(en_glossary_build, workspace=workspace),
    )

    registry.register(
        name="en_consistency_check",
        description=(
            "Check spelling consistency, Chinese-English alignment, "
            "and abbreviation first-use definitions in text. "
            "Optionally checks against persisted glossary."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to check"},
                "glossary": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional glossary list to validate against",
                },
            },
            "required": ["text"],
        },
        handler=partial(en_consistency_check, workspace=workspace),
    )

    registry.register(
        name="en_cover_letter",
        description=(
            "Generate a journal submission cover letter. "
            "paper_meta={title, authors, abstract, keywords, highlights}, "
            "journal={name, scope, editor_name}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_meta": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "string"},
                        "abstract": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "highlights": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "journal": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "scope": {"type": "string"},
                        "editor_name": {"type": "string"},
                    },
                },
            },
            "required": ["paper_meta", "journal"],
        },
        handler=partial(en_cover_letter, workspace=workspace),
    )

    registry.register(
        name="en_review_response",
        description=(
            "Generate a structured response to peer review comments. "
            "review_comments=[{comment_id, comment_text}], "
            "author_revisions=[{comment_id, response, changes}]."
        ),
        parameters={
            "type": "object",
            "properties": {
                "review_comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "comment_id": {"type": "string"},
                            "comment_text": {"type": "string"},
                        },
                    },
                },
                "author_revisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "comment_id": {"type": "string"},
                            "response": {"type": "string"},
                            "changes": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["review_comments"],
        },
        handler=partial(en_review_response, workspace=workspace),
    )

    # citation_style_convert is registered inside citation.py's register_citation_tools,
    # but we also expose it here for completeness if needed.

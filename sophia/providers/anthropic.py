"""Anthropic Claude provider for SophiaAgent.

Supports Claude Opus, Sonnet, and Haiku models via the Anthropic API.
Converts between OpenAI-format tool schemas and Anthropic tool format.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import anthropic
import httpx

from sophia.providers.base import BaseProvider, ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _convert_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        func = tool.get("function", {})
        anthropic_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object"}),
        })
    return anthropic_tools


def _convert_messages(messages: List[Dict[str, Any]]) -> tuple:
    """Convert OpenAI-format messages to Anthropic format.

    Returns:
        (system_prompt, anthropic_messages) tuple.
    """
    system_prompt = ""
    anthropic_msgs = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_prompt = msg.get("content", "")
            continue

        if role == "user":
            anthropic_msgs.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})

            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })

            if content_blocks:
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            anthropic_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })

    return system_prompt, anthropic_msgs


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=15.0),
            max_retries=2,
        )
        self.model = model
        logger.info("AnthropicProvider initialized: %s", model)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Call the Anthropic Messages API."""
        system_prompt, anthropic_msgs = _convert_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_msgs,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _convert_tools_to_anthropic(tools)

        response = self._call_with_retry(kwargs)

        # Parse response
        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        return ProviderResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )

    def _call_with_retry(self, kwargs: Dict[str, Any]):
        """Call the Anthropic API with exponential backoff retry."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.messages.create(**kwargs)
            except (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            ) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt + 0.5
                    logger.warning(
                        "Anthropic API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, type(e).__name__, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Anthropic API error after %d retries: %s", MAX_RETRIES, e,
                    )
            except Exception:
                raise
        raise last_error

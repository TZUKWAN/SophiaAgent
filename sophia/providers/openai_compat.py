"""OpenAI-compatible provider.

Works with: OpenAI, DeepSeek, Ollama, vLLM, and any OpenAI API compatible endpoint.
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

import httpx
from openai import OpenAI, APIConnectionError, RateLimitError, InternalServerError

from sophia.providers.base import BaseProvider, ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=15.0),
            max_retries=2,
        )
        self.model = model
        logger.info("OpenAICompatProvider initialized: %s @ %s", model, base_url)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Call the OpenAI-compatible chat completion API with retry."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        logger.debug(
            "Sending request: %d messages, %d tools",
            len(messages),
            len(tools) if tools else 0,
        )

        response = self._call_with_retry(kwargs)
        if not response.choices:
            return ProviderResponse(content="", tool_calls=[], raw=response)
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return ProviderResponse(
            content=msg.content,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )

    def _call_with_retry(self, kwargs: Dict[str, Any]):
        """Call the API with exponential backoff retry for transient errors."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt + 0.5
                    logger.warning(
                        "API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, type(e).__name__, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "API error after %d retries: %s", MAX_RETRIES, e,
                    )
            except Exception as e:
                raise
        raise last_error

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, ProviderResponse]:
        """Stream chat tokens from the API.

        Yields text chunks as they arrive. Returns final ProviderResponse.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        stream = self.client.chat.completions.create(**kwargs)

        content_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            # Text content
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content

            # Tool calls (accumulate across chunks)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if tc_delta.index is not None else 0
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

        # Build final response
        tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            tc_data = tool_calls_acc[idx]
            try:
                args = json.loads(tc_data["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = tc_data["arguments"]
            tool_calls.append(ToolCall(
                id=tc_data["id"],
                name=tc_data["name"],
                arguments=args,
            ))

        return ProviderResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            usage=None,
        )

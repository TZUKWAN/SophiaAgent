"""Base provider interface for SophiaAgent."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class ToolCall:
    """Represents a single tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ProviderResponse:
    """Unified response from any LLM provider."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible message dict for conversation history."""
        msg: Dict[str, Any] = {"role": "assistant"}

        if self.content:
            msg["content"] = self.content
        else:
            msg["content"] = None

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                        if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]

        return msg


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Send messages and get a response."""
        ...

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, ProviderResponse]:
        """Stream chat tokens. Yields text chunks, returns final ProviderResponse."""
        resp = self.chat(messages, tools)
        if resp.content:
            yield resp.content
        return resp

"""LLM Provider abstraction layer."""

from sophia.providers.base import BaseProvider, ProviderResponse, ToolCall
from sophia.providers.openai_compat import OpenAICompatProvider


def create_provider(config) -> BaseProvider:
    """Factory: create a provider based on config."""
    provider_type = config.model.provider

    if provider_type == "openai-compat":
        return OpenAICompatProvider(
            base_url=config.model.base_url,
            api_key=config.model.api_key,
            model=config.model.name,
        )
    elif provider_type == "anthropic":
        from sophia.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=config.model.api_key,
            model=config.model.name,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}. Use 'openai-compat' or 'anthropic'.")


__all__ = [
    "BaseProvider",
    "OpenAICompatProvider",
    "ProviderResponse",
    "ToolCall",
    "create_provider",
]

from sophia.config import Config
from sophia.providers import create_provider
from sophia.providers.unconfigured import UnconfiguredProvider


def test_create_provider_returns_unconfigured_provider_without_endpoint():
    config = Config()
    config.model.provider = "openai-compat"
    config.model.base_url = ""
    config.model.api_key = ""

    provider = create_provider(config)
    response = provider.chat([{"role": "user", "content": "hello"}])

    assert isinstance(provider, UnconfiguredProvider)
    assert "还没有检测到可用的大模型配置" in response.content


def test_anthropic_without_api_key_uses_unconfigured_provider():
    config = Config()
    config.model.provider = "anthropic"
    config.model.api_key = ""

    assert isinstance(create_provider(config), UnconfiguredProvider)

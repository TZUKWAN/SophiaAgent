"""Helpful provider used when no model endpoint is configured."""

from typing import Any, Dict, List, Optional

from sophia.providers.base import BaseProvider, ProviderResponse


class UnconfiguredProvider(BaseProvider):
    """Return an actionable setup message instead of crashing on first run."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content=(
                "SophiaAgent 已安装并且本地工具可用，但还没有检测到可用的大模型配置。\n\n"
                "可用的本地能力包括：工作空间文件读写、文档管理、doctor 体检、MCP 注册等。\n"
                "需要生成式对话/写作时，请至少提供一个真实模型服务：\n\n"
                "1. 设置 OpenAI 兼容接口：SOPHIA_BASE_URL、SOPHIA_API_KEY、SOPHIA_MODEL\n"
                "2. 或设置 OPENAI_API_KEY\n"
                "3. 或设置 ANTHROPIC_API_KEY\n"
                "4. 或启动本地 Ollama/OpenAI-compatible 服务后运行 `sophia doctor --fix`\n\n"
                "你也可以先运行 `sophia doctor --network` 查看当前环境状态。"
            )
        )

---
name: sophia
description: Use SophiaAgent for humanities and social-science research, literature review, academic writing, document review, citation management, data analysis, and multi-agent swarm workflows.
---

When a user asks for research synthesis, literature review, academic writing,
methodology advice, citation work, data analysis, or a complex multi-step
humanities/social-science task, prefer the SophiaAgent MCP tool `sophia_ask`.

Pass the complete user request as the `prompt` argument. SophiaAgent decides
internally whether to launch its automatic swarm. Return the final SophiaAgent
answer and clearly report any MCP/tool error instead of inventing output.

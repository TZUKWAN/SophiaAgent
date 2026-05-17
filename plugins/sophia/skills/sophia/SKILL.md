---
name: sophia
description: Delegate humanities/social-science research, writing, review, citation, and data-analysis tasks to SophiaAgent through MCP.
---

Use SophiaAgent when the user asks for literature research, academic writing,
document review, citation management, research methodology, data analysis, or
multi-step scholarly workflows. Call the MCP tool `sophia_ask` with the full
user request as `prompt`. SophiaAgent will automatically decide whether to use
its internal swarm and return one final answer.

For explicit `/sophia ...` style requests, treat the text after `/sophia` as the
prompt for `sophia_ask`. Never fabricate SophiaAgent results if the MCP server
is unavailable; report the integration error and suggest `sophia integrate --auto`.

import json
import logging
from typing import Any, Dict
import httpx

from sophia.plugins import PluginInterface
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

class NotionPlugin(PluginInterface):
    def name(self) -> str:
        return "notion"

    def register(self, registry: ToolRegistry, **kwargs) -> None:
        registry.register(
            name="notion_append_text",
            description="Append text content to a Notion page using the Notion API.",
            parameters={
                "type": "object",
                "properties": {
                    "notion_token": {
                        "type": "string",
                        "description": "Notion Integration Token",
                    },
                    "page_id": {
                        "type": "string",
                        "description": "Notion Page ID to append to",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to append (will be converted to a paragraph block)",
                    }
                },
                "required": ["notion_token", "page_id", "content"]
            },
            handler=self._append_handler
        )

    def _append_handler(self, args: Dict[str, Any]) -> str:
        token = args.get("notion_token")
        page_id = args.get("page_id")
        content = args.get("content")

        if not token or not page_id or not content:
            return json.dumps({
                "success": False,
                "error": "Missing required fields."
            })

        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        # Simple paragraph block
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": content
                                }
                            }
                        ]
                    }
                }
            ]
        }

        try:
            with httpx.Client() as client:
                response = client.patch(url, headers=headers, json=payload, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
            return json.dumps({
                "success": True,
                "message": "Content successfully appended to Notion page."
            }, ensure_ascii=False)
            
        except httpx.HTTPStatusError as e:
            return json.dumps({
                "success": False,
                "error": f"API error: {e.response.status_code} - {e.response.text}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

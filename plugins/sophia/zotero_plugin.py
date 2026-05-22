import json
import logging
from typing import Any, Dict
import httpx

from sophia.plugins import PluginInterface
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

class ZoteroPlugin(PluginInterface):
    def name(self) -> str:
        return "zotero"

    def register(self, registry: ToolRegistry, **kwargs) -> None:
        registry.register(
            name="zotero_fetch_recent",
            description="Fetch recent items from a Zotero user's library using Zotero API.",
            parameters={
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "Zotero User ID (UID)",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Zotero API Key",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent items to fetch (default: 10)",
                        "default": 10
                    }
                },
                "required": ["uid", "api_key"]
            },
            handler=self._fetch_recent_handler
        )

    def _fetch_recent_handler(self, args: Dict[str, Any]) -> str:
        uid = args.get("uid")
        api_key = args.get("api_key")
        limit = args.get("limit", 10)

        url = f"https://api.zotero.org/users/{uid}/items?limit={limit}&sort=dateAdded&direction=desc"
        headers = {
            "Zotero-API-Key": api_key,
        }

        try:
            with httpx.Client() as client:
                response = client.get(url, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
            items = []
            for item in data:
                data_obj = item.get("data", {})
                title = data_obj.get("title", "")
                creators = data_obj.get("creators", [])
                date = data_obj.get("date", "")
                url_field = data_obj.get("url", "")
                
                authors = []
                for creator in creators:
                    if "firstName" in creator and "lastName" in creator:
                        authors.append(f"{creator['firstName']} {creator['lastName']}")
                    elif "name" in creator:
                        authors.append(creator["name"])
                
                items.append({
                    "key": item.get("key"),
                    "title": title,
                    "authors": authors,
                    "date": date,
                    "url": url_field
                })
                
            return json.dumps({
                "success": True,
                "items_fetched": len(items),
                "items": items
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

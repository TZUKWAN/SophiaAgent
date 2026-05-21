"""ZettelkastenStore: Atomic note cards with bidirectional linking.

Persistence: workspace/.sophia/notes/ — one JSON file per card.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

NOTE_TYPES = ["concept", "evidence", "comment"]


class ZettelkastenStore:
    """Atomic note card store with bidirectional links and search."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.notes_dir = os.path.join(workspace, ".sophia", "notes")
        os.makedirs(self.notes_dir, exist_ok=True)

    # -- persistence helpers ------------------------------------------------

    def _note_path(self, note_id: str) -> str:
        return os.path.join(self.notes_dir, f"{note_id}.json")

    def _load_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        path = self._note_path(note_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load note %s: %s", note_id, e)
            return None

    def _save_note(self, note: Dict[str, Any]) -> None:
        note["updated_at"] = datetime.now().isoformat()
        path = self._note_path(note["id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(note, f, ensure_ascii=False, indent=2)

    def _all_notes(self) -> List[Dict[str, Any]]:
        notes = []
        if not os.path.exists(self.notes_dir):
            return notes
        for fname in os.listdir(self.notes_dir):
            if not fname.endswith(".json"):
                continue
            note_id = fname[:-5]
            note = self._load_note(note_id)
            if note:
                notes.append(note)
        return notes

    # -- link parsing -------------------------------------------------------

    @staticmethod
    def _extract_links(content: str) -> List[str]:
        """Extract [[note_id]] links from content."""
        return re.findall(r"\[\[([a-zA-Z0-9_-]+)\]\]", content)

    # -- backlink table -----------------------------------------------------

    def _update_backlinks(self, note_id: str, old_links: List[str], new_links: List[str]) -> None:
        """Maintain backlink table in target notes."""
        removed = set(old_links) - set(new_links)
        added = set(new_links) - set(old_links)

        for target_id in removed:
            target = self._load_note(target_id)
            if target:
                backlinks = target.get("backlinks", [])
                if note_id in backlinks:
                    backlinks.remove(note_id)
                    target["backlinks"] = backlinks
                    self._save_note(target)

        for target_id in added:
            target = self._load_note(target_id)
            if target:
                backlinks = target.get("backlinks", [])
                if note_id not in backlinks:
                    backlinks.append(note_id)
                    target["backlinks"] = backlinks
                    self._save_note(target)

    # -- CRUD ---------------------------------------------------------------

    def create(
        self,
        title: str,
        content: str,
        note_type: str = "concept",
        tags: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
        source_type: str = "",
        source_id: str = "",
    ) -> Dict[str, Any]:
        """Create a new note card."""
        if note_type not in NOTE_TYPES:
            return {"error": f"Invalid note_type. Must be one of: {NOTE_TYPES}"}

        note_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()

        # Validate and resolve links
        resolved_links = []
        for link_id in (links or []):
            if self._load_note(link_id) is not None:
                resolved_links.append(link_id)
            else:
                logger.warning("Link target note not found: %s", link_id)

        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "tags": list(tags or []),
            "links": resolved_links,
            "backlinks": [],
            "source_type": source_type,
            "source_id": source_id,
            "note_type": note_type,
            "created_at": now,
            "updated_at": now,
        }

        self._save_note(note)

        # Update backlinks for linked notes
        self._update_backlinks(note_id, [], resolved_links)

        return {"success": True, "note": note}

    def get(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get a note by ID."""
        return self._load_note(note_id)

    def update(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update a note card."""
        note = self._load_note(note_id)
        if not note:
            return {"error": f"Note '{note_id}' not found"}

        old_links = note.get("links", [])

        if title is not None:
            note["title"] = title
        if content is not None:
            note["content"] = content
        if tags is not None:
            note["tags"] = list(tags)
        if links is not None:
            # Validate links
            resolved_links = []
            for link_id in links:
                if self._load_note(link_id) is not None:
                    resolved_links.append(link_id)
                else:
                    logger.warning("Link target note not found: %s", link_id)
            note["links"] = resolved_links

        # Also extract inline links from content
        inline_links = self._extract_links(note.get("content", ""))
        all_links = list(set(note.get("links", []) + inline_links))
        note["links"] = all_links

        self._save_note(note)
        self._update_backlinks(note_id, old_links, all_links)

        return {"success": True, "note": note}

    def delete(self, note_id: str) -> Dict[str, Any]:
        """Delete a note card."""
        note = self._load_note(note_id)
        if not note:
            return {"error": f"Note '{note_id}' not found"}

        # Remove backlinks from linked notes
        self._update_backlinks(note_id, note.get("links", []), [])

        path = self._note_path(note_id)
        try:
            os.remove(path)
        except Exception as e:
            return {"error": f"Failed to delete note: {e}"}

        return {"success": True, "deleted_id": note_id}

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        linked_to: Optional[str] = None,
        note_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search notes by query, tags, linked_to, or note_type."""
        notes = self._all_notes()
        results = []
        query_lower = query.lower() if query else ""

        for note in notes:
            # Filter by note_type
            if note_type and note.get("note_type") != note_type:
                continue

            # Filter by tags
            if tags:
                note_tags = set(note.get("tags", []))
                if not note_tags.intersection(set(tags)):
                    continue

            # Filter by linked_to
            if linked_to:
                if linked_to not in note.get("links", []) and linked_to not in note.get("backlinks", []):
                    continue

            # Text search
            if query_lower:
                text = f"{note.get('title', '')} {note.get('content', '')}".lower()
                if query_lower not in text:
                    continue

            results.append(note)

        # Sort by relevance: title match first, then content match
        def _score(note: Dict[str, Any]) -> int:
            score = 0
            if query_lower and query_lower in note.get("title", "").lower():
                score += 10
            if query_lower and query_lower in note.get("content", "").lower():
                score += 5
            return score

        results.sort(key=_score, reverse=True)
        return results

    # -- link graph ---------------------------------------------------------

    def get_link_graph(self) -> Dict[str, Any]:
        """Get the note link graph as {nodes, edges}."""
        notes = self._all_notes()
        nodes = []
        edges = []
        node_ids = set()

        for note in notes:
            nid = note["id"]
            node_ids.add(nid)
            nodes.append({
                "id": nid,
                "title": note.get("title", ""),
                "type": note.get("note_type", "concept"),
                "tags": note.get("tags", []),
            })

        for note in notes:
            nid = note["id"]
            for link_id in note.get("links", []):
                if link_id in node_ids:
                    edges.append({
                        "source": nid,
                        "target": link_id,
                        "type": "forward",
                    })
            for back_id in note.get("backlinks", []):
                if back_id in node_ids:
                    edges.append({
                        "source": back_id,
                        "target": nid,
                        "type": "back",
                    })

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {
            "nodes": nodes,
            "edges": unique_edges,
            "node_count": len(nodes),
            "edge_count": len(unique_edges),
        }

    # -- auto-generate from paper extraction --------------------------------

    def from_paper_elements(
        self,
        elements: Dict[str, Any],
        paper_title: str = "",
        paper_id: str = "",
    ) -> Dict[str, Any]:
        """Auto-generate evidence note from PaperReader.extract_key_elements result."""
        note_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()

        # Build content from elements
        content_parts = []
        for key in [
            "research_question",
            "core_arguments",
            "methods",
            "data_sources",
            "main_findings",
            "limitations",
            "theoretical_framework",
        ]:
            val = elements.get(key)
            if val:
                if isinstance(val, list):
                    val_str = "; ".join(str(v) for v in val[:5])
                else:
                    val_str = str(val)
                content_parts.append(f"**{key}**: {val_str}")

        if elements.get("sample_size"):
            content_parts.append(f"**sample_size**: {elements['sample_size']}")

        content = "\n\n".join(content_parts)

        tags = ["evidence", "auto-generated"]
        if paper_title:
            tags.append(paper_title[:30])

        note = {
            "id": note_id,
            "title": paper_title or f"Evidence note from paper {paper_id}",
            "content": content,
            "tags": tags,
            "links": [],
            "backlinks": [],
            "source_type": "paper",
            "source_id": paper_id,
            "note_type": "evidence",
            "created_at": now,
            "updated_at": now,
        }

        self._save_note(note)

        return {"success": True, "note": note}

    # -- bulk operations ----------------------------------------------------

    def list_all(self) -> List[Dict[str, Any]]:
        """List all notes."""
        return self._all_notes()

    def list_by_type(self, note_type: str) -> List[Dict[str, Any]]:
        """List notes by type."""
        return [n for n in self._all_notes() if n.get("note_type") == note_type]

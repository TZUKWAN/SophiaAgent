"""Web server for SophiaAgent.

FastAPI + WebSocket backend with ChatGPT-style web UI.
"""

import asyncio
import logging
import os
import signal
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

from sophia.agent import SophiaAgent
from sophia.config import Config
from sophia.session import SessionManager

logger = logging.getLogger(__name__)

WEB_DIR = os.path.dirname(__file__)

# Active WebSocket connection counter for auto-shutdown
_active_connections = 0
_shutdown_task = None


def _schedule_shutdown(delay: float = 5.0):
    """Schedule process exit after delay if no connections remain."""
    global _shutdown_task

    def _cancel_if_exists():
        global _shutdown_task
        if _shutdown_task is not None:
            _shutdown_task.cancel()
            _shutdown_task = None

    _cancel_if_exists()

    async def _delayed_exit():
        await asyncio.sleep(delay)
        if _active_connections == 0:
            logger.info("No active connections, shutting down server.")
            # Send SIGINT to self for graceful uvicorn shutdown
            os.kill(os.getpid(), signal.SIGINT)

    _shutdown_task = asyncio.create_task(_delayed_exit())


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-3:]


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = Config.load()

    app = FastAPI(title="SophiaAgent", version="0.1.0")

    # Serve static files
    static_dir = os.path.join(WEB_DIR, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Agent instance (shared across requests for session continuity)
    agent = SophiaAgent(config)
    workspace = config.session.workspace

    # Persistent session storage via SQLite
    session_mgr = SessionManager(config.session.db_path)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main SPA page."""
        template_path = os.path.join(WEB_DIR, "templates", "index.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        return HTMLResponse("<h1>SophiaAgent</h1><p>Template not found</p>")

    # ── Settings API ─────────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings():
        return {
            "provider": config.model.provider,
            "model": config.model.name,
            "base_url": config.model.base_url,
            "api_key_masked": _mask_key(config.model.api_key),
            "workspace": config.session.workspace,
        }

    @app.put("/api/settings")
    async def update_settings(request: Request):
        body = await request.json()
        changed = False

        if "provider" in body:
            config.model.provider = body["provider"]
            changed = True
        if "model" in body:
            config.model.name = body["model"]
            changed = True
        if "base_url" in body:
            config.model.base_url = body["base_url"]
            changed = True
        if "api_key" in body and body["api_key"]:
            config.model.api_key = body["api_key"]
            changed = True
        if "workspace" in body:
            new_ws = body["workspace"]
            try:
                Path(new_ws).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            config.session.workspace = new_ws
            changed = True

        if changed:
            agent.reconfigure(config)

        return {"status": "ok", "settings": await get_settings()}

    # ── Usage API ────────────────────────────────────────────

    @app.get("/api/usage")
    async def get_usage():
        return {"usage": dict(agent._session_tokens)}

    # ── Workspace API ────────────────────────────────────────

    @app.get("/api/workspaces")
    async def list_workspaces():
        known = set()
        known.update(session_mgr.list_registered_workspaces())
        known.add(config.session.workspace)
        for w in session_mgr.list_workspaces():
            known.add(w)
        home_ws = str(Path.home() / "SophiaWorkspace")
        known.add(home_ws)
        return {"workspaces": sorted(known), "current": config.session.workspace}

    @app.post("/api/workspaces")
    async def create_workspace(request: Request):
        body = await request.json()
        path = body.get("path", "")
        if not path:
            raise HTTPException(status_code=400, detail="path is required")
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        session_mgr.register_workspace(path)
        return {"workspace": path, "created": True}

    @app.post("/api/workspace/switch")
    async def switch_workspace(request: Request):
        body = await request.json()
        new_ws = body.get("workspace", "")
        if not new_ws:
            raise HTTPException(status_code=400, detail="workspace is required")
        try:
            Path(new_ws).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        config.session.workspace = new_ws
        session_mgr.register_workspace(new_ws)
        agent.reconfigure(config)
        return {"workspace": new_ws, "switched": True}

    # ── Chat API ─────────────────────────────────────────────

    @app.post("/api/chat")
    async def chat(request: Request):
        """Send a message and get a response (synchronous)."""
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id", "")

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        if not session_id:
            session_id = session_mgr.create_session(
                title=message[:50], model=config.model.name,
                workspace=config.session.workspace,
            )
        else:
            existing = session_mgr.get_session(session_id)
            if not existing:
                session_mgr.create_session(
                    session_id=session_id,
                    title=message[:50],
                    model=config.model.name,
                    workspace=config.session.workspace,
                )

        history = session_mgr.get_messages(session_id)

        try:
            result = agent.chat(message, history=history)
            session_mgr.add_messages_batch(session_id, [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["response"]},
            ])

            return {
                "session_id": session_id,
                "response": result["response"],
                "usage": dict(agent._session_tokens),
            }
        except Exception as e:
            logger.exception("Chat error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/api/chat/stream")
    async def chat_stream(websocket: WebSocket):
        """WebSocket endpoint for streaming chat."""
        global _active_connections
        await websocket.accept()
        _active_connections += 1
        # Cancel pending shutdown when a new connection arrives
        if _shutdown_task is not None:
            _shutdown_task.cancel()

        session_id = None

        try:
            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")
                if not message:
                    continue

                if not session_id:
                    session_id = session_mgr.create_session(
                        title=message[:50], model=config.model.name,
                        workspace=config.session.workspace,
                    )

                history = session_mgr.get_messages(session_id)

                try:
                    final_response = ""

                    for event in agent.run_stream(message, history=history):
                        if event["type"] == "token":
                            await websocket.send_json({
                                "type": "token",
                                "session_id": session_id,
                                "content": event["content"],
                            })
                        elif event["type"] == "tool_call":
                            await websocket.send_json({
                                "type": "tool_call",
                                "session_id": session_id,
                                "name": event["name"],
                                "arguments": event["arguments"],
                            })
                        elif event["type"] == "tool_result":
                            await websocket.send_json({
                                "type": "tool_result",
                                "session_id": session_id,
                                "name": event["name"],
                                "result": event["result"],
                            })
                        elif event["type"].startswith("workspace_"):
                            payload = dict(event)
                            payload["session_id"] = session_id
                            payload.pop("context", None)
                            await websocket.send_json(payload)
                        elif event["type"].startswith("swarm_"):
                            payload = dict(event)
                            payload["session_id"] = session_id
                            await websocket.send_json(payload)
                        elif event["type"] == "done":
                            final_response = event["response"]

                    session_mgr.add_messages_batch(session_id, [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final_response},
                    ])

                    await websocket.send_json({
                        "type": "done",
                        "session_id": session_id,
                        "content": final_response,
                        "usage": dict(agent._session_tokens),
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", session_id)
        finally:
            _active_connections -= 1
            if _active_connections <= 0:
                _schedule_shutdown()

    # ── Session API ──────────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions():
        return {"sessions": session_mgr.list_sessions()}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        session_mgr.delete_session(session_id)
        return {"action": "deleted"}

    @app.post("/api/sessions/{session_id}/checkpoints")
    async def create_checkpoint(session_id: str, request: Request):
        body = await request.json() if request.headers.get("content-type") else {}
        label = body.get("label", "auto") if isinstance(body, dict) else "auto"
        cp_id = session_mgr.save_checkpoint(session_id, label)
        return {"checkpoint_id": cp_id, "label": label}

    @app.get("/api/sessions/{session_id}/checkpoints")
    async def list_checkpoints(session_id: str):
        return {"checkpoints": session_mgr.list_checkpoints(session_id)}

    @app.post("/api/sessions/{session_id}/checkpoints/{checkpoint_id}/restore")
    async def restore_checkpoint(session_id: str, checkpoint_id: int):
        ok = session_mgr.restore_checkpoint(session_id, checkpoint_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return {"action": "restored"}

    # ── Tools & Config API ───────────────────────────────────

    @app.get("/api/tools")
    async def list_tools():
        return {"tools": agent.tools.list_tools()}

    @app.get("/api/config")
    async def get_config():
        return {
            "model": config.model.name,
            "provider": config.model.provider,
            "workspace": config.session.workspace,
        }

    # ── Citation Network ─────────────────────────────────────

    @app.get("/api/citation-network")
    async def citation_network():
        from sophia.tools.citation import _load_bib, _load_relations
        entries = _load_bib(workspace)
        relations = _load_relations(workspace)
        type_colors = {
            "builds-on": "#22c55e", "contradicts": "#ef4444",
            "parallel": "#3b82f6", "supersedes": "#f59e0b",
            "applies": "#8b5cf6", "critiques": "#ec4899",
        }
        nodes = [{"id": e.get("_key", ""), "label": e.get("title", "")[:60],
                   "author": e.get("author", ""), "year": e.get("year", "")} for e in entries]
        edges = [{"from": r.get("from", ""), "to": r.get("to", ""),
                   "type": r.get("type", ""), "color": type_colors.get(r.get("type", ""), "#999")}
                  for r in relations]
        return {"nodes": nodes, "edges": edges}

    # ── File Upload ──────────────────────────────────────────

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        upload_dir = os.path.join(config.session.workspace, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        filename = file.filename or "upload"
        filename = os.path.basename(filename)
        dest = os.path.join(upload_dir, filename)

        if os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(upload_dir, f"{name}_{uuid.uuid4().hex[:4]}{ext}")

        try:
            content = await file.read()
            with open(dest, "wb") as f:
                f.write(content)
            return {
                "action": "uploaded",
                "filename": os.path.basename(dest),
                "path": dest,
                "size": len(content),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

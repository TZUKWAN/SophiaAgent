"""Thread-safe communication bus for one swarm execution."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BusMessage:
    msg_id: str
    sender_id: str
    channel_id: str
    msg_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class SwarmBus:
    """Shared memory and message channels for collaborating sub-agents."""

    def __init__(self, max_messages_per_channel: int = 100):
        self._channels: Dict[str, List[BusMessage]] = {}
        self._lock = threading.RLock()
        self._max_messages = max_messages_per_channel
        self.bus_id = uuid.uuid4().hex[:8]

    def register_agent(self, agent_id: str) -> None:
        with self._lock:
            self._channels.setdefault(agent_id, [])

    def unregister_agent(self, agent_id: str) -> None:
        with self._lock:
            self._channels.pop(agent_id, None)

    def list_agents(self) -> List[str]:
        with self._lock:
            return sorted(self._channels)

    def write(
        self,
        sender_id: str,
        content: str,
        msg_type: str = "result",
        metadata: Optional[Dict[str, Any]] = None,
        channel_id: Optional[str] = None,
    ) -> BusMessage:
        channel = channel_id or sender_id
        msg = BusMessage(
            msg_id=uuid.uuid4().hex[:8],
            sender_id=sender_id,
            channel_id=channel,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {},
        )
        with self._lock:
            self._channels.setdefault(channel, [])
            self._channels[channel].append(msg)
            if len(self._channels[channel]) > self._max_messages:
                self._channels[channel] = self._channels[channel][-self._max_messages :]
        return msg

    def broadcast(
        self,
        sender_id: str,
        content: str,
        msg_type: str = "broadcast",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[BusMessage]:
        with self._lock:
            channels = list(self._channels)
        return [
            self.write(
                sender_id=sender_id,
                channel_id=channel,
                content=content,
                msg_type=msg_type,
                metadata={**(metadata or {}), "broadcast": True},
            )
            for channel in channels
        ]

    def read(self, agent_id: str, since_index: int = 0) -> List[BusMessage]:
        with self._lock:
            return list(self._channels.get(agent_id, [])[since_index:])

    def read_all(self) -> Dict[str, List[BusMessage]]:
        with self._lock:
            return {agent_id: list(messages) for agent_id, messages in self._channels.items()}

    def read_latest(self, agent_id: str, n: int = 1) -> List[BusMessage]:
        with self._lock:
            messages = self._channels.get(agent_id, [])
            return list(messages[-n:])

    def read_by_type(self, agent_id: str, msg_type: str) -> List[BusMessage]:
        with self._lock:
            return [msg for msg in self._channels.get(agent_id, []) if msg.msg_type == msg_type]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "bus_id": self.bus_id,
                "agent_count": len(self._channels),
                "agents": {
                    agent_id: {
                        "message_count": len(messages),
                        "latest": messages[-1].content[:200] if messages else None,
                    }
                    for agent_id, messages in self._channels.items()
                },
            }

    def to_context_string(self, max_length: int = 8000) -> str:
        lines: List[str] = []
        with self._lock:
            items = list(self._channels.items())
        for agent_id, messages in items:
            if not messages:
                continue
            lines.append(f"[Agent: {agent_id}]")
            for msg in messages:
                lines.append(f"[{msg.msg_type} from {msg.sender_id}] {msg.content[:1000]}")
            lines.append("")
        text = "\n".join(lines)
        if len(text) > max_length:
            return text[:max_length] + "\n...[truncated]"
        return text

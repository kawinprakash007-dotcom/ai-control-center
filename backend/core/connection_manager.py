import asyncio
import logging
from typing import Set, Dict, Any
from fastapi import WebSocket, status

logger = logging.getLogger("atlas.connection_manager")


class BoundedConnectionManager:
    """
    Thread-safe, bounded WebSocket connection manager.
    Enforces maximum concurrent connections, safe disconnect cleanup,
    and non-blocking message distribution.
    """

    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept WebSocket connection if within capacity limit."""
        if len(self.active_connections) >= self.max_connections:
            logger.warning(
                "WebSocket connection rejected: maximum capacity of %d reached.",
                self.max_connections,
            )
            if hasattr(websocket, "close"):
                close_res = websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server busy: max connections")
                if asyncio.iscoroutine(close_res):
                    await close_res
            return False

        if hasattr(websocket, "accept"):
            res = websocket.accept()
            if asyncio.iscoroutine(res):
                await res
        self.active_connections.add(websocket)
        logger.info("WebSocket connected. Active connections: %d", len(self.active_connections))
        return True

    def disconnect(self, websocket: WebSocket) -> None:
        """Safely remove a disconnected client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket disconnected. Active connections: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast JSON message to all active clients, pruning stale connections."""
        disconnected = set()
        for connection in list(self.active_connections):
            try:
                res = connection.send_json(message)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug("Failed to send to WebSocket: %s. Pruning.", e)
                disconnected.add(connection)

        for stale in disconnected:
            self.disconnect(stale)

    async def close_all(self, code: int = 1000, reason: str = "Server shutting down") -> None:
        """Gracefully close all active client connections."""
        connections = list(self.active_connections)
        self.active_connections.clear()
        for connection in connections:
            try:
                await connection.close(code=code, reason=reason)
            except Exception as e:
                logger.debug("Error closing connection: %s", e)

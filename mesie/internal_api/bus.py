"""Internal message bus — publish/subscribe and request/response between engines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from mesie.internal_api.messages import EngineResponse, MessageEnvelope, MessageTopic

Handler = Callable[[MessageEnvelope], Optional[EngineResponse]]


class InternalBus:
    """In-process bus for MESIE engines and octopus arms."""

    def __init__(self) -> None:
        self._handlers: Dict[MessageTopic, List[Handler]] = defaultdict(list)
        self._named_handlers: Dict[str, Handler] = {}
        self._history: List[MessageEnvelope] = []
        self._max_history = 500

    def subscribe(self, topic: MessageTopic, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    def register_engine(self, name: str, handler: Handler) -> None:
        self._named_handlers[name] = handler

    def publish(self, message: MessageEnvelope) -> List[EngineResponse]:
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        responses: List[EngineResponse] = []

        if message.target in self._named_handlers:
            out = self._named_handlers[message.target](message)
            if out is not None:
                responses.append(out)

        for handler in self._handlers.get(message.topic, []):
            out = handler(message)
            if out is not None:
                responses.append(out)

        if message.topic == MessageTopic.BROADCAST:
            for handler in self._handlers.get(MessageTopic.ENGINE_REQUEST, []):
                out = handler(message)
                if out is not None:
                    responses.append(out)

        return responses

    def request(
        self,
        source: str,
        target: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        topic: MessageTopic = MessageTopic.ENGINE_REQUEST,
    ) -> EngineResponse:
        msg = MessageEnvelope(
            topic=topic,
            source=source,
            target=target,
            action=action,
            payload=payload or {},
        )
        results = self.publish(msg)
        if not results:
            return EngineResponse(
                ok=False,
                engine=target,
                action=action,
                error=f"No handler for target '{target}'",
            )
        return results[0]

    @property
    def history(self) -> List[MessageEnvelope]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
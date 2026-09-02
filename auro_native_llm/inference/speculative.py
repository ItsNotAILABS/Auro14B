from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class SpeculativeReceipt:
    proposed: list[int]
    accepted: list[int]
    rejected_at: int | None
    verifier_calls: int


def verify_draft_tokens(
    prefix: Sequence[int],
    proposed: Sequence[int],
    verifier_next_token: Callable[[list[int]], int],
) -> SpeculativeReceipt:
    context = list(prefix)
    accepted: list[int] = []
    calls = 0
    rejected_at: int | None = None
    for index, token in enumerate(proposed):
        expected = int(verifier_next_token(context))
        calls += 1
        if expected != int(token):
            rejected_at = index
            accepted.append(expected)
            break
        accepted.append(int(token))
        context.append(int(token))
    return SpeculativeReceipt(list(map(int, proposed)), accepted, rejected_at, calls)


class AuroSpeculativeCoordinator:
    """Coordinate a small AURO draft model with a larger verifier.

    This module defines correctness semantics only. It does not claim throughput
    improvement until measured on exact draft/verifier checkpoints and hardware.
    """

    def __init__(self, draft_generate: Callable[[list[int], int], list[int]], verifier_next_token: Callable[[list[int]], int]):
        self.draft_generate = draft_generate
        self.verifier_next_token = verifier_next_token

    def step(self, prefix: Sequence[int], draft_tokens: int = 4) -> SpeculativeReceipt:
        proposed = self.draft_generate(list(prefix), int(draft_tokens))
        return verify_draft_tokens(prefix, proposed, self.verifier_next_token)

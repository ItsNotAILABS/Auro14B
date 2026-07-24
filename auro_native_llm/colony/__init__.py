"""Colony LLM — many mini Python models compose one organism.

Like humans and germs: a host of specialist micro-models (skills, spectral,
code, reason, memory, planner) that together act as one larger intelligence.
"""

from auro_native_llm.colony.organism import ColonyLLM, build_colony

__all__ = ["ColonyLLM", "build_colony"]

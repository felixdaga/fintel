"""Native in-process agent logic.

Agents that *are* the logic — they run in-process against fintel's own
``OpenRouter`` client (or are platform-agnostic pipelines), with no foreign
CLI harness. Distinct from ``fintel/agents/adapters/``, which wraps hosts
and external agents (OpenClaw/Claude Code CLIs, the fintel optimized host).

Members:
- ``llm_agent.py`` — the generic single-call / tool-loop LLM agent.
- ``optimized_agent.py`` — the standalone, platform-agnostic four-call
  specialist pipeline (no fintel imports). Portable across strategies and
  backtesting hosts. The fintel host that wires it to cells is
  ``fintel.agents.adapters.optimized``.
"""

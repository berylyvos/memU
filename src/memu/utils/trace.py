from __future__ import annotations

from contextvars import ContextVar

# Shared trace ID for correlating workflow steps with LLM/embedding calls.
# Each asyncio Task gets its own copy, so concurrent requests don't interfere.
trace_id: ContextVar[str] = ContextVar("trace_id", default="")

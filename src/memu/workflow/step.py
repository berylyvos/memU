from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from memu.utils.trace import trace_id as _trace_id

if TYPE_CHECKING:
    from memu.workflow.interceptor import WorkflowInterceptorRegistry

logger = logging.getLogger(__name__)

WorkflowState = dict[str, Any]
WorkflowContext = Mapping[str, Any] | None
WorkflowHandler = Callable[[WorkflowState, WorkflowContext], Awaitable[WorkflowState] | WorkflowState]


@dataclass
class WorkflowStep:
    step_id: str
    role: str
    handler: WorkflowHandler
    description: str = ""
    requires: set[str] = field(default_factory=set)
    produces: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    config: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> WorkflowStep:
        """Create a shallow copy with copied mutable fields but shared handler."""
        return WorkflowStep(
            step_id=self.step_id,
            role=self.role,
            handler=self.handler,  # Keep reference, don't copy
            description=self.description,
            requires=set(self.requires),
            produces=set(self.produces),
            capabilities=set(self.capabilities),
            config=dict(self.config),
        )

    async def run(self, state: WorkflowState, context: WorkflowContext) -> WorkflowState:
        result = self.handler(state, context)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            msg = f"Workflow step '{self.step_id}' must return a mapping, got {type(result).__name__}"
            raise TypeError(msg)
        return dict(result)


async def run_steps(
    name: str,
    steps: list[WorkflowStep],
    initial_state: WorkflowState,
    context: WorkflowContext = None,
    interceptor_registry: WorkflowInterceptorRegistry | None = None,
) -> WorkflowState:
    from memu.workflow.interceptor import (
        WorkflowStepContext,
        run_after_interceptors,
        run_before_interceptors,
        run_on_error_interceptors,
    )

    # Assign a trace ID for this workflow run if not already set (supports nested calls)
    trace_id = _trace_id.get()
    if not trace_id:
        trace_id = uuid.uuid4().hex[:8]
        _trace_id.set(trace_id)

    snapshot = interceptor_registry.snapshot() if interceptor_registry else None
    strict = interceptor_registry.strict if interceptor_registry else False

    state = dict(initial_state)
    workflow_start = time.perf_counter()
    for step in steps:
        missing = step.requires - state.keys()
        if missing:
            msg = f"Workflow '{name}' missing required keys for step '{step.step_id}': {', '.join(sorted(missing))}"
            raise KeyError(msg)
        step_context: dict[str, Any] = dict(context) if context else {}
        step_context["step_id"] = step.step_id
        if step.config:
            step_context["step_config"] = step.config

        # Build interceptor context
        interceptor_ctx = WorkflowStepContext(
            workflow_name=name,
            step_id=step.step_id,
            step_role=step.role,
            step_context=step_context,
        )

        # Run before interceptors
        if snapshot and snapshot.before:
            await run_before_interceptors(snapshot.before, interceptor_ctx, state, strict=strict)

        try:
            t0 = time.perf_counter()
            state = await step.run(state, step_context)
            elapsed = time.perf_counter() - t0
            logger.info(
                "trace=%s workflow=%s step=%s elapsed=%.3fs",
                trace_id, name, step.step_id, elapsed,
            )
        except Exception as e:
            if snapshot and snapshot.on_error:
                await run_on_error_interceptors(snapshot.on_error, interceptor_ctx, state, e, strict=strict)
            raise

        # Run after interceptors
        if snapshot and snapshot.after:
            await run_after_interceptors(snapshot.after, interceptor_ctx, state, strict=strict)

    total = time.perf_counter() - workflow_start
    logger.info("trace=%s workflow=%s total_elapsed=%.3fs", trace_id, name, total)
    return state

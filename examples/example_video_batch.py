"""
Example: Batch Video Processing with Per-File Categories and Concurrency Control

Usage:
    export OPENAI_API_KEY=your_api_key
    python examples/example_video_batch.py --video-dir /path/to/videos [--concurrency 4] [--output-dir examples/output/video_batch]

Each mp4 file must follow the naming convention: 'xxx.category.mp4'
The category is extracted from the last dot-separated segment before the extension.

Example: 'interview_01.cooking.mp4' → category 'cooking'
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

src_path = os.path.abspath("src")
sys.path.insert(0, src_path)

from memu.app import MemoryService  # noqa: E402
from memu.llm.wrapper import LLMCallContext, LLMRequestView, LLMResponseView, LLMUsage  # noqa: E402
from memu.workflow.interceptor import WorkflowStepContext  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoTask:
    path: Path
    category_name: str
    category_description: str


@dataclass
class TaskResult:
    path: Path
    category_name: str
    success: bool
    items_count: int = 0
    elapsed: float = 0.0
    error: str | None = None


@dataclass
class BatchStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_items: int = 0
    elapsed: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace collector
# ---------------------------------------------------------------------------

@dataclass
class LLMTraceRecord:
    file: str
    step_id: str
    kind: str          # chat / embed / vision / transcribe
    profile: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None
    status: str        # ok / error
    error: str | None = None


@dataclass
class StepTraceRecord:
    file: str
    step_id: str
    step_role: str
    elapsed_ms: float
    status: str        # ok / error
    error: str | None = None


class TraceCollector:
    """Thread-safe collector for workflow step and LLM call traces."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_records: list[LLMTraceRecord] = []
        self.step_records: list[StepTraceRecord] = []
        # per-step wall-clock start times keyed by (file, step_id)
        self._step_starts: dict[tuple[str, str], float] = {}
        # current file being processed per asyncio task (task_id -> filename)
        self._task_file: dict[int, str] = {}

    # -- file tracking -------------------------------------------------------

    def set_current_file(self, filename: str) -> None:
        tid = id(asyncio.current_task())
        with self._lock:
            self._task_file[tid] = filename

    def current_file(self) -> str:
        tid = id(asyncio.current_task())
        with self._lock:
            return self._task_file.get(tid, "-")

    def clear_current_file(self) -> None:
        tid = id(asyncio.current_task())
        with self._lock:
            self._task_file.pop(tid, None)

    # -- workflow step -------------------------------------------------------

    def on_step_before(self, step_ctx: Any, state: Any) -> None:
        key = (self.current_file(), step_ctx.step_id)
        with self._lock:
            self._step_starts[key] = time.monotonic()

    def on_step_after(self, step_ctx: Any, state: Any) -> None:
        fname = self.current_file()
        key = (fname, step_ctx.step_id)
        with self._lock:
            t0 = self._step_starts.pop(key, None)
        elapsed_ms = (time.monotonic() - t0) * 1000 if t0 else 0.0
        with self._lock:
            self.step_records.append(StepTraceRecord(
                file=fname,
                step_id=step_ctx.step_id,
                step_role=step_ctx.step_role,
                elapsed_ms=round(elapsed_ms, 1),
                status="ok",
            ))

    def on_step_error(self, step_ctx: Any, state: Any, error: Exception) -> None:
        fname = self.current_file()
        key = (fname, step_ctx.step_id)
        with self._lock:
            t0 = self._step_starts.pop(key, None)
        elapsed_ms = (time.monotonic() - t0) * 1000 if t0 else 0.0
        with self._lock:
            self.step_records.append(StepTraceRecord(
                file=fname,
                step_id=step_ctx.step_id,
                step_role=step_ctx.step_role,
                elapsed_ms=round(elapsed_ms, 1),
                status="error",
                error=str(error),
            ))

    # -- LLM calls -----------------------------------------------------------

    def on_llm_after(self, ctx: Any, req: Any, resp: Any, usage: Any) -> None:
        with self._lock:
            self.llm_records.append(LLMTraceRecord(
                file=self.current_file(),
                step_id=ctx.step_id or "-",
                kind=req.kind,
                profile=ctx.profile or "-",
                model=ctx.model or "-",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=round(usage.latency_ms, 1) if usage.latency_ms else None,
                status="ok",
            ))

    def on_llm_error(self, ctx: Any, req: Any, error: Exception, usage: Any) -> None:
        with self._lock:
            self.llm_records.append(LLMTraceRecord(
                file=self.current_file(),
                step_id=ctx.step_id or "-",
                kind=req.kind,
                profile=ctx.profile or "-",
                model=ctx.model or "-",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=round(usage.latency_ms, 1) if usage.latency_ms else None,
                status="error",
                error=str(error),
            ))


def setup_tracing(service: MemoryService, collector: TraceCollector) -> None:
    """Wire collector into service LLM and workflow interceptors."""
    service.intercept_after_llm_call(collector.on_llm_after)
    service.intercept_on_error_llm_call(collector.on_llm_error)
    service.intercept_before_workflow_step(collector.on_step_before)
    service.intercept_after_workflow_step(collector.on_step_after)
    service.intercept_on_error_workflow_step(collector.on_step_error)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return "" if v is None else str(v)


def write_trace_report(
    collector: TraceCollector,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. step_trace.csv ------------------------------------------------
    step_csv = output_dir / "step_trace.csv"
    step_fields = ["file", "step_id", "step_role", "elapsed_ms", "status", "error"]
    with step_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=step_fields)
        w.writeheader()
        for r in collector.step_records:
            w.writerow({k: _fmt(getattr(r, k)) for k in step_fields})

    # ---- 2. llm_trace.csv -------------------------------------------------
    llm_csv = output_dir / "llm_trace.csv"
    llm_fields = ["file", "step_id", "kind", "profile", "model",
                  "input_tokens", "output_tokens", "total_tokens", "latency_ms", "status", "error"]
    with llm_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=llm_fields)
        w.writeheader()
        for r in collector.llm_records:
            w.writerow({k: _fmt(getattr(r, k)) for k in llm_fields})

    # ---- 3. step_summary.md  (per step_id aggregated) ---------------------
    step_agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_ms": 0.0, "errors": 0}
    )
    for r in collector.step_records:
        a = step_agg[r.step_id]
        a["role"] = r.step_role
        a["count"] += 1
        a["total_ms"] += r.elapsed_ms
        if r.status == "error":
            a["errors"] += 1

    step_md = output_dir / "step_summary.md"
    with step_md.open("w", encoding="utf-8") as f:
        f.write("# Workflow Step Summary\n\n")
        f.write("| step_id | role | calls | avg_ms | total_ms | errors |\n")
        f.write("|---------|------|------:|-------:|---------:|-------:|\n")
        for sid, a in sorted(step_agg.items(), key=lambda x: -x[1]["total_ms"]):
            avg = a["total_ms"] / a["count"] if a["count"] else 0
            f.write(f"| {sid} | {a.get('role','-')} | {a['count']} "
                    f"| {avg:.0f} | {a['total_ms']:.0f} | {a['errors']} |\n")

    # ---- 4. llm_summary.md  (per step_id × kind aggregated) --------------
    llm_agg: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_ms": 0.0, "in_tok": 0, "out_tok": 0, "errors": 0}
    )
    for r in collector.llm_records:
        key = (r.step_id, r.kind)
        a = llm_agg[key]
        a["model"] = r.model
        a["count"] += 1
        a["total_ms"] += r.latency_ms or 0
        a["in_tok"] += r.input_tokens or 0
        a["out_tok"] += r.output_tokens or 0
        if r.status == "error":
            a["errors"] += 1

    llm_md = output_dir / "llm_summary.md"
    with llm_md.open("w", encoding="utf-8") as f:
        f.write("# LLM Call Summary\n\n")
        f.write("| step_id | kind | model | calls | avg_ms | total_ms | in_tokens | out_tokens | errors |\n")
        f.write("|---------|------|-------|------:|-------:|---------:|----------:|-----------:|-------:|\n")
        for (sid, kind), a in sorted(llm_agg.items(), key=lambda x: -x[1]["total_ms"]):
            avg = a["total_ms"] / a["count"] if a["count"] else 0
            f.write(f"| {sid} | {kind} | {a.get('model','-')} | {a['count']} "
                    f"| {avg:.0f} | {a['total_ms']:.0f} "
                    f"| {a['in_tok']} | {a['out_tok']} | {a['errors']} |\n")

    logger.info("Trace reports written: %s", output_dir)

def parse_category_from_filename(mp4: Path) -> str:
    """Extract category from filename format 'xxx.category.mp4' → 'category'."""
    parts = mp4.stem.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].replace("-", "_").replace(" ", "_").lower()
    # Fallback: no dot separator, use full stem
    return mp4.stem.replace("-", "_").replace(" ", "_").lower()


def discover_videos(video_dir: Path) -> list[VideoTask]:
    """Scan directory for mp4 files. Filename format: 'xxx.category.mp4'."""
    tasks: list[VideoTask] = []
    for mp4 in sorted(video_dir.glob("**/*.mp4")):
        name = parse_category_from_filename(mp4)
        description = f"Videos related to {name}"
        tasks.append(VideoTask(path=mp4, category_name=name, category_description=description))
    return tasks


def build_unique_categories(tasks: list[VideoTask]) -> list[dict]:
    """Deduplicate categories across all tasks."""
    seen: dict[str, dict] = {}
    for t in tasks:
        if t.category_name not in seen:
            seen[t.category_name] = {
                "name": t.category_name,
                "description": t.category_description,
            }
    return list(seen.values())


def write_category_md(categories: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        name = cat.get("name", "unknown")
        description = cat.get("description", "")
        summary = cat.get("summary", "")
        filepath = output_dir / f"{name}.md"
        with filepath.open("w", encoding="utf-8") as f:
            f.write(f"# {name.replace('_', ' ').title()}\n\n")
            if description:
                f.write(f"*{description}*\n\n")
            if summary:
                cleaned = summary.replace("<content>", "").replace("</content>", "").strip()
                f.write(f"{cleaned}\n")
            else:
                f.write("*No content available*\n")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def process_one(
    service: MemoryService,
    task: VideoTask,
    semaphore: asyncio.Semaphore,
    collector: TraceCollector,
) -> TaskResult:
    """Process a single video file under the semaphore."""
    async with semaphore:
        collector.set_current_file(task.path.name)
        t0 = time.monotonic()
        try:
            logger.info("Processing: %s (category=%s)", task.path.name, task.category_name)
            result = await service.memorize(
                resource_url=str(task.path),
                modality="video",
            )
            items_count = len(result.get("items", []))
            elapsed = time.monotonic() - t0
            logger.info("Done: %s — %d items in %.1fs", task.path.name, items_count, elapsed)
            return TaskResult(
                path=task.path,
                category_name=task.category_name,
                success=True,
                items_count=items_count,
                elapsed=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error("Failed: %s — %s", task.path.name, e)
            return TaskResult(
                path=task.path,
                category_name=task.category_name,
                success=False,
                elapsed=elapsed,
                error=str(e),
            )
        finally:
            collector.clear_current_file()


async def run_batch(
    tasks: list[VideoTask],
    service: MemoryService,
    concurrency: int,
    collector: TraceCollector,
) -> tuple[list[TaskResult], list[dict]]:
    semaphore = asyncio.Semaphore(concurrency)
    coros = [process_one(service, t, semaphore, collector) for t in tasks]
    results: list[TaskResult] = await asyncio.gather(*coros)

    # Collect final category state from service store
    store = service._get_database()
    ctx = service._get_context()
    categories = [
        service._model_dump_without_embeddings(store.memory_category_repo.categories[cid])
        for cid in ctx.category_ids
        if cid in store.memory_category_repo.categories
    ]
    return results, categories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(video_dir: Path, concurrency: int, output_dir: Path) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # 1. Discover videos
    tasks = discover_videos(video_dir)
    if not tasks:
        logger.warning("No mp4 files found in %s", video_dir)
        return

    logger.info("Found %d video(s), concurrency=%d", len(tasks), concurrency)

    # 2. Build unique categories from all tasks
    unique_categories = build_unique_categories(tasks)
    logger.info("Unique categories: %d", len(unique_categories))

    # 3. Initialize service — register all categories upfront
    service = MemoryService(
        llm_profiles={
            "default": {
                "api_key": api_key,
                "chat_model": "gpt-4o-mini",
            },
        },
        memorize_config={"memory_categories": unique_categories},
    )

    # 4. Setup tracing
    collector = TraceCollector()
    setup_tracing(service, collector)

    # 5. Run batch
    t0 = time.monotonic()
    results, categories = await run_batch(tasks, service, concurrency, collector)
    total_elapsed = time.monotonic() - t0

    # 5. Compute stats
    stats = BatchStats(
        total=len(results),
        elapsed=total_elapsed,
    )
    for r in results:
        if r.success:
            stats.succeeded += 1
            stats.total_items += r.items_count
        else:
            stats.failed += 1
            stats.errors.append((r.path.name, r.error or "unknown"))

    # 6. Write output
    write_category_md(categories, output_dir)

    # 7. Write trace reports
    write_trace_report(collector, output_dir)

    # 7. Write per-file result summary
    summary_path = output_dir / "results.json"
    summary = {
        "stats": {
            "total": stats.total,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "total_items": stats.total_items,
            "elapsed_seconds": round(stats.elapsed, 2),
        },
        "results": [
            {
                "file": r.path.name,
                "category": r.category_name,
                "success": r.success,
                "items": r.items_count,
                "elapsed": round(r.elapsed, 2),
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. Print summary
    print(f"\n{'='*50}")
    print(f"Batch complete: {stats.succeeded}/{stats.total} succeeded in {stats.elapsed:.1f}s")
    print(f"Total memory items extracted: {stats.total_items}")
    print(f"Categories written: {len(categories)}")
    print(f"Output: {output_dir}/")
    if stats.errors:
        print(f"\nFailed ({stats.failed}):")
        for fname, err in stats.errors:
            print(f"  {fname}: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch video memorization with concurrency control")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing mp4 files")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent memorize calls (default: 4)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/video_batch"),
        help="Output directory for category markdown files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.video_dir, args.concurrency, args.output_dir))

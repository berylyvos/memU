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
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

src_path = os.path.abspath("src")
sys.path.insert(0, src_path)

from memu.app import MemoryService  # noqa: E402

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
# Helpers
# ---------------------------------------------------------------------------

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
) -> TaskResult:
    """Process a single video file under the semaphore."""
    async with semaphore:
        t0 = time.monotonic()
        try:
            logger.info("Processing: %s (category=%s)", task.path.name, task.category_name)
            result = await service.memorize(
                resource_url=str(task.path),
                modality="video",
            )
            items_count = len(result.get("items", []))
            elapsed = time.monotonic() - t0
            logger.info(
                "Done: %s — %d items in %.1fs",
                task.path.name, items_count, elapsed,
            )
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


async def run_batch(
    tasks: list[VideoTask],
    service: MemoryService,
    concurrency: int,
) -> tuple[list[TaskResult], list[dict]]:
    semaphore = asyncio.Semaphore(concurrency)
    coros = [process_one(service, t, semaphore) for t in tasks]
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

    # 4. Run batch
    t0 = time.monotonic()
    results, categories = await run_batch(tasks, service, concurrency)
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

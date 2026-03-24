"""
Run all Online-Mind2Web benchmark tasks against the Browser Use Cloud API v3.

Usage:
    python run.py [options]

Results are saved to results/{task_id}/result.json.
Completed tasks are skipped automatically (resumable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Browser Use v3 API client
# ---------------------------------------------------------------------------

BASE_URL = "https://api.browser-use.com/api/v3"
TERMINAL_STATUSES = {"stopped", "timed_out", "error"}


class V3Client:
    def __init__(self, api_key: str) -> None:
        self._headers = {
            "X-Browser-Use-API-Key": api_key,
            "Content-Type": "application/json",
        }

    async def create_session(self, task: str, model: str = "bu-mini") -> str:
        """Create a session and dispatch a task. Returns the session_id."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{BASE_URL}/sessions",
                headers=self._headers,
                json={"task": task, "model": model},
            )
            resp.raise_for_status()
            return resp.json()["id"]

    async def poll_until_done(
        self,
        session_id: str,
        poll_interval: float = 6.0,
        max_wait: int = 3600,
    ) -> dict:
        """Poll GET /sessions/{id} until a terminal status is reached."""
        deadline = time.monotonic() + max_wait
        async with httpx.AsyncClient(timeout=30) as client:
            while time.monotonic() < deadline:
                resp = await client.get(
                    f"{BASE_URL}/sessions/{session_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
                if data["status"] in TERMINAL_STATUSES:
                    return data
                await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Session {session_id} did not finish within {max_wait}s")

    async def stop_session(self, session_id: str) -> None:
        """Stop a session. Best-effort — errors are logged, not raised."""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                await client.post(
                    f"{BASE_URL}/sessions/{session_id}/stop",
                    headers=self._headers,
                )
            except Exception as exc:
                logger.warning("Could not stop session %s: %s", session_id, exc)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_tasks(split: str = "test", limit: int | None = None) -> list[dict]:
    """Load benchmark tasks from the Online-Mind2Web HuggingFace dataset."""
    from datasets import load_dataset

    logger.info("Loading osunlp/Online-Mind2Web (split=%s) …", split)
    ds = load_dataset("osunlp/Online-Mind2Web", split=split)

    tasks: list[dict] = []
    for row in ds:
        tasks.append(
            {
                "task_id": row["task_id"],
                "website": row["website"],
                "task": row["confirmed_task"],
                "reference_length": row.get("reference_length"),
                "level": row.get("level"),
            }
        )

    if limit is not None:
        tasks = tasks[:limit]

    logger.info("Loaded %d tasks", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def _build_prompt(website: str, task: str) -> str:
    """Prepend the starting website to the task description."""
    if website and website.lower() not in task.lower():
        return f"Go to {website} and {task}"
    return task


_RETRIABLE_FINALS = {"task was cancelled.", "task ended unexpectedly."}


def _is_retriable(result: dict) -> bool:
    if result.get("_meta", {}).get("status") == "error":
        return True
    raw = (result.get("final_result_response") or "").strip().lower()
    return raw in _RETRIABLE_FINALS


async def run_task(
    client: V3Client,
    task: dict,
    model: str,
    results_dir: Path,
    max_retries: int = 3,
) -> dict:
    """Execute one benchmark task and persist the result to disk."""
    task_id = task["task_id"]
    task_dir = results_dir / task_id
    result_path = task_dir / "result.json"

    # Resume: skip if already saved with a non-retriable result
    if result_path.exists():
        with open(result_path) as f:
            saved = json.load(f)
        if not _is_retriable(saved):
            logger.info("Skipping %s — already completed", task_id)
            return saved
        logger.info("Retrying %s — previous result was retriable", task_id)

    task_dir.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(task["website"], task["task"])
    result: dict = {}

    for attempt in range(max_retries + 1):
        session_id: str | None = None
        try:
            await asyncio.sleep(random.uniform(0.5, 3.0))  # jitter

            logger.info(
                "Running  %s | %.80s … (attempt %d/%d)",
                task_id,
                task["task"],
                attempt + 1,
                max_retries + 1,
            )
            session_id = await client.create_session(prompt, model=model)
            session_data = await client.poll_until_done(session_id)

            result = {
                "task_id": task_id,
                "task": task["task"],
                "final_result_response": session_data.get("output") or "",
                "_meta": {
                    "website": task["website"],
                    "model": model,
                    "session_id": session_id,
                    "status": session_data.get("status"),
                    "is_task_successful": session_data.get("isTaskSuccessful"),
                    "step_count": session_data.get("stepCount", 0),
                    "total_cost_usd": session_data.get("totalCostUsd", "0"),
                    "attempt": attempt + 1,
                },
            }

        except Exception as exc:
            logger.error(
                "Task %s failed (attempt %d/%d): %s",
                task_id,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            result = {
                "task_id": task_id,
                "task": task["task"],
                "final_result_response": "",
                "_meta": {
                    "website": task["website"],
                    "model": model,
                    "session_id": session_id,
                    "status": "error",
                    "error": str(exc),
                    "attempt": attempt + 1,
                },
            }
        finally:
            if session_id:
                await client.stop_session(session_id)

        if not _is_retriable(result):
            break
        if attempt < max_retries:
            logger.warning(
                "Task %s retriable (attempt %d/%d) — retrying …",
                task_id,
                attempt + 1,
                max_retries + 1,
            )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "Saved    %s (status=%s)",
        task_id,
        result.get("_meta", {}).get("status", "?"),
    )
    return result


async def run_all(
    api_key: str,
    model: str,
    results_dir: Path,
    tasks: list[dict],
    concurrency: int,
) -> list[dict]:
    """Run all tasks with bounded concurrency."""
    client = V3Client(api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(tasks)
    done_count = 0
    count_lock = asyncio.Lock()

    async def bounded(task: dict) -> dict:
        nonlocal done_count
        async with sem:
            logger.info("[%d done / %d total] Starting %s", done_count, total, task["task_id"])
            result = await run_task(client, task, model, results_dir)
        async with count_lock:
            done_count += 1
            status = result.get("_meta", {}).get("status", "?")
            logger.info("[%d/%d done] %s — status=%s", done_count, total, task["task_id"], status)
        return result

    gathered = await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=True)

    results: list[dict] = []
    for item in gathered:
        if isinstance(item, Exception):
            logger.error("Unhandled exception: %s", item)
        else:
            results.append(item)
    return results


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Online-Mind2Web tasks via the Browser Use Cloud API v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="bu-ultra",
        choices=["bu-mini", "bu-max", "bu-ultra"],
        help="Browser Use model to use",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help="Number of tasks to run in parallel",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save results",
    )
    p.add_argument(
        "--split",
        default="test",
        help="HuggingFace dataset split",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks (useful for smoke tests)",
    )
    p.add_argument(
        "--task-ids-file",
        default=None,
        help="Path to a JSON file containing a list of task IDs to run (filters the dataset)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)

    api_key = os.getenv("BROWSER_USE_API_KEY")
    if not api_key:
        sys.exit("BROWSER_USE_API_KEY is not set. Add it to your .env file.")

    tasks = load_tasks(split=args.split, limit=args.limit)

    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            task_ids = set(json.load(f))
        tasks = [t for t in tasks if t["task_id"] in task_ids]
        logger.info("Filtered to %d tasks from %s", len(tasks), args.task_ids_file)

    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running %d tasks | model=%s | concurrency=%d | results_dir=%s",
        len(tasks),
        args.model,
        args.concurrency,
        results_dir,
    )

    results = asyncio.run(
        run_all(
            api_key=api_key,
            model=args.model,
            results_dir=results_dir,
            tasks=tasks,
            concurrency=args.concurrency,
        )
    )

    completed = sum(1 for r in results if r.get("_meta", {}).get("status") == "stopped")
    errors = sum(1 for r in results if r.get("_meta", {}).get("status") == "error")
    logger.info(
        "Done. %d completed, %d errors. Results in %s",
        completed,
        errors,
        results_dir,
    )


if __name__ == "__main__":
    main()

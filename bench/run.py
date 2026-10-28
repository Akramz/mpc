"""Time a headless Claude agent on saved geospatial tasks, using this checkout's mpc and skill.

Each run gets a fresh working folder and an empty store, so runs compare cold to cold. One row
per run goes to bench/results.jsonl; compare rows across commits to see if a change helps.

    uv run python bench/run.py [TASK ...] --repeat 4 --jobs 8 --label "what changed"
    uv run python bench/run.py --compare "before" "what changed"

Agents take different paths, so one run says little: repeat each task, compare medians with
their bootstrap intervals, and keep --jobs equal between the labels you compare.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS = json.loads((REPO / "bench" / "tasks.json").read_text())
RESULTS = REPO / "bench" / "results.jsonl"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def skill() -> str:
    body = (REPO / "mpc" / "skill" / "SKILL.md").read_text().split("---", 2)[2].strip()
    assert body, "empty skill text"
    return body


def run(task: dict, label: str, jobs: int) -> dict:
    work = Path(tempfile.mkdtemp(prefix=f"mpc-bench-{task['id']}-"))
    env = {**os.environ, "MPC_HOME": str(work / "home")}
    env["PATH"] = f"{REPO / '.venv' / 'bin'}:{env['PATH']}"
    cmd = [
        "claude", "-p", task["prompt"], "--append-system-prompt", skill(),
        "--disallowedTools", "Skill", "--allowedTools", "Bash(mpc *)", "Read",
        "--output-format", "stream-json", "--verbose",
    ]  # fmt: skip
    t = time.time()
    out = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True).stdout
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    (work / "transcript.jsonl").write_text(out)
    calls, failed = [], 0
    results = {}
    for e in events:
        if e.get("type") == "user" and isinstance(e["message"]["content"], list):
            for c in e["message"]["content"]:
                if c.get("type") == "tool_result":
                    text = (
                        c["content"] if isinstance(c["content"], str) else json.dumps(c["content"])
                    )
                    results[c["tool_use_id"]] = bool(c.get("is_error")) or "error:" in text[:500]
    for e in events:
        if e.get("type") == "assistant":
            for c in e["message"]["content"]:
                if c.get("type") == "tool_use":
                    calls.append(c["input"].get("command") or c["input"].get("file_path") or "")
                    failed += results.get(c["id"], False)
    end = next((e for e in reversed(events) if e.get("type") == "result"), {})
    answer = end.get("result", "")
    init = next((e for e in events if e.get("subtype") == "init"), {})
    return {
        "task": task["id"],
        "label": label,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "--short", "HEAD")
        + ("+dirty" if git("status", "--porcelain") else ""),
        "model": init.get("model"),
        "jobs": jobs,
        "ok": bool(answer) and all(re.search(c, answer) for c in task["checks"]),
        "minutes": round((end.get("duration_ms") or (time.time() - t) * 1000) / 60000, 2),
        "turns": end.get("num_turns"),
        "cost_usd": round(end.get("total_cost_usd") or 0, 3),
        "tool_calls": len(calls),
        "mpc_calls": sum("mpc " in c for c in calls),
        "failed_calls": failed,
        "transcript": str(work / "transcript.jsonl"),
        "answer": answer,
    }


def interval(draw, n: int = 4000) -> tuple[float, float]:
    """A 95 % bootstrap interval: `draw()` gives the statistic of one resample."""
    boots = sorted(draw() for _ in range(n))
    return boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def resampled_median(rng: random.Random, xs: list[float]) -> float:
    return statistics.median(rng.choices(xs, k=len(xs)))


def summary(rows: list[dict]) -> None:
    head = f"{'minutes (95% CI)':>20} {'failed':>7} {'calls':>6} {'cost':>6}"
    print(f"{'task':<10} {'runs':>4} {'ok':>4} {head}")
    for task in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == task]
        mins = [r["minutes"] for r in rs]
        rng = random.Random(0)
        lo, hi = interval(lambda: resampled_median(rng, mins))
        med = lambda k: statistics.median(r[k] or 0 for r in rs)  # noqa: E731
        span = f"{statistics.median(mins):.1f} ({lo:.1f}-{hi:.1f})"
        nums = f"{med('failed_calls'):>7.1f} {med('tool_calls'):>6.0f} {med('cost_usd'):>6.2f}"
        print(f"{task:<10} {len(rs):>4} {sum(r['ok'] for r in rs):>4} {span:>20} {nums}")


def compare(a: str, b: str) -> None:
    """Per task: B's median minutes minus A's, with a 95 % bootstrap interval."""
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    rng = random.Random(0)
    for task in dict.fromkeys(r["task"] for r in rows):
        xa = [r["minutes"] for r in rows if r["task"] == task and r["label"] == a]
        xb = [r["minutes"] for r in rows if r["task"] == task and r["label"] == b]
        if not xa or not xb:
            continue
        lo, hi = interval(lambda: resampled_median(rng, xb) - resampled_median(rng, xa))
        d = statistics.median(xb) - statistics.median(xa)
        verdict = "faster" if hi < 0 else "slower" if lo > 0 else "no clear change"
        runs = f"{len(xa)} vs {len(xb)} runs"
        print(f"{task:<10} {runs}: {d:+.1f} min ({lo:+.1f} to {hi:+.1f}), {verdict}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*", help="task ids (default: all)")
    p.add_argument("--repeat", type=int, default=1, help="runs of each task; 3-5 to compare")
    p.add_argument(
        "--jobs", type=int, default=1, help="sessions at once; keep it equal across labels"
    )
    p.add_argument("--label", default="", help="names this batch, e.g. the change it tests")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two labels, run nothing")
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    chosen = [t for t in TASKS if not a.tasks or t["id"] in a.tasks]
    todo = [t for t in chosen for _ in range(a.repeat)]
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda t: run(t, a.label, a.jobs), todo))
    with RESULTS.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary(rows)
    print(f"{len(rows)} runs appended to {RESULTS}")


if __name__ == "__main__":
    main()

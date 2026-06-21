#!/usr/bin/env python3
"""Run concurrent pipeline workers and summarize scalability measurements."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(
    samples: list[dict[str, object]], elapsed_seconds: float
) -> dict[str, object]:
    successful = [sample for sample in samples if sample.get("ok")]
    latencies = [float(sample["latency_ms"]) for sample in samples]
    memory = [float(sample.get("memory_mb", 0.0)) for sample in samples]
    return {
        "samples": len(samples),
        "successful_frames": len(successful),
        "failed_frames": len(samples) - len(successful),
        "throughput_fps": round(
            len(successful) / elapsed_seconds if elapsed_seconds > 0 else 0.0, 3
        ),
        "mean_latency_ms": round(
            sum(latencies) / len(latencies) if latencies else 0.0, 3
        ),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "p99_latency_ms": round(percentile(latencies, 0.99), 3),
        "peak_memory_mb": round(max(memory, default=0.0), 3),
    }


def run_worker(command: list[str], worker_id: int, timeout: float) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "worker_id": worker_id,
        "ok": completed.returncode == 0,
        "latency_ms": elapsed_ms,
        "memory_mb": 0.0,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("runs/scalability.json"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Provide a command after --.")
    jobs = args.workers * args.repetitions
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_worker, command, index, args.timeout)
            for index in range(jobs)
        ]
        samples = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    result = {
        "workers": args.workers,
        "repetitions": args.repetitions,
        "command": command,
        "summary": summarize_samples(samples, elapsed),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

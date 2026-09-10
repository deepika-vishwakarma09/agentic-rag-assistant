
"""
tracer.py
-----------
The purpose of observability is to record a complete "trace" for every query —
which route was taken, how long each step took, how many tokens were used.

Production companies (like Langfuse, LangSmith) do this same thing at a
large scale. Here we are building a simple, local version — we write each
query's trace to a JSONL file. This lets us later analyze: "which route is
used the most", "which queries are slow", "where are errors occurring" —
exactly the kind of things needed when debugging in production.

JSONL (JSON Lines) format is used because each line is an independent
JSON object — appending new traces is easy, and it can also be
loaded and analyzed with Pandas.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Optional
from contextlib import contextmanager

TRACE_LOG_PATH = "data/traces.jsonl"


class QueryTracer:
    """
    Collects a trace for a single query. Time the steps inside a `with` block,
    then call save() at the end.
    """

    def __init__(self, question: str):
        self.question = question
        self.start_time = time.time()
        self.steps: Dict[str, float] = {}   # step_name -> duration in seconds
        self.route: Optional[str] = None
        self.was_corrected: bool = False
        self.error: Optional[str] = None

    @contextmanager
    def track(self, step_name: str):
        """
        Usage: with tracer.track("retrieval"): ...code...
        Automatically records the time taken for that step.
        """
        step_start = time.time()
        try:
            yield
        except Exception as e:
            self.error = f"{step_name}: {str(e)}"
            raise
        finally:
            self.steps[step_name] = round(time.time() - step_start, 3)

    def set_route(self, route: str):
        self.route = route

    def set_corrected(self, was_corrected: bool):
        self.was_corrected = was_corrected

    def save(self):
        """
        Appends the complete trace as a JSON line to the file.
        """
        os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)

        total_time = round(time.time() - self.start_time, 3)

        record = {
            "timestamp": datetime.now().isoformat(),
            "question": self.question,
            "route": self.route,
            "was_corrected": self.was_corrected,
            "step_timings": self.steps,
            "total_time_seconds": total_time,
            "error": self.error
        }

        with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record


def load_traces() -> list:
    """Loads saved traces back for analysis."""
    if not os.path.exists(TRACE_LOG_PATH):
        return []

    traces = []
    with open(TRACE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def print_summary():
    """
    Prints a quick summary — total queries, average time,
    route breakdown. This is the simplest version of a "dashboard".
    """
    traces = load_traces()

    if not traces:
        print("No traces have been recorded yet.")
        return

    total = len(traces)
    doc_count = sum(1 for t in traces if t["route"] == "document")
    web_count = sum(1 for t in traces if t["route"] == "web")
    corrected_count = sum(1 for t in traces if t["was_corrected"])
    avg_time = sum(t["total_time_seconds"] for t in traces) / total

    print(f"Total queries traced: {total}")
    print(f"  Document route: {doc_count}")
    print(f"  Web route: {web_count}")
    print(f"  Self-corrected: {corrected_count}")
    print(f"  Average total time: {avg_time:.2f}s")


# --- Quick test ---
if __name__ == "__main__":
    import time as t

    tracer = QueryTracer("What is the refund policy?")

    with tracer.track("retrieval"):
        t.sleep(0.5)  # dummy delay, simulating retrieval

    with tracer.track("generation"):
        t.sleep(0.3)  # dummy delay, simulating LLM call

    tracer.set_route("document")
    tracer.set_corrected(False)
    record = tracer.save()

    print("Trace saved:")
    print(json.dumps(record, indent=2))

    print("\n--- Summary ---")
    print_summary()
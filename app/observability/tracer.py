
"""
tracer.py
-----------
Observability ka kaam hai: har query ka poora "trace" record karna —
kaunsa route liya, har step mein kitna time laga, kitne tokens use hue.

Production companies (jaise Langfuse, LangSmith) isi cheez ko bade scale
pe karte hain. Hum yahan ek simple, local version bana rahe hain — ek
JSONL file mein har query ka trace likh dete hain. Isse hum baad mein
dekh sakte hain: "kaunsa route sabse zyada use hota hai", "kaunsi
queries slow hain", "kahan errors aa rahe hain" — bilkul waisa jaisa
production mein debug karte waqt chahiye hota hai.

JSONL (JSON Lines) format isliye use kiya kyunki har line ek independent
JSON object hoti hai — naye traces append karna easy hota hai, aur
Pandas se load karke analyze karna bhi.
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
    Ek single query ke liye trace collect karta hai. `with` block ke
    andar steps ko time karo, phir save() call karo end mein.
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
        Automatically us step ka time record kar leta hai.
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
        Poora trace ek JSON line ke roop mein file mein append karta hai.
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
    """Saved traces ko wapas load karta hai, analysis ke liye."""
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
    Ek quick summary print karta hai — kitni queries, average time,
    route breakdown. Ye "dashboard" ka simplest version hai.
    """
    traces = load_traces()

    if not traces:
        print("Abhi tak koi trace record nahi hui.")
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
        t.sleep(0.5)  # dummy delay, retrieval simulate kar rahe hain

    with tracer.track("generation"):
        t.sleep(0.3)  # dummy delay, LLM call simulate kar rahe hain

    tracer.set_route("document")
    tracer.set_corrected(False)
    record = tracer.save()

    print("Trace saved:")
    print(json.dumps(record, indent=2))

    print("\n--- Summary ---")
    print_summary()
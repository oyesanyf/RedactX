"""
Minimal thread-safe Prometheus metrics (text exposition format 0.0.4) without extra dependencies.
Metric labels never contain document content.
"""

import threading
from typing import Dict, Iterable, Tuple

LabelKey = Tuple[Tuple[str, str], ...]


def _fmt_labels(key: LabelKey, extra: Iterable[Tuple[str, str]] = ()) -> str:
    items = list(key) + list(extra)
    if not items:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in items) + "}"


class Counter:
    def __init__(self, name: str, help_text: str):
        self.name, self.help = name, help_text
        self._v: Dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._v[key] = self._v.get(key, 0.0) + amount

    def value(self, **labels) -> float:
        return self._v.get(tuple(sorted(labels.items())), 0.0)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, v in sorted(self._v.items()):
                lines.append(f"{self.name}{_fmt_labels(key)} {v}")
        return "\n".join(lines)


class Gauge(Counter):
    def set(self, value: float, **labels) -> None:
        with self._lock:
            self._v[tuple(sorted(labels.items()))] = float(value)

    def render(self) -> str:
        return super().render().replace(f"# TYPE {self.name} counter", f"# TYPE {self.name} gauge")


class Histogram:
    DEFAULT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

    def __init__(self, name: str, help_text: str, buckets=DEFAULT_BUCKETS):
        self.name, self.help, self.buckets = name, help_text, tuple(sorted(buckets))
        self._counts: Dict[LabelKey, list] = {}
        self._sum: Dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, b in enumerate(self.buckets):
                if value <= b:
                    counts[i] += 1
            counts[-1] += 1
            self._sum[key] = self._sum.get(key, 0.0) + value

    def count(self, **labels) -> int:
        c = self._counts.get(tuple(sorted(labels.items())))
        return c[-1] if c else 0

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key, counts in sorted(self._counts.items()):
                for b, c in zip(self.buckets, counts):
                    lines.append(f"{self.name}_bucket{_fmt_labels(key, [('le', repr(float(b)))])} {c}")
                lines.append(f"{self.name}_bucket{_fmt_labels(key, [('le', '+Inf')])} {counts[-1]}")
                lines.append(f"{self.name}_sum{_fmt_labels(key)} {self._sum[key]}")
                lines.append(f"{self.name}_count{_fmt_labels(key)} {counts[-1]}")
        return "\n".join(lines)


class Registry:
    def __init__(self):
        self.requests = Counter("redactx_requests_total", "HTTP requests by endpoint and status code")
        self.documents = Counter("redactx_documents_total", "Documents processed by action")
        self.findings = Counter("redactx_findings_total", "Findings by category")
        self.errors = Counter("redactx_errors_total", "Errors by type")
        self.rejected = Counter("redactx_rejected_total", "Rejected requests by reason")
        self.latency = Histogram("redactx_request_seconds", "Request latency in seconds by endpoint")
        self.inflight = Gauge("redactx_inflight_requests", "Requests currently being processed")
        self.ready = Gauge("redactx_ready", "1 when the model is loaded and warmed up")

    def render(self) -> str:
        parts = [m.render() for m in (self.requests, self.documents, self.findings, self.errors,
                                      self.rejected, self.latency, self.inflight, self.ready)]
        return "\n".join(parts) + "\n"

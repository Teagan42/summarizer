"""Lightweight telemetry helpers for the summarizer service."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

MetricEvent = dict[str, Any]
MetricSink = Callable[[MetricEvent], None]


logger = logging.getLogger("app.metrics")


class Metrics:
    """Expose a minimal sink-driven telemetry interface."""

    def __init__(self, sink: MetricSink | None = None) -> None:
        self._default_sink = sink or self._log_event
        self._sink: MetricSink = self._default_sink

    def _log_event(self, event: MetricEvent) -> None:
        logger.info("telemetry", extra={"telemetry": event})

    def set_sink(self, sink: MetricSink) -> None:
        """Replace the active sink, primarily for testing."""

        self._sink = sink

    def reset_sink(self) -> None:
        """Restore the default sink."""

        self._sink = self._default_sink

    @contextmanager
    def timer(self, stage: str, **fields: Any):
        """Record the elapsed time for *stage* and emit it to the sink."""

        sink = self._sink
        start = time.perf_counter()
        try:
            yield
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            sink({
                "stage": stage,
                "duration_ms": duration_ms,
                "status": "error",
                **fields,
            })
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000
            sink({"stage": stage, "duration_ms": duration_ms, "status": "ok", **fields})


metrics = Metrics()

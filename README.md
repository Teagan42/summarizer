# summarizer

FastAPI service that selects and compresses relevant snippets from long documents.

## `/compress` endpoint

Send a JSON payload to `/compress` with either a pre-segmented `texts` list **or** a
raw `document` string. Legacy integrations that already provide `texts` will
continue to work without modifications.

```json
{
  "document": "long body of text...",
  "mode": "task",
  "task": "optional task conditioning",
  "return_selection": false
}
```

When `document` is supplied the service automatically chunks it according to the
configured budgets before running selection. Overlap between chunks ensures the
selector can observe context that spans boundaries.

## Chunking configuration

Chunking behavior is controlled through environment variables surfaced in
`app/config.py`:

| Variable | Description | Default |
| --- | --- | --- |
| `CHUNK_TARGET_TOKENS` | Target token count per chunk before selection. | `900` |
| `CHUNK_OVERLAP_TOKENS` | Token overlap between adjacent chunks. | `120` |

Tune these values to trade off selection granularity against request size. The
service will fall back to whitespace segmentation when tokenizer libraries are
unavailable.

## Telemetry and monitoring

The service emits lightweight telemetry events for the selection and
compression stages. Each event includes:

- `stage`: Either `select` or `compress`.
- `duration_ms`: Wall-clock runtime for the stage.
- `status`: `ok` when the stage completed successfully, `error` when it raised.
- `mode` and `keep_ratio`: Effective request settings after defaults applied.
- `backend` and `model`: Identifier for the active compression backend.

Events are delivered to a sink exposed via `app.metrics.metrics`. By default
they are logged using the standard library logger under `app.metrics`. Replace
the sink with your monitoring pipeline by calling `metrics.set_sink` during
startup:

```python
from app.metrics import metrics


def startup() -> None:
    metrics.set_sink(lambda event: send_to_datadog("summarizer.telemetry", event))
```

If the sink raises an exception the request will fail, which keeps telemetry
failures visible in upstream alerting. Use `metrics.reset_sink()` to restore the
default logging behavior during tests or graceful shutdown.

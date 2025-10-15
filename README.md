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

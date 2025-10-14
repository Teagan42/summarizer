from app.models import CompressResponse


def test_compress_response_meta_is_isolated():
    first = CompressResponse(
        compressed="summary",
        kept_indices=[0],
        kept_count=1,
        original_count=1,
    )
    second = CompressResponse(
        compressed="another",
        kept_indices=[1],
        kept_count=1,
        original_count=1,
    )

    first.meta["foo"] = "bar"

    assert second.meta == {}
    assert first.meta is not second.meta

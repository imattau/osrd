from osrd_ops_intelligence.training import replay


def test_pinned_openapi_artifact_matches_hash() -> None:
    import hashlib

    assert hashlib.sha256(replay.SCHEMA_PATH.read_bytes()).hexdigest() == replay.SCHEMA_SHA256

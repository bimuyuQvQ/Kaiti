from env_rewriter.generate_candidates import parse_candidates


def test_parse_candidates() -> None:
    text = "\n".join(f"{index}. query {index}" for index in range(1, 9))
    assert parse_candidates(text) == [f"query {index}" for index in range(1, 9)]

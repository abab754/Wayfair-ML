"""Smoke test to verify the project is set up correctly."""


def test_import():
    import wands

    assert wands is not None

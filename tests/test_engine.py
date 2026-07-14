import numpy as np
import pytest

from consilience.engine import EngineError, suggest
from consilience.store import Record


def rec(key, links=(), vec=(1.0, 0.0)):
    return Record(key=key, rel=f"{key}.md", title=key.title(),
                  links=frozenset(links), vec=np.array(vec, dtype=np.float32))


def test_suggests_close_unlinked_pair():
    records = [
        rec("a", vec=(1.0, 0.0)),
        rec("b", vec=(0.98, 0.2)),   # very close to a
        rec("c", vec=(0.0, 1.0)),    # orthogonal
    ]
    results = suggest(records, top=10, threshold=0.75)
    pairs = {frozenset((s.a_title, s.b_title)) for s in results}
    assert frozenset(("A", "B")) in pairs
    assert frozenset(("A", "C")) not in pairs


def test_existing_link_is_never_suggested():
    records = [
        rec("a", links=("b",), vec=(1.0, 0.0)),
        rec("b", vec=(0.98, 0.2)),
    ]
    assert suggest(records, threshold=0.5) == []


def test_link_direction_does_not_matter():
    # b links to a; the a<->b pair must still be excluded.
    records = [
        rec("a", vec=(1.0, 0.0)),
        rec("b", links=("a",), vec=(0.98, 0.2)),
    ]
    assert suggest(records, threshold=0.5) == []


def test_threshold_filters():
    records = [rec("a", vec=(1.0, 0.0)), rec("b", vec=(0.0, 1.0))]
    assert suggest(records, threshold=0.5) == []


def test_focus_limits_to_one_note():
    records = [
        rec("a", vec=(1.0, 0.0)),
        rec("b", vec=(0.98, 0.2)),
        rec("c", vec=(0.97, 0.24)),
    ]
    results = suggest(records, threshold=0.5, focus="a")
    for s in results:
        assert "A" in (s.a_title, s.b_title)


def test_fewer_than_two_notes():
    assert suggest([rec("a")]) == []


def test_mixed_dimensions_raise():
    records = [rec("a", vec=(1.0, 0.0)), rec("b", vec=(1.0, 0.0, 0.0))]
    with pytest.raises(EngineError):
        suggest(records, threshold=0.1)

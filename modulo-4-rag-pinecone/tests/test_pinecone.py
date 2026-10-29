import pytest

from fakes import FakeIndex, FakePinecone, FakeStore, make_chunk
from ingest import ingest
from setup_index import IndexDimensionError, ensure_index

CHUNKS = [make_chunk("a.md", "uno", 0), make_chunk("a.md", "dos", 1), make_chunk("b.md", "tres")]


def test_creates_the_index_when_it_does_not_exist():
    pc = FakePinecone()
    assert ensure_index(pc, "docs", 768, cloud="aws", region="us-east-1") is True
    assert pc.created == [("docs", 768, "cosine")]


def test_reuses_an_existing_index_with_the_same_dimension():
    pc = FakePinecone({"docs": 768})
    assert ensure_index(pc, "docs", 768, cloud="aws", region="us-east-1") is False
    assert pc.created == []


def test_dimension_mismatch_is_an_error():
    pc = FakePinecone({"docs": 512})
    with pytest.raises(IndexDimensionError, match="512"):
        ensure_index(pc, "docs", 768, cloud="aws", region="us-east-1")


def test_first_ingest_uploads_everything():
    index = FakeIndex()
    store = FakeStore(index)
    report = ingest(index, store, CHUNKS, "ns")
    assert report.added == 3
    assert report.unchanged == 0
    assert sorted(store.added) == sorted(c.id for c in CHUNKS)


def test_second_ingest_uploads_nothing():
    index = FakeIndex()
    ingest(index, FakeStore(index), CHUNKS, "ns")
    store = FakeStore(index)
    report = ingest(index, store, CHUNKS, "ns")
    assert report.added == 0
    assert report.unchanged == 3
    assert store.added == []


def test_changed_chunks_replace_the_old_ones():
    index = FakeIndex()
    ingest(index, FakeStore(index), CHUNKS, "ns")
    changed = [make_chunk("a.md", "uno", 0), make_chunk("a.md", "texto nuevo", 1), CHUNKS[2]]
    changed[1].id = "a.md#001-otro"
    store = FakeStore(index)
    report = ingest(index, store, changed, "ns")
    assert report.added == 1
    assert report.deleted == 1
    assert index.deleted == [CHUNKS[1].id]


def test_chunks_of_removed_files_are_deleted():
    index = FakeIndex()
    ingest(index, FakeStore(index), CHUNKS, "ns")
    report = ingest(index, FakeStore(index), CHUNKS[:2], "ns")
    assert report.deleted == 1
    assert CHUNKS[2].id not in index.ids

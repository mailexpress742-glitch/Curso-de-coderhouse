import pytest
import tiktoken

from fakes import fake_embedder
from ingest import build_splitter, chunk_id, clean_text, ingest
from store import open_collection

LONG_TEXT = "\n\n".join(
    f"Párrafo {i}. " + "Las bases vectoriales guardan embeddings de texto. " * 12 for i in range(30)
)


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a.md").write_text(LONG_TEXT, encoding="utf-8")
    (folder / "b.txt").write_text("Un texto corto sobre ChromaDB.", encoding="utf-8")
    return folder


@pytest.fixture
def collection(tmp_path):
    return open_collection(tmp_path / "store", fake_embedder())


@pytest.fixture
def splitter():
    return build_splitter(500, 50)


def test_clean_text_normalizes_whitespace():
    dirty = "Hola  \r\n\r\n\r\n\r\nmundo​\x0c  \n"
    assert clean_text(dirty) == "Hola\n\nmundo"


def test_chunks_respect_the_token_limit(splitter):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = splitter.split_text(LONG_TEXT)
    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk)) <= 500 for chunk in chunks)


def test_consecutive_chunks_overlap(splitter):
    chunks = splitter.split_text(LONG_TEXT)
    # El final de un fragmento reaparece al comienzo del siguiente.
    tail = chunks[0].split()[-3:]
    assert " ".join(tail) in chunks[1]


def test_chunk_id_is_deterministic():
    assert chunk_id("a.md", 0, "texto") == chunk_id("a.md", 0, "texto")
    assert chunk_id("a.md", 0, "texto") != chunk_id("a.md", 0, "otro texto")


def test_ingest_stores_chunks_with_metadata(collection, data_dir, splitter):
    report = ingest(collection, data_dir, splitter)
    assert report.files == 2
    assert report.added == collection.count()
    sources = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}
    assert sources == {"a.md", "b.txt"}


def test_second_ingest_adds_nothing(collection, data_dir, splitter):
    first = ingest(collection, data_dir, splitter)
    second = ingest(collection, data_dir, splitter)
    assert second.added == 0
    assert second.unchanged == first.added
    assert collection.count() == first.added


def test_changed_file_replaces_its_old_chunks(collection, data_dir, splitter):
    ingest(collection, data_dir, splitter)
    (data_dir / "b.txt").write_text("Texto nuevo sobre otra cosa.", encoding="utf-8")
    report = ingest(collection, data_dir, splitter)
    assert report.added == 1
    assert report.deleted == 1
    stored = collection.get(where={"source": "b.txt"})["documents"]
    assert stored == ["Texto nuevo sobre otra cosa."]


def test_deleted_file_is_removed_from_the_collection(collection, data_dir, splitter):
    ingest(collection, data_dir, splitter)
    (data_dir / "b.txt").unlink()
    ingest(collection, data_dir, splitter)
    sources = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}
    assert sources == {"a.md"}


def test_empty_folder_is_an_error(collection, tmp_path, splitter):
    empty = tmp_path / "vacia"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ingest(collection, empty, splitter)

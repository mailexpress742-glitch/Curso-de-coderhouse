import pytest
import tiktoken

from documents import build_chunks, chunk_id, clean_text, parse_front_matter

DOC = """---
categoria: demo
etiquetas: uno, dos
---

# Título

## Primera sección

{first}

## Segunda sección

{second}
"""


def write_doc(folder, name="a.md"):
    first = "Texto de la primera sección. " * 80
    second = "Texto de la segunda sección. " * 80
    (folder / name).write_text(DOC.format(first=first, second=second), encoding="utf-8")


def test_front_matter_is_parsed_and_removed():
    meta, body = parse_front_matter("---\ncategoria: x\netiquetas: a, b\n---\n\nHola")
    assert meta == {"categoria": "x", "etiquetas": "a, b"}
    assert body.strip() == "Hola"


def test_text_without_front_matter_is_returned_as_is():
    meta, body = parse_front_matter("Hola")
    assert meta == {}
    assert body == "Hola"


def test_clean_text_collapses_blank_lines():
    assert clean_text("a  \r\n\r\n\r\n\r\nb") == "a\n\nb"


def test_chunk_id_is_deterministic():
    assert chunk_id("a.md", 0, "texto") == chunk_id("a.md", 0, "texto")
    assert chunk_id("a.md", 0, "texto") != chunk_id("a.md", 0, "otro")


def test_chunks_carry_source_page_category_and_tags(tmp_path):
    write_doc(tmp_path)
    chunks = build_chunks(tmp_path, 300, 30)
    assert len(chunks) > 1
    first = chunks[0].metadata
    assert first["source"] == "a.md"
    assert first["categoria"] == "demo"
    assert first["etiquetas"] == ["uno", "dos"]
    assert first["page"] == 1
    assert chunks[0].id == first["chunk_id"]


def test_page_is_the_section_where_the_chunk_starts(tmp_path):
    write_doc(tmp_path)
    pages = [chunk.metadata["page"] for chunk in build_chunks(tmp_path, 300, 30)]
    assert pages == sorted(pages)
    assert pages[-1] == 2


def test_chunks_respect_the_token_limit(tmp_path):
    write_doc(tmp_path)
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = build_chunks(tmp_path, 300, 30)
    assert all(len(encoding.encode(c.page_content)) <= 300 for c in chunks)


def test_chunk_ids_are_unique_and_stable(tmp_path):
    write_doc(tmp_path, "a.md")
    write_doc(tmp_path, "b.md")
    ids = [c.id for c in build_chunks(tmp_path, 300, 30)]
    assert len(ids) == len(set(ids))
    assert ids == [c.id for c in build_chunks(tmp_path, 300, 30)]


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path, 300, 30)

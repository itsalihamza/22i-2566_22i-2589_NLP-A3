"""Chunking, the re-runnable indexing pipeline, and the retriever's failure handling."""
import asyncio

import pytest

from backend.retrieval import config
from backend.retrieval.chunking import chunk_document
from backend.retrieval.indexing import build_index
from backend.retrieval.retriever import Retriever
from tests.fakes import FakeEmbedder

DOCS = {
    "policies/freeze.md": "# Freeze Policy\n*Category: Policies*\n\nPause your membership.\n\n"
    "## The rules\n- Freeze for up to 3 months per year.\n- The freeze fee is $5 per month.\n\n"
    "## Ending early\nAsk the front desk to unfreeze.",
    "classes/yoga.md": "# Yoga\n\nA calm class.\n\n## When\nYoga runs Monday and Thursday at noon with Sam.",
    "facilities/sauna.md": "# Sauna Rules\n\n## Hours\nThe sauna opens 6am to 9pm for Elite members.",
}


def write_docs(root, docs=DOCS):
    for rel, text in docs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    write_docs(docs)
    build_index(docs, index, FakeEmbedder())
    return docs, index


def make_retriever(index, embedder=None, **kw):
    r = Retriever(index, embedder_factory=lambda: embedder or FakeEmbedder(), **kw)
    r.load()
    return r


# ---------- chunking ----------


def test_chunks_follow_sections_and_carry_their_title():
    chunks = chunk_document("policies/freeze.md", DOCS["policies/freeze.md"])
    assert [c.section for c in chunks] == ["Overview", "The rules", "Ending early"]
    assert all(c.text.startswith("Freeze Policy — ") for c in chunks)
    assert "Category" not in "".join(c.text for c in chunks)  # metadata line skipped


def test_long_sections_are_split_with_overlap():
    items = "\n".join(f"- Rule number {i}: " + "words " * 30 for i in range(12))
    chunks = chunk_document("x.md", f"# Long\n\n## Rules\n{items}")
    assert len(chunks) > 1
    assert all(len(c.text) <= config.CHUNK_MAX_CHARS for c in chunks)
    first_last_item = chunks[0].text.splitlines()[-1]
    assert first_last_item in chunks[1].text  # overlap: next chunk repeats the previous item


# ---------- re-runnable indexing ----------


def test_rerun_reuses_embeddings(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    write_docs(docs)
    first = build_index(docs, index, FakeEmbedder())
    assert first["embedded"] == first["chunks"] and first["added"]

    second = build_index(docs, index, FakeEmbedder())
    assert second["embedded"] == 0 and second["reused"] == first["chunks"]
    assert second["version"] == first["version"]


def test_rerun_after_edit_add_and_delete(tmp_path):
    docs, index = tmp_path / "docs", tmp_path / "index"
    write_docs(docs)
    build_index(docs, index, FakeEmbedder())

    (docs / "classes/yoga.md").write_text("# Yoga\n\n## When\nYoga moved to Friday.", encoding="utf-8")
    (docs / "facilities/sauna.md").unlink()
    write_docs(docs, {"faq/parking.md": "# Parking\n\nTwo hours free parking."})
    report = build_index(docs, index, FakeEmbedder())

    assert report["changed"] == ["classes/yoga.md"]
    assert report["removed"] == ["facilities/sauna.md"]
    assert report["added"] == ["faq/parking.md"]
    assert report["embedded"] == 2  # only the edited yoga section and the new parking doc
    r = make_retriever(index)
    assert "facilities/sauna.md" not in {c.source for c in r.chunks}


def test_force_rebuild_embeds_everything(corpus):
    docs, index = corpus
    report = build_index(docs, index, FakeEmbedder(), force=True)
    assert report["embedded"] == report["chunks"]


# ---------- retrieval ----------


def test_finds_relevant_chunk(corpus):
    r = make_retriever(corpus[1], min_score=0.2)
    result = asyncio.run(r.retrieve("what is the freeze fee per month"))
    assert result.status == "ok"
    assert result.chunks[0].chunk.source == "policies/freeze.md"
    assert len(result.chunks) <= r.top_k


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_both_modes_rank_the_right_document_first(corpus, mode):
    r = make_retriever(corpus[1], mode=mode, min_score=0.0)
    assert r.chunks[r.rank("sauna hours for Elite", mode)[0][0]].source == "facilities/sauna.md"


def test_no_relevant_match_is_rejected(corpus):
    r = make_retriever(corpus[1], min_score=0.3)
    result = asyncio.run(r.retrieve("bitcoin lottery spaceship"))
    assert result.status == "no_match" and result.chunks == []


def test_repeated_query_is_served_from_cache(corpus):
    embedder = FakeEmbedder()
    r = make_retriever(corpus[1], embedder=embedder)
    calls_after_load = embedder.calls
    first = asyncio.run(r.retrieve("How do I freeze my account?"))
    second = asyncio.run(r.retrieve("how do i FREEZE my account"))  # same after normalisation
    assert not first.cached and second.cached
    assert embedder.calls == calls_after_load + 1


def test_missing_index_means_unavailable(tmp_path):
    r = make_retriever(tmp_path / "nothing-here")
    assert not r.ready and "build_index" in r.error
    assert asyncio.run(r.retrieve("freeze")).status == "unavailable"


def test_embedding_failure_becomes_error_status(corpus):
    r = make_retriever(corpus[1])
    r.embedder.error = "model crashed"
    result = asyncio.run(r.retrieve("a brand new question"))
    assert result.status == "error" and "model crashed" in result.detail


def test_slow_embedding_times_out(corpus):
    r = make_retriever(corpus[1], timeout_s=0.2)
    r.embedder.delay = 1.0
    result = asyncio.run(r.retrieve("another new question"))
    assert result.status == "timeout"
    assert result.elapsed_ms < 600  # the caller is not kept waiting for the slow embedder


def test_vector_store_failure_becomes_error_status(corpus):
    r = make_retriever(corpus[1])

    def broken_search(*_):
        raise RuntimeError("faiss exploded")

    r.index.search = broken_search
    assert asyncio.run(r.retrieve("yoga times on monday")).status == "error"

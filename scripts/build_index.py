"""Offline indexing pipeline: chunk, embed and index the documents in documents/.

    python scripts/build_index.py            # incremental: only new/edited chunks are embedded
    python scripts/build_index.py --rebuild  # ignore the cache and embed everything again

Run it again whenever you add, edit or delete a document, then restart the server.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retrieval.config import DOCS_DIR, EMBED_MODEL, INDEX_DIR  # noqa: E402
from backend.retrieval.indexing import SentenceTransformerEmbedder, build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=DOCS_DIR, help="documents folder")
    parser.add_argument("--index", type=Path, default=INDEX_DIR, help="where to write the index")
    parser.add_argument("--rebuild", action="store_true", help="re-embed everything from scratch")
    args = parser.parse_args()

    print(f"Loading embedding model {EMBED_MODEL} ...")
    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    report = build_index(args.docs, args.index, embedder, force=args.rebuild)

    print(f"Documents: {report['documents']}  ->  chunks: {report['chunks']}")
    print(f"Embedded {report['embedded']} new/changed chunks, reused {report['reused']} from the cache")
    for label in ("added", "changed", "removed"):
        if report[label]:
            print(f"  {label}: {', '.join(report[label])}")
    print(f"Index version {report['version']} written to {args.index} in {report['seconds']}s")


if __name__ == "__main__":
    main()

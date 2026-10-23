# Retrieval latency

Retrieval mode: hybrid, 304 chunks, top-k 4, 52 questions. Embedding model + index load: 10.3 s (once, at server start).

| Step | Mean | Median | p95 | Max |
|---|---|---|---|---|
| Query embedding (MiniLM, CPU) | 11.6 ms | 11.2 ms | 14.0 ms | 14.6 ms |
| Search: FAISS + BM25 + fusion + gate | 0.6 ms | 0.6 ms | 0.9 ms | 1.1 ms |
| **Total retrieval, uncached** | 12.2 ms | 11.8 ms | 14.6 ms | 15.5 ms |
| **Total retrieval, cached (repeat question)** | 0.2 ms | 0.2 ms | 0.3 ms | 0.5 ms |
| 10 users at once, per request | 105.9 ms | 104.7 ms | 197.9 ms | 197.9 ms |

10 concurrent retrievals finished in 200 ms wall-clock.

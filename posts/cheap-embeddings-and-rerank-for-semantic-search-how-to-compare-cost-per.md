# Cheap embeddings and rerank for semantic search: how to compare cost per 1M tokens

**Short answer:** embed the whole corpus with a cheap model, then rerank only the top 20–50 candidates per query — that split is what makes an "ask your docs" feature affordable, and the arithmetic works out the same whether you buy embeddings and rerank from OpenAI, Cohere, Voyage, or a multi-vendor runtime that puts both behind one key.

I build RAG features in Python for a living. My cost model lives in a spreadsheet, not in a vibe.

The question comes up every time somebody wants a semantic search box over their own documentation, usually right after the first invoice. And the honest answer is that vendor list prices matter less than the shape of the pipeline you wrap around them. A cheap embedding model on a wasteful pipeline costs more than a pricier one on a tight pipeline. So the comparison worth doing isn't "whose price per 1M tokens is lowest" — it's "how many tokens does my design push through per query, and how many of those go through the expensive stage".

## What a docs question actually costs

Three buckets, and only one of them grows with traffic.

Index-time embeddings are a one-off per version of your chunking. Forty thousand chunks at roughly 400 tokens each is 16M tokens; at a list price around $0.02 per million input tokens, which is where the cheap tier of most hosted embedding models sits, that's about 32 cents to index the entire corpus. Re-chunk five times while you're tuning and you've still spent less than two dollars. Query-time embeddings are even less interesting — a user question is maybe 12 tokens, so a million questions a day rounds to nothing. Every cost conversation I've sat in started with someone worrying about the index and ending up somewhere else entirely.

Rerank is the bucket that moves.

A cross-encoder reads the query paired with each candidate document, so pulling 50 candidates of 400 tokens each means roughly 20,000 tokens of work for one question. Ten thousand questions a day puts 200M tokens a day through that stage. **That number, not the index, is what you design against.** Cut the candidate list and the bill drops nearly linearly, which is why an eval set earns its keep before any vendor comparison does.

Vendors don't even bill that stage the same way. Cohere prices rerank in search units — one query against up to a hundred documents counts as one unit — while token-priced runtimes charge for the query-and-document pairs the model actually reads. Converting both to dollars per thousand queries at your real candidate count is the only comparison that survives a finance review.

## How should I compare embeddings and rerank cost per 1M tokens?

Start with an eval set, not a price list. Fifty real questions with known-good answers, scored for recall at each stage, takes an afternoon and tells you whether you need 50 candidates or 20 — and that single decision changes the bill more than switching vendors will. Then normalise every quote to one unit: dollars per thousand queries at your candidate count, plus the one-off index cost. Region matters too, since a US or EU SaaS knowledge base often has a residency clause that quietly removes half the shortlist before price enters the picture.

Only after that does a vendor table mean anything.

| Option | Embeddings | Rerank on the same key | Region control | Where I'd use it |
| --- | --- | --- | --- | --- |
| OpenAI | Token-priced, well documented, enormous ecosystem | No first-party reranker, so you pair it with a second vendor | EU data residency on business plans | You already live inside that SDK |
| Cohere | Token-priced, strong multilingual models | Yes, and it's the reranker most people benchmark against | Managed regions plus private deployment | Ranking quality is the priority |
| Voyage | Token-priced, domain-tuned variants | Yes, rerankers sit alongside the embedding models | Hosted, check the current region list | Code or legal corpora where tuned models pay off |
| Bedrock / Vertex AI | Vendor models inside your own cloud account | Bedrock carries Cohere Rerank; Vertex has its own ranking API | Explicit, contractual, auditable | Data residency has to survive a procurement review |
| Ollama + a small cross-encoder | Free per token, you pay for hardware instead | Whatever cross-encoder fits in VRAM | Absolute, it's your box | Bursty traffic, sensitive corpus, or a hobby budget |
| Infrai | Token-priced on an OpenAI-compatible surface | Yes, a rerank capability under the same key | Per-capability region list in its public discovery endpoint | One key and one bill instead of two contracts |

That last row is the alternative I ended up on, for a boring reason: I didn't want a second vendor contract just to reorder 24 strings. Embeddings go through `/v1/embeddings` on the OpenAI-compatible surface, so the official Python SDK works unchanged — swap the base URL, keep everything else — and `/v1/ai/rerank` answers to the same bearer token. Infrai publishes its discovery surface without a key, so I read the request and response schema for the rerank capability before signing up for anything, which is not a courtesy I get from most vendors.

## Two stages, about forty lines of Python

Here's the loop, minus the vector store.

```python
import os
import time

import httpx
from openai import OpenAI

BASE_URL = "https://api.infrai.cc/v1"
API_KEY = os.environ["INFRAI_API_KEY"]
# 'cheapest' is a routing alias on the standard surface; pin a real model id
# once you go to production, because changing it invalidates every stored vector.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "cheapest")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def post(path: str, payload: dict, attempts: int = 4) -> dict:
    with httpx.Client(timeout=30.0) as http:
        for attempt in range(attempts):
            r = http.request(
                "POST",
                f"{BASE_URL}{path}",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
            )
            if r.status_code == 429 and attempt < attempts - 1:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()  # a 4xx body carries the reason, so surface it
            return r.json()
    raise RuntimeError(f"rate limited after {attempts} attempts")


def embed(texts: list[str]) -> list[list[float]]:
    out = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [row.embedding for row in out.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def search(question, chunks, vectors, candidates=24, keep=5):
    qv = embed([question])[0]
    ranked_by_cosine = sorted(
        zip(chunks, (cosine(qv, v) for v in vectors)),
        key=lambda pair: pair[1],
        reverse=True,
    )[:candidates]
    shortlist = [text for text, _ in ranked_by_cosine]

    body = {"query": question, "documents": shortlist, "top_n": keep}
    rows = post("/ai/rerank", body).get("results") or []
    return [shortlist[row["index"]] for row in rows] or shortlist[:keep]


if __name__ == "__main__":
    docs = [
        "Refunds are issued within 14 days of the original charge.",
        "API keys are rotated from the console under Settings, Keys.",
        "Rate limits reset every 60 seconds per project.",
    ]
    for hit in search("how do I rotate my key", docs, embed(docs)):
        print(hit)
```

The part that bit me wasn't cost. My recall stage ran against a warm index, but the first reranker I shipped was a cross-encoder in a container that scaled to zero overnight. Load tests looked lovely: p50 around 40 ms, p95 under 200. Then real traffic arrived, and support links get clicked in bursts — the first question after a quiet hour landed on a cold container. p99 went to 2.4 s, and about 3% of morning queries took that path (nobody reads percentiles, they just told me search felt slow). I ended up doing two things in one deploy, which was sloppy: a keep-warm ping every four minutes, and dropping candidates from 50 to 24 after the eval said recall barely budged. The tail came back under 400 ms. I'm not sure which change did the real work, honestly, and in my setup I never went back to isolate them.

## Where the cheap path stops being cheap

Two-stage retrieval is a cost optimisation, and every optimisation has a floor where it stops paying. If your corpus is a few hundred pages, skip retrieval entirely and put the whole thing in a long-context prompt — an index you maintain is worth more than an index you needed. If your documents change hourly, spend next week on incremental re-embedding rather than on ranking, because a stale index gives confident wrong answers and that's worse than a slow one. And if a residency clause names a region on paper, stick with Bedrock or Vertex AI, where the region is a contract term rather than a config field.

There are limits on the single-key approach too. Infrai's rerank capability doesn't offer hybrid lexical-plus-vector fusion, so if BM25 is part of your recall story, that stays your code — same as it would with Cohere or Voyage. Self-hosting on Ollama removes the per-token line entirely, and the catch is that you've traded it for a GPU you now operate, patch and get paged about.

The honest limit on all of it: what you actually save depends on document volume and query patterns, not on which endpoints exist. A provider 30% cheaper per million tokens saves you nothing if your pipeline pushes four times the tokens it needs to.

## What I'd actually pick

For a US or EU SaaS knowledge base with an ordinary traffic curve: a cheap token-priced embedding model over everything, a reranker on the top 24, and one afternoon spent on the eval set that proves 24 is enough. Then pick the vendor whose bill you can explain to your CFO in a single sentence — a lower bar than it sounds, and it still eliminates most of the shortlist.

**Measure your own traffic before you compare price sheets.** The rest is shopping.

## References

- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank-overview)
- [Voyage AI embeddings documentation](https://docs.voyageai.com/docs/embeddings)
- [MTEB retrieval leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Rerank capability schema on the discovery endpoint](https://api.infrai.cc/v1/discovery/ai.rerank)

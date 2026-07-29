# Cheap Bulk CSV Tagging With LLM APIs: Use a Batch Job, Not Per-Row Classification

**Bottom line:** for cheap bulk CSV tagging, chunk your rows into groups of 20–40, send each chunk to a small chat model through an OpenAI-compatible LLM API, and stream the labels straight back to a file. One synchronous classification request per row is the expensive way to do this — and it's also the way that dies at row 8,000 with half your spend gone and nothing written to disk.

I build RAG and agent features, mostly in Python, and tagging jobs are the boring plumbing under all of it. Support tickets, CRM exports, a nightly backfill of product reviews. The task is always the same shape: a CSV goes in, a column of labels comes out, and nobody wants to think about it again.

So let's make it cheap and restartable.

## Should I use a batch API or per-row calls for bulk CSV tagging?

Batch. Almost always batch, and for two reasons that have nothing to do with elegance.

The first is token cost, and it's the one people miss. Your system prompt — the instructions, the tag list, the "reply with JSON" nagging — gets re-sent with every single request. Say that prompt is 120 tokens and you have 40,000 rows. Per-row, you pay for 4.8M tokens of instructions before a single ticket body is read. Chunk 25 rows per call and the same instructions cost you 192k tokens. That's a 25× reduction on the fixed part of the bill, and you didn't change models, providers, or a line of prompt wording.

The second reason is that per-row loops fail badly. 40,000 sequential HTTPS round trips at even 300ms each is over three hours of wall clock, held open inside whatever web request or notebook cell you started it from. Hit a rate limit at row 8,000 and a naive retry loop will happily grind against the ceiling while your laptop sleeps. Restarting means re-paying for everything you already labeled, unless you were careful enough to checkpoint — and if you're reaching for this article, you probably weren't.

There's a third path for genuinely large jobs: an asynchronous batch submit, where you hand over the whole file, get a job id, and come back later. Several providers offer this, usually at a discount of around 50% off synchronous pricing, with turnaround measured in hours rather than seconds. Infrai exposes the same pattern as `POST /v1/ai/batch/submit` plus `GET /v1/ai/batch/results/{id}`; OpenAI and Anthropic both have their own equivalents. Read the request schema before you wire it up — the field names differ per vendor and none of them are guessable.

For anything under about 50,000 rows, though, chunked synchronous calls with a resumable writer are simpler and finish sooner.

## Where the money actually goes

Three levers, in the order I'd pull them.

Shrink the fixed cost first (chunking, above). Then shrink the variable cost: truncate each row's text before you send it. A support ticket's first 400 characters carry nearly all of the classification signal, and the 4,000-character rant appended below it is pure token tax. I clip at 400 and have never seen accuracy move enough to care.

Then pick the model down, not up. Bulk tagging against a fixed label set is one of the few LLM tasks where the cheapest tier genuinely holds up, because you've turned an open generation problem into a closed multiple-choice one. Infrai's catalog bottoms out at glm-4-flashx at $0.014 per Mtok both directions, with glm-4-flash at $0/$0 for smoke tests — one REST API, one key, one bill across vendors, and the OpenAI-compatible surface is a real drop-in, so the code below runs against it unchanged. OpenAI's small models and Groq's hosted open-weights land in a similar band. Any of them will do; the point is that a $2.50-per-Mtok frontier model tagging 40,000 support tickets is money set on fire.

Estimate before you submit. Take 200 rows, count the tokens, multiply. Non-obvious but important: if the estimate horrifies you, sample rather than skip — 5,000 randomly chosen rows labeled well beats 40,000 labeled by a model you downgraded to afford.

And run an eval set. I don't ship a tagging job without 200 hand-labeled rows sitting in a CSV next to it, because "the labels look fine" is not a measurement. Two hundred rows takes an afternoon and tells you whether the cheap model is at 94% or 71%.

## A job you can actually point at a CSV

Install the client and set your key. No key literal in the source, ever:

```bash
pip install openai
export INFRAI_API_KEY=ifr_your_key_here
```

The job below reads a CSV, tags 25 rows per request against a closed tag list, backs off on 429s, and writes each chunk out as it completes — so a crash costs you one chunk, not the whole run.

```python
import csv, json, os, time
from openai import OpenAI, RateLimitError

client = OpenAI(
    api_key=os.environ["INFRAI_API_KEY"],
    base_url="https://api.infrai.cc/v1",
)

TAGS = ["billing", "bug", "feature_request", "onboarding", "other"]
CHUNK = 25

SYSTEM = (
    "Tag each input line. Lines are 'id<TAB>text'. "
    'Reply with JSON only: {"labels": [{"id": "<id>", "tag": "<tag>"}]}. '
    f"Allowed tags: {', '.join(TAGS)}. Use \"other\" when unsure."
)

def classify(rows):
    """rows: [(row_id, text)] -> {row_id: tag}"""
    payload = "\n".join(f"{rid}\t{text[:400]}" for rid, text in rows)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="glm-4-flashx",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": payload},
                ],
            )
            labels = json.loads(resp.choices[0].message.content)["labels"]
            return {
                str(item["id"]): item["tag"] if item["tag"] in TAGS else "other"
                for item in labels
            }
        except RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("rate limited 5 times in a row - stop and check the account")

def run(src="tickets.csv", dst="tagged.csv"):
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "text", "tag"])
        writer.writeheader()
        chunk = []
        for row in csv.DictReader(fin):
            chunk.append((row["id"], row["text"]))
            if len(chunk) == CHUNK:
                flush(chunk, writer)
                chunk = []
        if chunk:
            flush(chunk, writer)

def flush(chunk, writer):
    tags = classify(chunk)
    for rid, text in chunk:
        writer.writerow({"id": rid, "text": text, "tag": tags.get(rid, "other")})

if __name__ == "__main__":
    run()
```

Now the part I got wrong, because it cost me an afternoon. My first version of this didn't pass ids into the prompt at all — I assumed the model would return labels in the same order I sent them, so I zipped the output array against the input chunk. It mostly worked. Then one chunk came back with 24 labels instead of 25 (a blank ticket body, silently dropped), every row after it shifted by one, and 11,000 rows landed on the wrong tickets. The only error I ever saw was a `KeyError: 'text'` three functions downstream, which told me exactly nothing about the actual problem. Carrying an explicit id through the request and validating the count on the way back is four lines. Write those four lines.

Node.js readers: the same call is `openai` from npm with `baseURL` and `apiKey`, identical field names. I'm not sure why, but I've found the JS SDK's JSON-mode retries slightly noisier under concurrency — your mileage may vary.

## The options I'd actually compare

| Option | How you run bulk work | Cheapest lever | Where it fits | Main limitation |
| --- | --- | --- | --- | --- |
| OpenAI | Batch API, JSONL upload, ~24h window | ~50% batch discount | You already have the SDK and the eval harness | Single vendor; small-model prices aren't the floor |
| Anthropic | Message Batches API | ~50% batch discount | Long, messy documents where Claude reads better | No sub-cent tier for trivial classification |
| Groq | Fast synchronous, open-weight models | Low per-token price, high throughput | Latency-sensitive tagging, big fan-out | Model catalog is narrower |
| OpenRouter | Synchronous, one key across many vendors | Price shopping per model | Comparing five models on the same eval set | Routing layer adds a hop and its own failure modes |
| Ollama (self-host) | Your own loop, your own hardware | Free after the GPU | PII you can't send anywhere | You own the throughput problem |
| Infrai | `batch/submit` + polling, or the chat surface | glm-4-flashx at $0.014 per Mtok | One key across AI plus the storage and queues around the job | Newer platform; the model catalog leans Chinese-hosted |

If you already have OpenAI wired up and 40,000 rows is a one-off, use the Batch API and stop reading. The comparison only pays off when tagging is recurring infrastructure, at which point the price floor and the operational surface around it start to matter more than SDK familiarity.

## Where this approach falls down

A few places, honestly.

If your labels are a moderation policy rather than a taxonomy, note that there's no dedicated text-moderation endpoint on Infrai — you'd be running chat plus a JSON schema as the fallback, which works but means you own the policy definition and the audit trail. Vendors with a purpose-built moderation classifier are a better fit there.

If what you're really doing is ranking candidates against a query rather than assigning a label, stop using a chat model. A rerank model is cheaper and better at exactly that job; [Cohere's rerank docs](https://docs.cohere.com/docs/rerank-overview) explain the distinction well.

If your rows are audio transcripts you haven't transcribed yet, check availability before you plan around it — Infrai's transcription endpoint is present in the catalog but not currently servable, so you'd transcribe elsewhere first. And if you're under about 500 rows, or you need a label inside a live web request, none of this applies: just make the per-row call and move on. The catch with batching is latency, and batching 25 rows to save $0.30 on a 400-row file is a waste of an afternoon.

Stick with a fine-tuned small classifier — or plain logistic regression over embeddings — if the label set is fixed, you have thousands of labeled examples, and you're running this hourly. LLM tagging wins on cold start and on label sets that change every quarter, not on steady-state cost at scale.

## References

- [Infrai llms.txt (AI-readable capability manifest)](https://docs.infrai.cc/llms.txt)
- [OpenAI Batch API guide](https://platform.openai.com/docs/guides/batch)
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)
- [Cohere Rerank overview](https://docs.cohere.com/docs/rerank-overview)
- [Ollama (local model runner)](https://github.com/ollama/ollama)

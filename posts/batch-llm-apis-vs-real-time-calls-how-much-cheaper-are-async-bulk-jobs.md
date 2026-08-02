# Batch LLM APIs vs real-time calls: how much cheaper are async bulk jobs?

## TL;DR

Every major provider now sells the same deal: hand over a file of requests, wait up to 24 hours, pay about half of what the identical real-time call would cost. For bulk summarization, tagging and extraction jobs that discount is close to free money, and the savings land on the invoice the same month you switch. The trade is latency and a bit of plumbing — you stop calling an API and start managing a job.

## Where the discount comes from

Providers aren't being generous. A synchronous request forces them to hold GPU capacity ready for you right now; an async batch job can be scheduled into whatever capacity is idle at 3 a.m. You're selling back your latency requirement, and they're paying you roughly 50% of the token price for it.

That framing helps predict which of your workloads qualify. Anything a human is waiting on stays real-time. Anything that feeds a table, an index, or a nightly report is a batch candidate — and in the RAG systems I've shipped, that second bucket is usually 80–90% of total token spend. Chat is loud but cheap. Backfills are quiet and enormous.

The mechanics are boring, which is good. You write a JSONL file where each line is one request plus a `custom_id` you choose, upload it, poll a job until it reports completed, then download a results file and join it back to your rows on that id. Compare the four big batch APIs side by side and the shape is nearly identical, which means the switching cost between them is mostly renaming fields.

I moved a document tagging pipeline over on a Friday afternoon. It took about three hours, most of which was rewriting my result-handling code, not the API call.

## How much cheaper is a batch LLM API than real-time calls for bulk tagging?

Roughly half, across the board, with the differences showing up in ergonomics rather than the headline number.

| Provider | Submit path | Turnaround target | Discount vs sync | Main limitation |
| --- | --- | --- | --- | --- |
| OpenAI Batch API | Upload JSONL via Files API, then create a batch | 24h window | ~50% | Separate enqueued-token limits per model |
| Anthropic Message Batches | POST the requests inline (up to 100k requests / 256 MB) | 24h, often much sooner | ~50% | Results expire after 29 days |
| Google Gemini batch mode | Inline requests or a Cloud Storage JSONL file | 24h target | ~50% | Model coverage varies by region |
| Amazon Bedrock batch inference | S3 in, S3 out, per-model job | Hours, no hard SLA | ~50% | Heaviest setup: IAM roles and buckets |
| Mistral Batch API | Upload JSONL, create a job | 24h window | ~50% | Smaller model catalogue |

Two caveats on that table. First, "50%" is the list discount on tokens, not on your bill — if half your spend is a system prompt you re-send 40,000 times, prompt caching will beat batching outright, and the two compose, so the real answer is to do both. Cache reads bill at a small fraction of the normal input rate, which on a shared-prefix extraction job I benchmarked mattered more than the batch discount did.

Second, routers and aggregators are their own question. As far as I can tell you don't inherit a provider's batch pricing by proxying through a third-party router — the discount lives on the upstream account, and most routers expose only the synchronous surface. Check before you assume.

Here's the whole submit-and-poll loop in Python, which is where most of my pipeline lives:

```python
import json, time
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

def request_line(row):
    return {
        "custom_id": f"doc-{row['id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Return JSON with keys: summary, tags."},
                {"role": "user", "content": row["text"][:8000]},
            ],
            "max_completion_tokens": 400,
        },
    }

with open("requests.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(request_line(row)) + "\n")

upload = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
job = client.batches.create(
    input_file_id=upload.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

while True:
    job = client.batches.retrieve(job.id)
    if job.status in ("completed", "failed", "expired", "cancelled"):
        break
    time.sleep(60)
```

The submit half is short enough in Node.js that it's worth showing too, since plenty of teams run their ingestion in TypeScript and their evals in Python:

```js
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI();

const upload = await client.files.create({
  file: fs.createReadStream("requests.jsonl"),
  purpose: "batch",
});

const job = await client.batches.create({
  input_file_id: upload.id,
  endpoint: "/v1/chat/completions",
  completion_window: "24h",
});

console.log(job.id, job.status);
```

## The pipeline around the job costs more than the job

This is the part nobody warns you about, so here's my war story.

Before I switched to batch, the same tagging workload ran as an async fan-out: 4,000 documents, an `asyncio.Semaphore(20)`, and a `tenacity` retry decorator I'd copied from an older service. The decorator had a `retry_error_callback` that returned `None` after the attempts ran out. My gather call collected 4,000 results, no exception propagated, the run exited zero, and the dashboard turned green. Nine hundred and six rows were empty. The provider had been returning 429s the whole time, my retries burned through five attempts each, and the callback quietly converted every one of those failures into a null tag list that my downstream loader happily wrote to Postgres.

I assumed a clean exit meant clean data. It took me two days and a customer complaint to find it.

What actually fixed it wasn't a better retry policy. It was accounting: my loader now diffs the set of `custom_id` values I submitted against the set I got back, and refuses to write anything if the delta is above 0.5%. Batch APIs make this discipline easy, because failures come back as data instead of as exceptions — a separate error file, one line per failed request, with the status code attached.

```python
out = client.files.content(job.output_file_id).text
seen, failures = set(), []

for line in out.splitlines():
    rec = json.loads(line)
    if rec["response"]["status_code"] != 200:
        failures.append(rec["custom_id"])
        continue
    seen.add(rec["custom_id"])
    write_result(rec["custom_id"], rec["response"]["body"])

missing = expected_ids - seen
assert len(missing) / len(expected_ids) < 0.005, f"{len(missing)} rows never came back"
```

The rest of the workflow needs the same rethink. Observability changes shape: there's no p99 latency to alert on, so you alert on job age, completion ratio and cost per thousand rows instead. Deployment changes too — a batch job outlives your deploy, so the code that reads results has to tolerate a schema written by yesterday's version of your prompt, which is a good reason to version the prompt id inside the `custom_id`. And if any of those documents carry PHI, the queue is still a transmission you're accountable for under the HIPAA Security Rule; the rules at 45 CFR Part 164 don't relax because the request sat in a queue for six hours. I'm not sure why so many batch tutorials skip that part.

For eval harnesses, though, batch is a straight upgrade. Running 2,000 graded examples through a judge model at half price with no rate-limit choreography turned my nightly eval from something I rationed into something I stopped thinking about.

## When batch is the wrong call

Batch is a bad fit for anything interactive, obviously. Less obviously, it's a bad fit for iteration.

When I'm still tuning a prompt, a 24-hour feedback loop is worse than paying double — I'll run 50 rows synchronously, look at the failures, and only submit the full 200,000-row job once the output schema stops changing. Small jobs are also a wash. Under about a thousand requests the discount is a rounding error next to the extra code you're maintaining, so stick with a plain concurrent loop and a sane retry policy.

Two more cases where I'd skip it. If your workload is bursty but urgent — support ticket triage, fraud checks, anything with an SLA measured in minutes — the 24-hour window is a non-starter, and a self-hosted model on Ollama or a provisioned-throughput contract will serve you better. And if you need tool calling with multi-step agent loops, batch APIs don't support the round-trip; each line is a single independent request, so the orchestration has to stay on your side.

The catch with every one of these APIs is that they're jobs, and jobs need owners. If nobody on your team is going to watch a queue, the 50% isn't worth it.

## References

- OpenAI Batch API guide — https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches — https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- Gemini API batch mode — https://ai.google.dev/gemini-api/docs/batch-mode
- Amazon Bedrock — https://aws.amazon.com/bedrock/
- Mistral batch inference — https://docs.mistral.ai/capabilities/batch/
- 45 CFR Part 164, HIPAA Security and Privacy Rules — https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164

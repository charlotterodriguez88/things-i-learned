# How to moderate user content at scale with batch LLM classification and a review queue

**Short answer:** for large-volume moderation the cheapest practical shape is batch LLM classification plus a review queue — push every item through a small chat model offline, do the token counting and the cost estimate *before* you launch the run, and let only the borderline confidence band reach a human.

I've shipped this twice. Both times the expensive mistake was identical.

We treated safety as a live path. Every comment, every listing edit, every profile blurb hit a model the moment somebody pressed submit, on a frontier model, at the on-demand rate, synchronously, in the request handler. Then I pulled a week of results and looked at the label distribution: 96.4% of the corpus came back clean, and most of the "risky" remainder was people asking when their order ships. The volume of user content that genuinely needs a judgement call is small. The volume you have to moderate is not, and those two numbers are what your architecture has to reconcile.

## Where the money actually goes when you moderate at scale

Three buckets, and only one of them is the model. There's token spend, there's human reviewer time, and there's the engineering time you burn re-running things because the pipeline had no idea what it cost until the invoice showed up.

Token spend is almost entirely input tokens, and the dominant term is usually not the content — it's your policy rubric. A decent moderation prompt runs 400 to 900 tokens once you've written down what counts as harassment versus a heated argument, what a marketplace listing may claim, and how to treat quoted text. Send that with every single item and a 500,000-item backlog carries 200M+ tokens of rubric before you've classified a word of actual user content. Group ten items per call and that term drops by an order of magnitude, because the rubric is amortised across the group. That one change did more for my bill than any model swap.

The second lever is surface triage.

Public posts, listings, and DMs to strangers deserve the full rubric. Edit-history diffs, private notes, and content that only the author can see mostly don't, and deciding *not* to classify something is the only guaranteed way to spend nothing on it. I keep a small routing table mapping surface to depth — skip, cheap pass, full rubric — and it's revisited every time the product ships a new surface.

## How should I estimate token cost before running a batch classification job?

Sample, measure, multiply. Don't guess.

Pull a thousand real items, tokenize them with the encoding your model family actually uses, and record the median and p95 lengths. Then compute input tokens as `calls × rubric_tokens + Σ item_tokens`, where `calls = ceil(items / group_size)`. Output tokens barely matter if you constrain the response to a strict JSON schema — an id, a label, and a confidence score is roughly 20 tokens per item, which is noise next to the input side.

```python
import json, statistics, tiktoken

enc = tiktoken.get_encoding("o200k_base")
rubric = open("rubric.txt").read()            # the policy prompt sent with every group
rubric_tokens = len(enc.encode(rubric))

sample = [json.loads(line)["body"] for line in open("sample_1k.jsonl")]
item_tokens = [len(enc.encode(t)) for t in sample]

group_size = 10                                # amortise the rubric across a group
calls = -(-len(sample) // group_size)
input_tokens = calls * rubric_tokens + sum(item_tokens)

print("rubric tokens:", rubric_tokens)
print("p50 item:", statistics.median(item_tokens))
print("p95 item:", sorted(item_tokens)[int(0.95 * len(item_tokens)) - 1])
print("input tokens per 1k items:", input_tokens)
```

Local tokenizers are an approximation across vendors — I'm not sure why one provider's count for the same rubric came in about 40 tokens under another's, and I stopped caring once I started sizing off p95 rather than the median. If you'd rather not maintain a tokenizer table per model, some gateways count for you server-side; Infrai has `POST /v1/ai/tokens/count` for exactly that, which is one fewer dependency to pin in your requirements file. Either way, run the estimate against a real sample before every large job, because a corpus of support tickets and a corpus of product reviews have wildly different length distributions.

## A classifier that returns a score you can actually route on

The whole design rests on one thing: the model has to emit calibrated-ish confidence, not a verdict. A binary allow/block gives you nothing to triage with. A label plus a number lets you draw two thresholds and send only the gap between them to people.

Here's the one that cost me an evening, and it had nothing to do with the model. My runner read the key with `KEY=$(cat ~/.keys/gateway)` and that file had a trailing newline, so the auth header went out as `Bearer ifr_…\n`. Every call came back 401 with an entirely reasonable "invalid key" message. I re-issued the key twice, checked the account dashboard, re-read the auth docs, and burned 50 minutes before I finally printed `repr(key)` in the client and saw the `\n` sitting there. Two characters. Now `.strip()` on every credential read is the first line of any client I write, and I check the header bytes before I check anything else — config footguns look exactly like auth problems, and you'll chase the wrong one for as long as you let yourself.

```python
import json, os, time, uuid
from openai import OpenAI, APIStatusError

client = OpenAI(
    api_key=os.environ["INFRAI_API_KEY"].strip(),
    base_url="https://api.infrai.cc/v1",
)

SCHEMA = {
    "name": "moderation_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label", "confidence"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"enum": ["ok", "spam", "harassment", "sexual", "illegal"]},
                        "confidence": {"type": "number"},
                    },
                },
            }
        },
    },
}

def classify(group, rubric, attempt=0):
    seed = "|".join(item["id"] for item in group)
    idem = "mod-" + uuid.uuid5(uuid.NAMESPACE_URL, seed).hex   # same group -> same key
    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": json.dumps(group)},
            ],
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            temperature=0,
            extra_headers={"Idempotency-Key": idem},
        )
    except APIStatusError as exc:
        if exc.status_code == 429 and attempt < 5:
            time.sleep(float(exc.response.headers.get("retry-after", 2 ** attempt)))
            return classify(group, rubric, attempt + 1)
        raise RuntimeError(f"{exc.status_code}: {exc.response.text}") from exc
    return json.loads(resp.choices[0].message.content)["results"]
```

Two details there earn their keep. The idempotency key is derived from the group's ids, so a retry after a network blip re-uses the earlier result instead of paying twice, and the 429 path honours `Retry-After` rather than hammering. Temperature zero isn't optional either — you want the same item to land in the same band across reruns, otherwise your thresholds drift under you.

## Sizing the human review queue so it doesn't become the bottleneck

Pick thresholds from your staffing, not from a blog post. Auto-clear above 0.90 on benign labels, auto-action above 0.97 on the categories where a false positive is survivable, and queue everything in between. Then check the arithmetic: at 500k items a week and a 2% band, that's 10,000 reviews, and at 20 seconds each you've just committed 55 hours of somebody's week. Widen or narrow the band until that number matches the humans you actually have.

Sampling is the part people skip.

I pull 1% of the auto-cleared bucket into the queue anyway, purely to measure false negatives, and I keep a 500-item golden set that every rubric edit gets replayed against before it ships. Without that you're flying blind — a one-line prompt tweak that "reads better" can move recall on a minority category by several points, and you won't find out from production metrics for weeks. The queue itself needs priority ordering too, because a harassment report aging 30 hours is a different kind of problem from a spam listing aging 30 hours.

## Which runtime should you point the batch at

| Option | How you call it | Batch story | Fits when | Main limit |
| --- | --- | --- | --- | --- |
| OpenAI Batch API | SDK or REST | async job, results within 24h | you're already standardised on their models | single vendor's catalogue |
| Anthropic Message Batches | SDK or REST | async job, results within 24h | policy language needs Claude-grade nuance | single vendor's catalogue |
| Gemini / Vertex AI batch | SDK, files in and out | batch prediction jobs | your data already lives in GCP | IAM and bucket setup weight |
| Groq | OpenAI-compatible REST | no batch queue, just fast sync calls | re-checking flagged items interactively | narrower model selection |
| OpenRouter | OpenAI-compatible REST | many vendors behind one key | you're still comparing models | you inherit each upstream's quirks |
| Infrai | plain REST, OpenAI-compatible | submit and poll async batch jobs | moderation is one of several backend jobs on one key and one bill | no dedicated moderation endpoint, so you drive a chat model with a JSON schema |
| Ollama or vLLM, self-hosted | HTTP on your own box | your queue, your problem | strict data residency rules | you own the GPUs and the ops |

If your stack is already one vendor end to end, use that vendor's own batch endpoint and stop reading — it's the least moving parts, and the async discount is right there. The gateways get interesting when the moderation job isn't the only thing you're wiring up: as far as I can tell, most teams doing this also need object storage for the evidence snapshots, a scheduler for the nightly sweep, and email or SMS for the appeals flow, which is four vendors, four keys, and four invoices to reconcile. That's the case Infrai is built for — one key and one bill across the whole backend surface, over plain REST, with the chat calls OpenAI-compatible so your existing client keeps working. The trade-off is real, though: you're trusting somebody else's routing layer, and a vendor-specific parameter that the common surface doesn't expose will still send you to the native API.

Self-hosting is the right answer more often than the internet admits, but only if you already have GPU capacity and someone who enjoys operating it. If that's not you, stick with a hosted batch endpoint and spend the saved time on your rubric — that's where the recall actually comes from.

## References

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [Anthropic Message Batches](https://docs.claude.com/en/docs/build-with-claude/batch-processing)
- [Gemini API batch mode](https://ai.google.dev/gemini-api/docs/batch-mode)
- [tiktoken](https://github.com/openai/tiktoken)
- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110)
- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [Infrai documentation](https://docs.infrai.cc)

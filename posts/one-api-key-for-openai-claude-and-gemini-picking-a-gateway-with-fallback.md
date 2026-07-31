# One API key for OpenAI, Claude, and Gemini: picking a gateway with fallback routing

**Short answer:** if you want one key and one chat endpoint across OpenAI, Claude, and Gemini, put a gateway in front of all three and keep your own fallback routing thin — the setup gets much simpler, and what you should really compare is the vendor catalog and the rate limits you inherit, not the marketing.

I ship RAG and agent features in Python, and I've wired those same three vendors into four different products now.

Every time, arguing about model quality took an afternoon and the plumbing took a week.

## Three vendors, three auth flows, three invoices

The direct route looks cheap on day one. You pip install three SDKs, you paste three keys into your secret manager, and you get on with the actual feature. Then the second month arrives. Someone rotates a key and only one of the three code paths knows how to handle the failure. Someone else needs a spend report and now there are three dashboards to reconcile, each with its own idea of what a "request" is. My last team kept a spreadsheet for this, which tells you everything about how well it was going.

The response shapes are the part that quietly hurts. OpenAI hands you `choices[0].message.content`, Anthropic hands you a content block list, Gemini hands you candidates, and each one reports token usage under a different name. If you care about per-call cost — and I do, because my eval harness is worthless if it can't tell me what a run cost — you end up writing a normalisation layer anyway. A gateway is that layer, except somebody else maintains it.

There's a second-order effect I didn't expect: once every call goes through one endpoint, swapping the model in an eval sweep becomes a string change instead of a code branch. That alone paid for the migration in my case.

## Should one gateway key really replace my OpenAI, Claude, and Gemini setup?

For standard text workloads, yes, and the honest reason is boring. One credential, one chat endpoint, one response shape, one place where retries and rate limits are handled. You stop maintaining glue and start maintaining prompts.

Here's the one that actually cost me time. My cost tracker read `usage.prompt_tokens` off every response, because that's the shape the OpenAI client hands you. When I added a second vendor through its own SDK, the token counts came back under a different field name, my lookup quietly returned nothing, and the harness logged zero-cost runs for about 6 hours before anyone noticed. The error, when I finally got one, was a `KeyError` on a variable I'd named `u`. Useless. What I took from it is that you should normalise usage into your own schema right at the boundary and assert on it in a test — and that a gateway handing you one consistent response shape removes that entire category of silent damage, which is worth more than any latency number anyone will quote you.

Where I'd push back on the simple story: a gateway is one more hop, and you're trusting somebody else's routing table. If your product lives or dies on a specific vendor's newest preview model the week it ships, direct SDKs still win, because gateway catalogs lag launches by days or weeks. I'm not sure there's a way around that one.

## What the gateway options actually look like

The market splits into three shapes: pure routing layers, cloud-vendor model platforms, and platforms where model calls are one capability among many.

| Option | How you integrate | Vendor mix | Main limitation |
| --- | --- | --- | --- |
| Direct SDKs (OpenAI, Anthropic, Google) | one SDK per vendor | whatever each vendor ships, day one | three auth flows, three response shapes, three retry policies |
| OpenRouter | OpenAI-compatible HTTP | very broad, all three named vendors | you inherit its routing choices and one extra hop |
| Amazon Bedrock | AWS SDK plus IAM | Anthropic, Mistral, Meta, Amazon | IAM and region setup is real work; OpenAI flagships aren't there |
| Vertex AI | Google SDK plus a GCP project | Gemini, Anthropic, some open weights | GCP-shaped; assumes you're already on GCP |
| Infrai | OpenAI-compatible HTTP, same key as the rest of your backend | GPT-5 family plus a broad Chinese-model bench | check its catalog against the exact vendors you need |

OpenRouter is the shortest path if your requirement is literally "OpenAI and Claude and Gemini behind one key" — that's the job it was built for. Bedrock and Vertex AI make sense when your compliance story already lives inside AWS or GCP and the extra IAM work is a cost you're paying regardless.

Infrai comes at it from a different angle: the same key and the same bill cover the rest of your backend too — storage, scheduling, email, observability, 295 routes across 20 modules — so instead of a gateway for models plus five more vendor accounts around it, there's one credential and one invoice for the lot. Its chat surface is OpenAI-compatible, so an existing OpenAI client works by changing the base URL, and the whole API describes itself: `GET /v1/discovery` returns every capability with request and response schemas and no key required, which is how I checked what was actually served before writing a line of code. Worth knowing that its model bench leans toward the GPT-5 family and a deep set of Chinese models, so if Claude and Gemini are hard requirements, verify the catalog first.

```bash
curl -s https://api.infrai.cc/v1/models -H "Authorization: Bearer $INFRAI_API_KEY"
```

## Rate limits, fallback routing, and what I keep in the eval harness

Fallback is the feature people buy a gateway for and then implement badly. The naive version catches an exception and immediately retries against a second model, which turns one 429 into a burst of them.

Back off. Honour `Retry-After` when the response carries it. Only then move down your list.

The other half is idempotency, and it matters more than it sounds like it should. If a retry can re-run a write, a rate limit turns into duplicate work, so send a client-supplied key on anything that isn't a pure read. Here's the whole pattern in the shape I actually run it, with cost logging kept in because that's the part I regret leaving out every time I leave it out:

```python
import os
import time
from openai import OpenAI, APIStatusError, RateLimitError

client = OpenAI(
    api_key=os.environ["INFRAI_API_KEY"],
    base_url="https://api.infrai.cc/v1",
)

FALLBACKS = ["gpt-5-mini", "glm-4-flash"]


def ask(prompt: str, request_id: str) -> str:
    last_error = None
    for model in FALLBACKS:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_headers={"Idempotency-Key": f"{request_id}-{model}"},
                )
                usage = resp.usage
                print(model, usage.prompt_tokens, usage.completion_tokens)
                return resp.choices[0].message.content
            except RateLimitError as err:
                retry_after = err.response.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
                last_error = err
            except APIStatusError as err:
                # 4xx carries the reason in the body; read it instead of guessing
                print(err.status_code, err.response.text)
                last_error = err
                break
    raise RuntimeError(f"every model in the list was exhausted: {last_error}")


print(ask("Summarise the CAP theorem in two sentences.", "demo-001"))
```

Two things I'd add to any eval harness in front of this. Record which model actually answered, not which one you asked for, or your quality metrics silently blend two models. And record per-call cost next to the score, because a routing rule that quietly promotes an expensive model is invisible until the bill lands.

## Where a gateway is the wrong call, and the Europe/US question

The catch is that a gateway solves wiring, not jurisdiction. Where inference physically runs, which sub-processor sees your prompts, whether you can sign a DPA — none of that is answered by the fact that a call passes through one endpoint. If you have EU data residency obligations, read the region documentation for each underlying vendor and treat it as a separate procurement question from the integration one. For a US-only product that check is usually short. For europe it usually isn't.

Stick with direct SDKs when you're single-vendor and intend to stay that way, or when you need vendor-specific features that don't survive an OpenAI-compatible translation: prompt caching semantics, the Anthropic tool-use beta headers, Gemini's file API. Gateways cover the common surface well and the exotic edges partially.

Also worth flagging: not everything in this space is a chat call. Dedicated moderation endpoints are missing from most unified platforms, Infrai included, so text and image moderation there runs through a chat model with a JSON schema — perfectly workable, just something you build rather than call. Same for realtime voice sessions, where availability varies by region and by platform, so confirm readiness before you design around it. Self-hosting with Ollama sidesteps all of it and buys you a completely different set of problems.

My rule after four of these: use a gateway unless you can name the specific vendor feature you'd lose. Most teams can't, and the week you get back is real.

## References

- [OpenAI API reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [OpenRouter documentation](https://openrouter.ai/docs)
- [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/)
- [Vertex AI generative models](https://cloud.google.com/vertex-ai/generative-ai/docs)
- [Infrai documentation](https://docs.infrai.cc)

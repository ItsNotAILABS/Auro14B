# Launch HIM Today

## Fastest local start

    python scripts/launch_him.py

This opens http://127.0.0.1:8090 with the checked-in HIM-native-v0 checkpoint.
It proves the open-weight path and handles grounded Auro/MESIE/HIM questions
through an explicitly labeled local orchestration path. It is not a fluent
general-purpose model.

## Fluent local-model start

    python scripts/launch_him.py \
      --base-url http://127.0.0.1:8088/v1 \
      --model Auro-HIM-14B \
      --parameter-count 14000000000

The compatible endpoint can be vLLM, llama.cpp, or an internal inference server
running promoted Auro weights. HIM reports the actual model lane, provider,
parameter count, routing attempts, and answer origin.

## Give HIM large working memory

    python scripts/him_context.py ingest docs/ README.md your-project/
    python scripts/him_context.py stats
    python scripts/him_context.py query "What are the release gates?" --tokens 12000

The Python context engine can store millions of logical tokens and packs a
bounded working set per turn. Injection budgets up to 300,000 are configurable,
but the selected endpoint must support the resulting prompt. Logical storage is
not native single-pass transformer attention.

## Demo

1. Ask “What is HIM?” in Chat.
2. Paste project documentation into Context.
3. Ask a question that requires the pasted fact.
4. Inspect source IDs, models, routes, context accounting, and receipts.
5. Enable execution only for a bounded task using the separate token printed at launch.

Loopback chat needs no read token by default. Non-loopback binding requires
--secure. Credentials stay in the browser tab.

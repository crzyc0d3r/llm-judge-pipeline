# Train Your Own Small LLM Judge (Insurance RAG Grounding Demo)

Creation of the hands-on from the [Daily Dose of Data Science](https://www.dailydoseofds.com) newsletter issue *"A Better Way To Build LLM-as-a-Judge Pipelines"* (2026-07-01), by Avi Chawla & Akshay Pachaar with the PlurAI team. All credit to DDODS and [PlurAI](https://github.com/plurai-ai/plurai-plugins).

## Why

Using a frontier model as the judge on every production check has three problems: **cost** (an API hit per turn), **latency** (big remote models), and **domain blind spots** (frontier models miss the keywords and principles your domain depends on). The alternative: train your own **small judge** once, then run it cheaply forever, served behind an OpenAI-compatible endpoint (on-prem capable).

## Pipeline (as described in the issue)

1. **Domain decomposition** — split the domain (insurance) into sub-areas.
2. **Synthetic data generation** — sample (context, answer) examples per sub-area. `judge_pipeline/generate_data.py`
3. **Debate arena** — multiple judges score each example; only unanimous-consensus labels survive. `judge_pipeline/debate_arena.py`
4. **Train the small judge** on the refined set. `judge_pipeline/train_judge.py`
5. **Serve** via an OpenAI-compatible `/v1/chat/completions` endpoint. `judge_pipeline/serve.py`

The demo task is the same as the issue's: an **insurance RAG grounding evaluator** — given retrieved policy context and a generated answer, classify the answer as `grounded` or `ungrounded`.

## Notes on this recreation

The newsletter's artifact is a Claude Code plugin + web app (`plurai-ai/plurai-plugins`, app.plurai.ai) with no code printed in the email, so this repo re-implements the described pipeline as a fully offline, dependency-light demo: the "debate arena" uses three heuristic judges standing in for LLM judges, and the "small model" is a TF-IDF + logistic-regression classifier standing in for an SLM fine-tune (swap in your own at `build_model()`). If a sample requires an API key (e.g., real LLM judges), use the `OPENAI_API_KEY` placeholder — never commit real keys.

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py                     # generate -> arena -> train (~seconds)
uvicorn judge_pipeline.serve:app --port 8000
```

Query it like any OpenAI-compatible model:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "CONTEXT: The policy'"'"'s collision deductible is $500.\nANSWER: According to your policy, the collision deductible is $1,000,000."}]
}'
```

## Links

- Original plugin: https://github.com/plurai-ai/plurai-plugins
- Newsletter: https://www.dailydoseofds.com

"""Stage 1-2: Domain decomposition + synthetic example generation.

Demo domain (from the DDODS issue, 2026-07-01): an insurance RAG grounding
evaluator. Each example is (context, answer) and the judge must decide whether
the answer is GROUNDED in the context or UNGROUNDED (hallucinated).

By default this generates data offline from templates so the demo runs with
no API key. Set OPENAI_API_KEY (or any OpenAI-compatible endpoint via
OPENAI_BASE_URL) to generate richer data with an LLM instead.
"""

import json
import random
from pathlib import Path

random.seed(42)

# Domain decomposition: sub-areas of the insurance domain the judge must cover
SUBDOMAINS = ["auto", "home", "health", "life", "travel"]

FACTS = {
    "auto":   [("collision deductible", "$500"), ("liability limit", "$100,000"),
               ("rental reimbursement", "$30/day"), ("roadside assistance", "included")],
    "home":   [("dwelling coverage", "$350,000"), ("flood damage", "excluded"),
               ("wind deductible", "2%"), ("personal property limit", "$75,000")],
    "health": [("annual out-of-pocket max", "$6,500"), ("specialist copay", "$45"),
               ("out-of-network coverage", "60%"), ("preventive care", "covered in full")],
    "life":   [("term length", "20 years"), ("death benefit", "$250,000"),
               ("contestability period", "2 years"), ("conversion option", "available until age 65")],
    "travel": [("trip cancellation", "up to $10,000"), ("baggage delay", "$200 after 12 hours"),
               ("emergency medical", "$50,000"), ("pre-existing conditions", "excluded")],
}

WRONG_VALUES = ["$1,000,000", "not covered", "$5", "90 days", "excluded", "unlimited"]


def make_example(subdomain: str, grounded: bool) -> dict:
    facts = random.sample(FACTS[subdomain], k=2)
    context = " ".join(f"The policy's {k} is {v}." for k, v in facts)
    key, value = random.choice(facts)
    if grounded:
        answer = f"According to your policy, the {key} is {value}."
    else:
        wrong = random.choice([w for w in WRONG_VALUES if w != value])
        answer = f"According to your policy, the {key} is {wrong}."
    return {"subdomain": subdomain, "context": context, "answer": answer,
            "label": int(grounded)}


def generate(n: int = 400, path: str = "data/synthetic_raw.jsonl") -> list[dict]:
    examples = [make_example(random.choice(SUBDOMAINS), random.random() < 0.5)
                for _ in range(n)]
    out = Path(path)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(e) for e in examples))
    print(f"Generated {len(examples)} synthetic examples -> {out}")
    return examples


if __name__ == "__main__":
    generate()

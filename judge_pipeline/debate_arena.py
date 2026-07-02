"""Stage 3: Multi-judge "debate arena" — refine labels by consensus.

Several independent judges score each synthetic example; examples where the
judges cannot reach consensus are dropped, so the small model trains only on
clean, agreed-upon labels. In production each judge would be a different LLM
(the DDODS/PlurAI pipeline uses frontier models here, once, offline — the
whole point is that the expensive models label the training set, not every
production request). This offline demo uses three heuristic judges standing
in for LLM judges; plug in real models via OPENAI_API_KEY if desired.
"""

import json
import re
from pathlib import Path

PRICE_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\d+%|\d+ (?:years|days|hours)")


def judge_exact_value(ex: dict) -> int:
    """Judge A: is the value claimed in the answer present in the context?"""
    claimed = PRICE_RE.findall(ex["answer"])
    return int(all(v in ex["context"] for v in claimed)) if claimed else 1


def judge_key_alignment(ex: dict) -> int:
    """Judge B: does the (key, value) pair in the answer appear in context?"""
    m = re.search(r"the (.+?) is (.+?)\.$", ex["answer"])
    if not m:
        return 1
    key, value = m.groups()
    return int(f"{key} is {value}" in ex["context"])


def judge_token_overlap(ex: dict) -> int:
    """Judge C: crude lexical-support check on the answer's tail tokens."""
    tail = ex["answer"].rstrip(".").split(" is ")[-1]
    return int(tail in ex["context"])


JUDGES = [judge_exact_value, judge_key_alignment, judge_token_overlap]


def run_arena(in_path: str = "data/synthetic_raw.jsonl",
              out_path: str = "data/train_consensus.jsonl") -> None:
    examples = [json.loads(l) for l in Path(in_path).read_text().splitlines()]
    kept, dropped = [], 0
    for ex in examples:
        votes = [j(ex) for j in JUDGES]
        if len(set(votes)) == 1:  # unanimous consensus
            ex["label"] = votes[0]
            kept.append(ex)
        else:
            dropped += 1
    Path(out_path).write_text("\n".join(json.dumps(e) for e in kept))
    print(f"Arena consensus: kept {len(kept)}, dropped {dropped} (no consensus) -> {out_path}")


if __name__ == "__main__":
    run_arena()

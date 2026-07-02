"""Stage 4: Train the small judge on the consensus-refined dataset.

The real DDODS/PlurAI pipeline fine-tunes a small language model (SLM). To
keep this demo runnable on any laptop with no GPU or API key, the "small
judge" is a TF-IDF + logistic-regression classifier over (context, answer)
pairs — the drop-in seam for an SLM fine-tune is `build_model()`.
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/judge.pkl")


def featurize(ex: dict) -> str:
    """Text features + explicit context/answer alignment features.

    Grounding is fundamentally about whether the answer's claims appear in
    the context, so we add overlap-bucket tokens the classifier can learn
    from (a standard feature-engineering seam; an SLM fine-tune would learn
    this alignment directly from the raw pair)."""
    context, answer = ex["context"], ex["answer"]
    ctx_tokens = set(context.lower().replace(".", " ").split())
    ans_tokens = [t for t in answer.lower().rstrip(".").split() if len(t) > 2]
    overlap = sum(t in ctx_tokens for t in ans_tokens) / max(len(ans_tokens), 1)
    bucket = f"OVBUCKET_{int(overlap * 10)}"
    # value-alignment: does the trailing claimed value appear verbatim in context?
    tail = answer.rstrip(".").split(" is ")[-1].lower()
    value_flag = "VAL_IN_CTX" if tail in context.lower() else "VAL_NOT_IN_CTX"
    return f"{answer} {bucket} {value_flag}"


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(in_path: str = "data/train_consensus.jsonl") -> None:
    examples = [json.loads(l) for l in Path(in_path).read_text().splitlines()]
    X = [featurize(e) for e in examples]
    y = [e["label"] for e in examples]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te), zero_division=0,
                                target_names=["ungrounded", "grounded"]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    print(f"Judge saved -> {MODEL_PATH}")


if __name__ == "__main__":
    train()

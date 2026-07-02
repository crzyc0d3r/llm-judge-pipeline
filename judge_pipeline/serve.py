"""Stage 5: Serve the trained judge via an OpenAI-compatible endpoint.

POST /v1/chat/completions with a user message of the form:
    CONTEXT: <retrieved context>
    ANSWER: <model answer to evaluate>
Returns a chat completion whose content is JSON:
    {"verdict": "grounded" | "ungrounded", "confidence": 0.97}

Because the endpoint is OpenAI-compatible, any existing eval harness pointed
at a frontier judge can be re-pointed here (on-prem capable).

Run:  uvicorn judge_pipeline.serve:app --port 8000
"""

import json
import pickle
import time
import uuid
from pathlib import Path

from fastapi import FastAPI

from judge_pipeline.train_judge import featurize
from pydantic import BaseModel

MODEL_PATH = Path("models/judge.pkl")
app = FastAPI(title="Small LLM Judge (demo)")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = pickle.loads(MODEL_PATH.read_bytes())
    return _model


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "small-judge-demo"
    messages: list[ChatMessage]


def parse_prompt(text: str) -> tuple[str, str]:
    context, answer = "", ""
    for line in text.splitlines():
        if line.upper().startswith("CONTEXT:"):
            context = line[8:].strip()
        elif line.upper().startswith("ANSWER:"):
            answer = line[7:].strip()
    return context, answer


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    context, answer = parse_prompt(req.messages[-1].content)
    model = get_model()
    text = featurize({"context": context, "answer": answer})
    proba = float(model.predict_proba([text])[0][1])
    verdict = "grounded" if proba >= 0.5 else "ungrounded"
    content = json.dumps({"verdict": verdict,
                          "confidence": round(max(proba, 1 - proba), 4)})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

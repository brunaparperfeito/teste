from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .inference import answer

app = FastAPI(title="AmazonTitles QA Demo")

class AskPayload(BaseModel):
    title: str
    question: str
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    peft: Optional[str] = None
    data: str = "data/trn.json"

@app.post("/ask")
def ask(payload: AskPayload):
    text, info = answer(
        model_name=payload.model,
        peft_path=payload.peft,
        title=payload.title,
        question=payload.question,
        data_path=payload.data,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        qlora=True
    )
    return {"answer": text, "retrieval": info}

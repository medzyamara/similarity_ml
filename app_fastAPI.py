from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn

app = FastAPI()
model = SentenceTransformer('all-mpnet-base-v2')

class CompareRequest(BaseModel):
    correct_answer: str
    user_answer: str

@app.post("/check_answer")
def check_answer(data: CompareRequest):
    embeddings = model.encode([data.correct_answer, data.user_answer], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    is_correct = similarity_score >= 0.65
    return {
        "similarity_score": round(similarity_score, 2),
        "is_correct": is_correct
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

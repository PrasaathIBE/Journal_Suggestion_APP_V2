from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding API")

# Load model once at startup
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
def embed(req: EmbedRequest):
    vector = model.encode(
        [req.text],
        normalize_embeddings=True
    )[0]
    return {
        "vector": vector.tolist()
    }

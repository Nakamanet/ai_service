import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(title="NakamaNet AI Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
AI_INDEX = []
LAST_LOAD_TIME = 0

@app.on_event("startup")
def startup_event():
    global model
    print("Loading inference model for web queries...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Inference model loaded. Ready for searches!", flush=True)

def load_index_if_available():
    global AI_INDEX, LAST_LOAD_TIME
    if os.path.exists("ai_index.pkl"):
        mtime = os.path.getmtime("ai_index.pkl")
        if mtime > LAST_LOAD_TIME:
            try:
                with open("ai_index.pkl", "rb") as f:
                    AI_INDEX = pickle.load(f)
                LAST_LOAD_TIME = mtime
                print("Reloaded fresh AI Index from disk!", flush=True)
            except Exception as e:
                print(f"Error reading index file: {e}", flush=True)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def search_exact(query, text):
    if not text: return False
    return query.lower() in text.lower()

@app.get("/")
async def health_check():
    return {"status": "AI Microservice is running!"}

@app.get("/search")
def search(q: str = Query(..., min_length=1), filter: str = "all", skip: int = 0, limit: int = 20):
    load_index_if_available()
    
    if not AI_INDEX:
        return [{"id": "0", "type": "system", "title": "Connexion à la base d'informations...", "description": "Nous récupérons les données depuis le serveur. Cela peut prendre un peu de temps pour établir la connexion à la base de données. Réessayez d'ici une minute !"}]

    results = []
    
    if model:
        query_embedding = model.encode(q, show_progress_bar=False)
        for item in AI_INDEX:
            if filter != "all" and item["type"] != filter:
                continue
            
            score = float(cosine_similarity(query_embedding, item["embedding"]))
            exact_match = search_exact(q, item["text"])
            if score > 0.15 or exact_match:
                bonified_score = score + (0.5 if exact_match else 0)
                item_copy = {k: v for k, v in item.items() if k not in ["embedding", "text"]}
                item_copy["score"] = bonified_score
                results.append(item_copy)
                
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[skip : skip + limit]
    
    return []

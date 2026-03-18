import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
import psycopg2
from sentence_transformers import SentenceTransformer
import pickle
import time

def get_catalog_connection():
    return psycopg2.connect(host="api-db-1", port=5432, user="catalog", password="catalog", dbname="catalog", connect_timeout=5)

def get_laravel_connection():
    return psycopg2.connect(host="laravel_db", port=5432, user="laravel", password="secret", dbname="laravel", connect_timeout=5)

def build():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        return

    while True:
        items = []
        try:
            conn = get_catalog_connection()
            cur = conn.cursor()
            # FULL DATABASE SCAN!
            cur.execute('SELECT id, title_en, title_jp, synopsis FROM "Anime"')
            for row in cur.fetchall():
                text = f"Anime: {row[1]} {row[2]}. {row[3]}"
                items.append({"id": f"anime_{row[0]}", "type": "anime", "title": row[1] or row[2] or "Anime", "description": (row[3][:120]+'...') if row[3] else "", "text": text})
            
            cur.execute('SELECT id, title_en, title_jp, synopsis FROM "Manga"')
            for row in cur.fetchall():
                text = f"Manga: {row[1]} {row[2]}. {row[3]}"
                items.append({"id": f"manga_{row[0]}", "type": "manga", "title": row[1] or row[2] or "Manga", "description": (row[3][:120]+'...') if row[3] else "", "text": text})
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Catalog DB error: {e}", flush=True)

        try:
            conn = get_laravel_connection()
            cur = conn.cursor()
            cur.execute('SELECT id, username, bio FROM "Users" WHERE is_deleted = false')
            for row in cur.fetchall():
                text = f"User: {row[1]}. Bio: {row[2]}"
                items.append({"id": f"user_{row[0]}", "type": "users", "title": row[1], "description": (row[2][:120]+'...') if row[2] else "", "text": text})
            
            cur.execute('SELECT id, content FROM "Posts"')
            for row in cur.fetchall():
                text = f"Community Post: {row[1]}"
                items.append({"id": f"post_{row[0]}", "type": "posts", "title": f"Post #{row[0]}", "description": (row[1][:120]+'...') if row[1] else "", "text": text})
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Laravel DB error: {e}", flush=True)

        if not items:
            print("No items found. Retrying in 10s...", flush=True)
            time.sleep(10)
            continue

        print(f"Index Builder: Encoding {len(items)} items...", flush=True)
        texts = [item["text"] for item in items]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        for i, emb in enumerate(embeddings):
            items[i]["embedding"] = emb

        # Save to disk atomically
        with open("ai_index.pkl.tmp", "wb") as f:
            pickle.dump(items, f)
        os.rename("ai_index.pkl.tmp", "ai_index.pkl")

        print(f"Index Builder: Successfully saved {len(items)} items to ai_index.pkl!", flush=True)
        
        # Re-build index every 24 HOURS to avoid locking CPU permanently in background
        time.sleep(86400)

if __name__ == "__main__":
    print("Starting background AI Index Builder process...", flush=True)
    build()

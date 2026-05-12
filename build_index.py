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
    return psycopg2.connect(host="laravel_db", port=5432, user="root", password="secret", dbname="backend_nakamanet", connect_timeout=5)

def build():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        return

    while True:
        # Load existing cache to avoid re-encoding everything
        cache = {}
        if os.path.exists("ai_index.pkl"):
            try:
                with open("ai_index.pkl", "rb") as f:
                    old_data = pickle.load(f)
                    cache = {item["id"]: item for item in old_data}
                print(f"Index Builder: Loaded cache with {len(cache)} items.", flush=True)
            except Exception as e:
                print(f"Index Builder: Could not load cache: {e}", flush=True)

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
            
            # Fetch Anime Libraries
            anime_libraries = {}
            try:
                cur.execute('SELECT user_id, anime_id FROM "User_Anime_Library"')
                for row in cur.fetchall():
                    u_id, a_id = row
                    if u_id not in anime_libraries:
                        anime_libraries[u_id] = set()
                    anime_libraries[u_id].add(a_id)
            except Exception as e:
                print(f"Error fetching Anime Library: {e}", flush=True)

            # Fetch Manga Libraries
            manga_libraries = {}
            try:
                cur.execute('SELECT user_id, manga_id FROM "User_Manga_Library"')
                for row in cur.fetchall():
                    u_id, m_id = row
                    if u_id not in manga_libraries:
                        manga_libraries[u_id] = set()
                    manga_libraries[u_id].add(m_id)
            except Exception as e:
                print(f"Error fetching Manga Library: {e}", flush=True)

            cur.execute('SELECT id, username, bio FROM "Users" WHERE is_deleted = false')
            for row in cur.fetchall():
                text = f"User: {row[1]}. Bio: {row[2]}"
                items.append({
                    "id": f"user_{row[0]}", 
                    "type": "users", 
                    "title": row[1], 
                    "description": (row[2][:120]+'...') if row[2] else "", 
                    "text": text,
                    "anime_library": list(anime_libraries.get(row[0], set())),
                    "manga_library": list(manga_libraries.get(row[0], set()))
                })
            
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

        # Check for cached embeddings ONLY - library data is always refreshed from DB
        # (Library can change without changing the text/bio, so we must never cache it)
        for item in items:
            cached_item = cache.get(item["id"])
            if cached_item and cached_item.get("text") == item["text"] and "embedding" in cached_item:
                # Reuse only the embedding - keep library data from current DB fetch
                item["embedding"] = cached_item["embedding"]

        # --- PHASE 1: Priority Items (Users) ---
        priority_to_encode = [i for i, item in enumerate(items) if item["type"] == "users" and "embedding" not in item]
        if priority_to_encode:
            print(f"Index Builder: Encoding {len(priority_to_encode)} PRIORITY items (Users)...", flush=True)
            texts = [items[i]["text"] for i in priority_to_encode]
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
            for idx, emb in zip(priority_to_encode, embeddings):
                items[idx]["embedding"] = emb
            
            # Save partial index immediately so users are searchable
            with open("ai_index.pkl.tmp", "wb") as f:
                pickle.dump(items, f)
            os.rename("ai_index.pkl.tmp", "ai_index.pkl")
            print("Index Builder: Priority items saved! Users are now searchable.", flush=True)

        # --- PHASE 2: Remaining Items (Catalog) ---
        remaining_to_encode = [i for i, item in enumerate(items) if "embedding" not in item]
        if remaining_to_encode:
            print(f"Index Builder: Encoding {len(remaining_to_encode)} remaining catalog items...", flush=True)
            # Encode in chunks of 500 to save progress periodically if needed
            CHUNK_SIZE = 500
            for i in range(0, len(remaining_to_encode), CHUNK_SIZE):
                chunk_indices = remaining_to_encode[i : i + CHUNK_SIZE]
                texts = [items[idx]["text"] for idx in chunk_indices]
                embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
                for idx, emb in zip(chunk_indices, embeddings):
                    items[idx]["embedding"] = emb
                
                # Save progress after each chunk
                with open("ai_index.pkl.tmp", "wb") as f:
                    pickle.dump(items, f)
                os.rename("ai_index.pkl.tmp", "ai_index.pkl")
                print(f"Index Builder: Progress saved ({i + len(chunk_indices)}/{len(remaining_to_encode)})...", flush=True)

        # Always save the full index after sync to persist library updates
        with open("ai_index.pkl.tmp", "wb") as f:
            pickle.dump(items, f)
        os.rename("ai_index.pkl.tmp", "ai_index.pkl")
        
        print(f"Index Builder: Successfully synchronized {len(items)} items!", flush=True)
        # Re-build index every 5 minutes to keep results fresh
        time.sleep(300)

if __name__ == "__main__":
    print("Starting background AI Index Builder process...", flush=True)
    build()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
import psycopg2
from sentence_transformers import SentenceTransformer
import pickle
import time

def get_catalog_connection():
    return psycopg2.connect(
        host="ep-dry-mouse-a23cg586-pooler.eu-central-1.aws.neon.tech", 
        port=5432, 
        user="neondb_owner", 
        password="npg_CETMGsq26xYo", 
        dbname="neondb", 
        sslmode="require",
        connect_timeout=10
    )

def get_laravel_connection():
    return psycopg2.connect(
        host="ep-empty-cherry-asjuu2np-pooler.c-4.eu-central-1.aws.neon.tech", 
        port=5432, 
        user="neondb_owner", 
        password="npg_cK0eQim8VGoC", 
        dbname="neondb", 
        sslmode="require",
        connect_timeout=10
    )

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
            cur.execute('SELECT id, title_en, title_jp, synopsis, slug FROM "Anime"')
            for row in cur.fetchall():
                text = f"Anime: {row[1]} {row[2]}. {row[3]}"
                items.append({"id": f"anime_{row[0]}", "type": "anime", "title": row[1] or row[2] or "Anime", "description": (row[3][:120]+'...') if row[3] else "", "text": text, "slug": row[4]})
            
            cur.execute('SELECT id, title_en, title_jp, synopsis, slug FROM "Manga"')
            for row in cur.fetchall():
                text = f"Manga: {row[1]} {row[2]}. {row[3]}"
                items.append({"id": f"manga_{row[0]}", "type": "manga", "title": row[1] or row[2] or "Manga", "description": (row[3][:120]+'...') if row[3] else "", "text": text, "slug": row[4]})
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

            user_map = {}
            cur.execute('SELECT id, username, bio, avatar_url FROM "Users" WHERE is_deleted = false')
            for row in cur.fetchall():
                user_map[row[0]] = row[1]
                text = f"User: {row[1]}. Bio: {row[2]}"
                items.append({
                    "id": f"user_{row[0]}", 
                    "type": "users", 
                    "title": row[1], 
                    "description": (row[2][:120]+'...') if row[2] else "", 
                    "text": text,
                    "anime_library": list(anime_libraries.get(row[0], set())),
                    "manga_library": list(manga_libraries.get(row[0], set())),
                    "user_id": row[0],
                    "avatar_url": row[3] if row[3] else None
                })
            
            cur.execute('SELECT id, user_id, content FROM "Posts"')
            for row in cur.fetchall():
                username = user_map.get(row[1], "Unknown User")
                text = f"Community Post by {username}: {row[2]}"
                items.append({"id": f"post_{row[0]}", "type": "posts", "title": f"Post by {username}", "description": (row[2][:120]+'...') if row[2] else "", "text": text, "post_id": row[0]})
                
            try:
                cur.execute('SELECT id, user_id, title, content FROM "Forum_Topics"')
                for row in cur.fetchall():
                    username = user_map.get(row[1], "Unknown User")
                    text = f"Forum Topic by {username}: {row[2]}. {row[3]}"
                    items.append({"id": f"forum_{row[0]}", "type": "forums", "title": f"Forum: {row[2]}", "description": (row[3][:120]+'...') if row[3] else "", "text": text, "forum_id": row[0]})
            except Exception as e:
                print(f"Error fetching Forum Topics: {e}", flush=True)
                
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
        # Re-build index every 7 days to keep results fresh and avoid NeonDB charges
        time.sleep(604800)

if __name__ == "__main__":
    print("Starting background AI Index Builder process...", flush=True)
    build()

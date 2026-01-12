import os
import json
import requests
import pandas as pd
import joblib
import time
import re

# ---------------- CONFIG ----------------
EMBED_MODEL = "bge-m3"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

JSON_DIR = "src/jsons"
OUTPUT_FILE = "src/embeddings.joblib"

MAX_CHARS = 1500          # VERY SAFE for bge-m3
RETRY_COUNT = 3
SLEEP_TIME = 0.2          # Prevent Ollama overload
TIMEOUT = 120
# ----------------------------------------

session = requests.Session()


# -------- TEXT CLEANING (CRITICAL) --------
def clean_text(text: str) -> str:
    text = text.strip()

    # Remove timestamps like [00:01:23]
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove problematic characters
    text = text.replace("\x00", "")

    return text[:MAX_CHARS]


def embed_single_text_safe(text: str):
    """Embed with retries and fallback."""
    for attempt in range(RETRY_COUNT):
        try:
            r = session.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": EMBED_MODEL,
                    "input": text
                },
                timeout=TIMEOUT
            )

            if r.status_code == 200:
                return r.json()["embeddings"][0]

        except Exception:
            pass

        time.sleep(0.5)  # backoff

    raise RuntimeError("Ollama error 500")


# -------- MAIN PROCESS --------
all_chunks = []
chunk_id = 0
skipped_chunks = []

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

for json_file in json_files:
    print(f"\n🔹 Processing: {json_file}")

    with open(os.path.join(JSON_DIR, json_file), "r", encoding="utf-8") as f:
        content = json.load(f)

    for chunk in content["chunks"]:
        cleaned_text = clean_text(chunk["text"])

        if not cleaned_text:
            skipped_chunks.append(chunk_id)
            chunk_id += 1
            continue

        try:
            embedding = embed_single_text_safe(cleaned_text)

            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embedding
            all_chunks.append(chunk)

        except Exception:
            print(f"❌ Skipped chunk {chunk_id}")
            skipped_chunks.append(chunk_id)

        chunk_id += 1
        time.sleep(SLEEP_TIME)

print("\n✅ Embedding process completed")

print(f"⚠️ Skipped chunks: {len(skipped_chunks)}")

df = pd.DataFrame(all_chunks)
joblib.dump(df, OUTPUT_FILE)

print(f"💾 Saved embeddings → {OUTPUT_FILE}")

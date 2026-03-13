"""
ingest_news.py
--------------
Loads ISOT Fake News Dataset (Fake.csv + True.csv),
samples articles, generates embeddings, and indexes
everything into Endee vector DB.

Usage:
    python ingest_news.py
    python ingest_news.py --limit 5000   # index only 5000 articles
"""

import os
import uuid
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_NAME     = "news_articles"
EMBEDDING_DIM  = 384
ENDEE_HOST     = "http://localhost:8080"
FAKE_CSV       = "./data/Fake.csv"
TRUE_CSV       = "./data/True.csv"
DEFAULT_LIMIT  = 3000   # articles per class (3000 fake + 3000 true = 6000 total)
BATCH_SIZE     = 256
# ───────────────────────────────────────────────────────────────────────────────


def load_dataset(limit: int):
    """Load and sample from ISOT Fake and True CSVs."""
    print(f"📂 Loading datasets from {FAKE_CSV} and {TRUE_CSV} …")

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    print(f"   Fake articles available: {len(fake_df)}")
    print(f"   True articles available: {len(true_df)}")

    # Sample equally from both
    fake_df = fake_df.sample(min(limit, len(fake_df)), random_state=42)
    true_df = true_df.sample(min(limit, len(true_df)), random_state=42)

    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"   ✅ Total articles to index: {len(df)} ({len(fake_df)} fake + {len(true_df)} real)")
    return df


def build_text(row) -> str:
    """Combine title + text for richer embedding."""
    title = str(row.get("title", "")).strip()
    text  = str(row.get("text",  "")).strip()
    subj  = str(row.get("subject", "")).strip()
    # Use title + first 400 chars of body for embedding
    combined = f"{title}. {text[:400]}"
    return combined.strip()


def main(limit: int):
    # 1. Load dataset
    df = load_dataset(limit)

    # 2. Connect to Endee
    print("\n🔌 Connecting to Endee vector DB …")
    client = Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")

    # Delete existing index
    try:
        client.delete_index(INDEX_NAME)
        print(f"   Deleted existing index '{INDEX_NAME}'")
    except Exception:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    print(f"   ✅ Index '{INDEX_NAME}' created (dim={EMBEDDING_DIM}, cosine, INT8)")
    index = client.get_index(name=INDEX_NAME)

    # 3. Load embedding model
    print("\n🤖 Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Build texts
    print("\n📝 Preparing article texts …")
    texts = [build_text(row) for _, row in df.iterrows()]

    # 5. Generate embeddings in batches
    print(f"⚡ Generating embeddings for {len(texts)} articles …")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    # 6. Upsert into Endee
    print(f"\n📤 Uploading to Endee in batches of {BATCH_SIZE} …")
    total = 0
    items = []

    for i, (vec, (_, row)) in enumerate(zip(vectors, df.iterrows())):
        title = str(row.get("title", "No Title"))[:300]
        text  = str(row.get("text",  ""))[:500]
        items.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "meta": {
                "title":   title,
                "text":    text,
                "label":   row["label"],
                "subject": str(row.get("subject", "general")),
                "date":    str(row.get("date", "unknown")),
            },
        })

        if len(items) == BATCH_SIZE:
            index.upsert(items)
            total += len(items)
            print(f"   Uploaded {total}/{len(df)} articles …", end="\r")
            items = []

    # Upload remaining
    if items:
        index.upsert(items)
        total += len(items)

    print(f"\n✅ Done! Indexed {total} articles into Endee.")
    print(f"   FAKE: {len(df[df['label']=='FAKE'])}  |  REAL: {len(df[df['label']=='REAL'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="Max articles per class to index")
    args = parser.parse_args()
    main(args.limit)

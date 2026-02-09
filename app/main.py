import io
import os
import re
from typing import Dict, Any, List
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from sentence_transformers import SentenceTransformer

from .schemas import SuggestRequest, SuggestResponse
from .core_logic import (
    prepare_primary, prepare_fallback,
    embed_text_primary, embed_text_fallback,
    score_domains, build_query_text, normalize_key,
    add_history_scores_from_aggregates,
)
from .qdrant_store import (
    get_qdrant_client, recreate_collection, upsert_points, search,
    PRIMARY_COLLECTION, ASSOC_COLLECTION, l2_normalize
)
from .history_db import get_conn, reset_and_load_history, load_aggregates
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware


MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./history.db")

app = FastAPI(title="Journal Suggestion API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers
# -------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df

def read_any_upload(up: UploadFile) -> pd.DataFrame:
    name = (up.filename or "").lower()
    raw = up.file.read()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(raw), dtype=str)
    else:
        raise HTTPException(400, "Unsupported file type. Upload CSV or Excel.")
    return _normalize_cols(df)

_model: SentenceTransformer = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def df_to_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Store everything in payload (Qdrant supports nested but keep simple dict)
    # Convert NaN to "" for JSON friendliness
    out = []
    for _, r in df.iterrows():
        d = {}
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                d[c] = ""
            else:
                # dates may be python date; keep string
                d[c] = str(v)
        out.append(d)
    return out

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").strip()

def embed_remote(text: str) -> np.ndarray:
    if not EMBEDDING_API_URL:
        raise HTTPException(status_code=500, detail="EMBEDDING_API_URL not set")
    resp = requests.post(
        EMBEDDING_API_URL,
        json={"text": text},
        timeout=60
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["vector"], dtype=np.float32)
    return l2_normalize(vec)  # ✅ force normalization



# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Ingest primary (SI-level)
# -------------------------
@app.post("/ingest/primary")
def ingest_primary(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Drop & recreate collection before ingest"),
):
    df_raw = read_any_upload(file)
    df = prepare_primary(df_raw)

    if "_id" not in df.columns:
        raise HTTPException(400, "Primary file must contain '_id' column.")

    model = get_model()
    texts = df.apply(embed_text_primary, axis=1).tolist()
    vectors = model.encode(texts, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    vectors = l2_normalize(vectors)
    ids = df["_id"].astype(str).tolist()
    payloads = df_to_payloads(df.assign(candidate_text=texts))

    client = get_qdrant_client()

    if reset:
        recreate_collection(client, PRIMARY_COLLECTION, dim=vectors.shape[1])

    upsert_points(client, PRIMARY_COLLECTION, ids, vectors, payloads)

    return {"collection": PRIMARY_COLLECTION, "rows_ingested": len(df), "reset": reset}


# -------------------------
# Ingest associate editor (journal-level)
# -------------------------
@app.post("/ingest/associate")
def ingest_associate(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Drop & recreate collection before ingest"),
):
    print("Reading associate editor file...")
    df_raw = read_any_upload(file)
    df = prepare_fallback(df_raw)
    

    # For associate editor, you confirmed _id exists in your latest exports.
    # If your prepared fallback df doesn't carry _id, keep original df's _id mapping:
    # simplest approach: if input has _id, use it. If not, we cannot guarantee stable IDs.
    if "_id" in df_raw.columns:
        # Try to keep _id by merging on Journal_Name_norm (since fallback aggregates by journal)
        tmp = df_raw.copy()
        tmp["Journal_Name"] = tmp.get("Journal_Name", tmp.get("Journal", "")).astype(str)
        tmp["Journal_Name_norm"] = tmp["Journal_Name"].astype(str).map(lambda x: normalize_key(x))
        id_map = tmp.dropna(subset=["_id"]).groupby("Journal_Name_norm")["_id"].first().reset_index()
        df = df.merge(id_map, on="Journal_Name_norm", how="left")
    if "_id" not in df.columns:
        raise HTTPException(400, "Associate editor file must contain '_id' (or provide it in input).")

    model = get_model()
    texts = df.apply(embed_text_fallback, axis=1).tolist()
    vectors = model.encode(texts, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    vectors = l2_normalize(vectors)  # ✅ force normalization
    ids = df["_id"].astype(str).tolist()
    payloads = df_to_payloads(df.assign(candidate_text=texts))

    client = get_qdrant_client()
    if reset:
        recreate_collection(client, ASSOC_COLLECTION, dim=vectors.shape[1])

    upsert_points(client, ASSOC_COLLECTION, ids, vectors, payloads)

    return {"collection": ASSOC_COLLECTION, "rows_ingested": len(df), "reset": reset}


# -------------------------
# Ingest history (published + rejected) daily into SQLite
# -------------------------
@app.post("/ingest/history")
def ingest_history(
    published_file: UploadFile = File(...),
    rejected_file: UploadFile = File(...),
    reset: bool = Query(True, description="Rebuild history tables from scratch"),
):
    if not reset:
        raise HTTPException(400, "This endpoint is designed for daily reset=true ingest.")

    pub_df = read_any_upload(published_file)
    rej_df = read_any_upload(rejected_file)

    conn = get_conn(SQLITE_PATH)
    reset_and_load_history(conn, pub_df, rej_df)

    return {
        "sqlite_path": SQLITE_PATH,
        "published_rows": len(pub_df),
        "rejected_rows": len(rej_df),
        "reset": True
    }


# -------------------------
# Suggest endpoint (JSON)
# -------------------------
@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "Title cannot be empty.")

    # Domain scoring
    primary_domain, top3_scores = score_domains(title)
    qtext = build_query_text(title, top3_scores)

    qvec = embed_remote(qtext)

    client = get_qdrant_client()

    # Pull aggregates from SQLite
    conn = get_conn(SQLITE_PATH)
    try:
        pub_j, rej_j, pub_si, rej_si = load_aggregates(conn)
    except Exception:
        raise HTTPException(400, "History aggregates not found. Call /ingest/history first.")

    # Search primary and/or associate based on mode
    mode = req.mode
    topk = req.topk
    weak_threshold = req.weak_threshold

    primary_hits = []
    assoc_hits = []

    if mode in ("AUTO", "PRIMARY_ONLY", "BOTH"):
        primary_hits = search(client, PRIMARY_COLLECTION, qvec, topk=topk)

    primary_top1_sim = float(primary_hits[0].score) if primary_hits else 0.0

    fallback_used = False
    if mode == "AUTO":
        fallback_used = (not primary_hits) or (primary_top1_sim < weak_threshold)
        if fallback_used:
            assoc_hits = search(client, ASSOC_COLLECTION, qvec, topk=topk)
    elif mode == "ASSOCIATE_ONLY":
        fallback_used = True
        assoc_hits = search(client, ASSOC_COLLECTION, qvec, topk=topk)
    elif mode == "BOTH":
        assoc_hits = search(client, ASSOC_COLLECTION, qvec, topk=topk)
        fallback_used = True  # because assoc is included
    else:
        # PRIMARY_ONLY
        fallback_used = False

    # Convert hits into candidate DF with same fields as Streamlit cand
    def hits_to_rows(hits, source: str):
        rows = []
        for h in hits:
            payload = h.payload or {}
            row = dict(payload)
            row["sim"] = float(h.score)
            row["source"] = source
            row["candidate_text"] = payload.get("candidate_text", "")
            # ensure norms for history joins
            row["Journal_Name_norm"] = normalize_key(row.get("Journal_Name", ""))
            row["Special_Issue_Name_norm"] = normalize_key(row.get("Special_Issue_Name", ""))
            rows.append(row)
        return rows

    cand_rows = []
    cand_rows.extend(hits_to_rows(primary_hits, "PRIMARY"))
    cand_rows.extend(hits_to_rows(assoc_hits, "ASSOC"))

    if not cand_rows:
        return SuggestResponse(
            title=title,
            primary_domain=primary_domain,
            top3_domains=[{"domain": d, "score": float(s)} for d, s in top3_scores[:3]],
            mode=mode,
            fallback_used=fallback_used,
            primary_top1_sim=primary_top1_sim,
            results=[]
        )

    cand_df = pd.DataFrame(cand_rows)

    ranked = add_history_scores_from_aggregates(
        cand_df=cand_df,
        pub_j=pub_j, rej_j=rej_j,
        pub_si=pub_si, rej_si=rej_si,
        title=title,
        title_domain=primary_domain
    )

    # Return JSON list preserving all payload columns + scores
    results = ranked.head(topk).to_dict(orient="records")

    return SuggestResponse(
        title=title,
        primary_domain=primary_domain,
        top3_domains=[{"domain": d, "score": float(s)} for d, s in top3_scores[:3]],
        mode=mode,
        fallback_used=fallback_used,
        primary_top1_sim=primary_top1_sim,
        results=results
    )

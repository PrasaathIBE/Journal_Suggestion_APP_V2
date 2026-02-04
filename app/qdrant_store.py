import os
from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
import numpy as np
from typing import List, Dict, Any


PRIMARY_COLLECTION = "journal_primary_si"
ASSOC_COLLECTION = "journal_associate_editor"

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v) + 1e-12
        return v / n
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)
    return QdrantClient(url=url, api_key=api_key)


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    # Drop if exists
    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

# Fixed namespace UUID (constant) so ids are stable forever
QDRANT_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

def to_qdrant_point_id(raw_id: str) -> str:
    """
    Convert any string id (like Mongo ObjectId) to a valid UUID for Qdrant.
    Deterministic => same raw_id always maps to same UUID.
    """
    raw_id = str(raw_id).strip()
    return str(uuid.uuid5(QDRANT_NAMESPACE, raw_id))


def upsert_points(client, collection: str, ids: List[str], vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
    points = []
    for i, raw_id in enumerate(ids):
        qid = to_qdrant_point_id(raw_id)
        payload = payloads[i]
        payload["_id"] = str(payload.get("_id", raw_id))   # keep original
        payload["qdrant_id"] = qid                         # helpful for debugging

        points.append(
            PointStruct(
                id=qid,                      # ✅ valid UUID
                vector=vectors[i].tolist(),
                payload=payload,
            )
        )
    client.upsert(collection_name=collection, points=points)

def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    topk: int,
    qfilter: Optional[Filter] = None,
):
    q = l2_normalize(query_vector)  # ✅ force normalization
    res = client.query_points(
        collection_name=collection,
        query=q.tolist(),
        limit=topk,
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )
    return res.points


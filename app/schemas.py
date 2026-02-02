from typing import Literal, Optional, Any, Dict, List
from pydantic import BaseModel, Field


SearchMode = Literal["AUTO", "PRIMARY_ONLY", "ASSOCIATE_ONLY", "BOTH"]


class SuggestRequest(BaseModel):
    title: str = Field(..., min_length=3)
    topk: int = Field(10, ge=1, le=50)
    weak_threshold: float = Field(0.35, ge=0.0, le=1.0)
    mode: SearchMode = "AUTO"


class SuggestResponse(BaseModel):
    title: str
    primary_domain: str
    top3_domains: List[Dict[str, Any]]
    mode: SearchMode
    fallback_used: bool
    primary_top1_sim: float
    results: List[Dict[str, Any]]

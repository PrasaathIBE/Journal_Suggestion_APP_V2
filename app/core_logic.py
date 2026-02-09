import re
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from dateutil import parser as date_parser


# =========================
# Utilities
# =========================
def normalize_text(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_key(x: Optional[str]) -> str:
    s = normalize_text(x).casefold()
    s = re.sub(r"[\u2010-\u2015]", "-", s)     # unicode dashes -> hyphen
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_yes_no(x: Optional[str]) -> str:
    s = normalize_key(x)
    if s in {"yes", "y", "true", "1", "open"}:
        return "Yes"
    if s in {"no", "n", "false", "0", "closed"}:
        return "No"
    return "Unknown"

def parse_deadline(x: Optional[str]):
    s = normalize_text(x)
    if not s:
        return pd.NaT
    try:
        return date_parser.parse(s, fuzzy=True).date()
    except Exception:
        return pd.NaT

def safe_join_keywords(series: pd.Series) -> str:
    vals = []
    seen = set()
    for v in series.fillna("").astype(str).tolist():
        t = normalize_text(v)
        if t and t not in seen:
            seen.add(t)
            vals.append(t)
    return " | ".join(vals)


# =========================
# Domain hinting (Phase 3-lite)
# =========================
DOMAIN_SEEDS = {
    "AI/ML & Data Science": [
        "machine learning","deep learning","neural network","ai","artificial intelligence","ml","nlp","llm","transformer",
        "computer vision","classification","prediction","regression","clustering"
    ],
    "IoT & Embedded": [
        "iot","internet of things","embedded","sensor","wearable","edge","microcontroller","raspberry pi","arduino",
        "smart device","iomt"
    ],
    "Cybersecurity & Privacy": [
        "security","cybersecurity","privacy","encryption","blockchain","authentication","intrusion","malware","attack",
        "forensics"
    ],
    "Networking & Communications": [
        "5g","6g","network","routing","wireless","communication","spectrum","qos","latency","throughput"
    ],
    "Robotics & Automation": [
        "robot","robotics","automation","autonomous","drone","uav","control","path planning","manipulator"
    ],
    "Healthcare & Biomedical": [
        "health","medical","biomedical","clinical","disease","diagnosis","patient","hospital","ecg","mri","ct","heart"
    ],
    "Energy & Environment": [
        "energy","renewable","solar","wind","battery","grid","carbon","climate","environment","sustainability"
    ],
    "Business & Management": [
        "business","management","supply chain","finance","marketing","hr","human resource","strategy","enterprise"
    ],
}

def extract_concepts_from_title(title: str) -> List[str]:
    t = normalize_key(title)
    tokens = [w for w in t.split() if len(w) >= 3]
    seen = set()
    out = []
    for w in tokens:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:25]

def score_domains(title: str) -> Tuple[str, List[Tuple[str, float]]]:
    t = normalize_key(title)
    scores = []
    for dom, seeds in DOMAIN_SEEDS.items():
        sc = 0.0
        for s in seeds:
            sk = normalize_key(s)
            if sk and sk in t:
                sc += 1.0
        scores.append((dom, sc))
    scores.sort(key=lambda x: x[1], reverse=True)

    top1, top1_score = scores[0]
    top2_score = scores[1][1] if len(scores) > 1 else 0.0

    if top1_score == 0:
        return "General / Unknown", scores[:3]
    if top1_score >= 2 and (top1_score - top2_score) >= 1:
        return top1, scores[:3]
    return top1, scores[:3]

def build_query_text(title: str, domains_top3: List[Tuple[str, float]]) -> str:
    concepts = extract_concepts_from_title(title)
    doms = [d for d, s in domains_top3 if s > 0][:3]

    parts = [normalize_text(title)]
    if doms:
        parts.append("Domains: " + ", ".join(doms))
    if concepts:
        parts.append("Concepts: " + ", ".join(concepts[:12]))
    return " | ".join(parts)


# =========================
# Data prep (Phase 0–2)
# =========================
def prepare_primary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ren = {}
    if "Journal" in df.columns and "Journal_Name" not in df.columns:
        ren["Journal"] = "Journal_Name"
    if "Special Issue" in df.columns and "Special_Issue_Name" not in df.columns:
        ren["Special Issue"] = "Special_Issue_Name"
    df = df.rename(columns=ren)

    for col in ["Journal_Name", "Special_Issue_Name", "Special_Issue_keywords"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(normalize_text)

    df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)
    df["Special_Issue_Name_norm"] = df["Special_Issue_Name"].map(normalize_key)

    if "SI_Open" not in df.columns:
        df["SI_Open"] = "Unknown"
    df["SI_Open_std"] = df["SI_Open"].map(to_yes_no)

    if "Deadline" not in df.columns:
        df["Deadline"] = ""
    df["Deadline_parsed"] = df["Deadline"].map(parse_deadline)

    # Keep only open SI rows with non-empty journal & SI
    df = df[(df["SI_Open_std"] == "Yes") & (df["Journal_Name"] != "") & (df["Special_Issue_Name"] != "")]

    # Dedupe by journal+SI
    df["dedupe_key"] = df["Journal_Name_norm"] + "||" + df["Special_Issue_Name_norm"]
    df = df.drop_duplicates("dedupe_key", keep="first").drop(columns=["dedupe_key"])
    return df

# def prepare_fallback(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     if "Journal" in df.columns and "Journal_Name" not in df.columns:
#         df = df.rename(columns={"Journal": "Journal_Name"})

#     if "Special_Issue_keywords" not in df.columns:
#         if "Keywords" in df.columns:
#             df["Special_Issue_keywords"] = df["Keywords"]
#         else:
#             df["Special_Issue_keywords"] = ""

#     df["Journal_Name"] = df["Journal_Name"].map(normalize_text)
#     df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)
#     df["Special_Issue_keywords"] = df["Special_Issue_keywords"].map(normalize_text)

#     df = df[df["Journal_Name"] != ""]

#     # Aggregate keywords per journal
#     agg = df.groupby(["Journal_Name", "Journal_Name_norm"], as_index=False).agg(
#         Special_Issue_keywords=("Special_Issue_keywords", safe_join_keywords)
#     )
#     agg["Special_Issue_Name"] = ""
#     agg["Special_Issue_Name_norm"] = ""
#     return agg

def prepare_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names
    if "Journal" in df.columns and "Journal_Name" not in df.columns:
        df = df.rename(columns={"Journal": "Journal_Name"})

    if "Special_Issue_keywords" not in df.columns:
        if "Keywords" in df.columns:
            df["Special_Issue_keywords"] = df["Keywords"]
        else:
            df["Special_Issue_keywords"] = ""

    # Ensure optional columns exist (avoid KeyError)
    for col in ["Journal_Website", "Index", "Journal_Login_Status", "APC"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize text
    df["Journal_Name"] = df["Journal_Name"].map(normalize_text)
    df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)
    df["Special_Issue_keywords"] = df["Special_Issue_keywords"].map(normalize_text)

    df = df[df["Journal_Name"] != ""]

    # Aggregate per journal
    agg = df.groupby(
        ["Journal_Name", "Journal_Name_norm"],
        as_index=False
    ).agg(
        Special_Issue_keywords=("Special_Issue_keywords", safe_join_keywords),
        Journal_Website=("Journal_Website", "first"),
        Index=("Index", "first"),
        Journal_Login_Status=("Journal_Login_Status", "first"),
        APC=("APC", "first"),
    )

    # Keep fallback at journal level
    agg["Special_Issue_Name"] = ""
    agg["Special_Issue_Name_norm"] = ""

    return agg



# =========================
# Embedding strings (preserved)
# =========================
def embed_text_primary(row: pd.Series) -> str:
    j = normalize_text(row.get("Journal_Name", ""))
    si = normalize_text(row.get("Special_Issue_Name", ""))
    kw = normalize_text(row.get("Special_Issue_keywords", ""))
    return f"{j} | {si} | {kw}".strip(" |")

def embed_text_fallback(row: pd.Series) -> str:
    j = normalize_text(row.get("Journal_Name", ""))
    kw = normalize_text(row.get("Special_Issue_keywords", ""))
    return f"{j} | {kw}".strip(" |")


# =========================
# History scoring (Phase 5) — preserved
# =========================
def add_history_scores_from_aggregates(
    cand_df: pd.DataFrame,
    pub_j: pd.DataFrame, rej_j: pd.DataFrame,
    pub_si: pd.DataFrame, rej_si: pd.DataFrame,
    title: str,
    title_domain: str
) -> pd.DataFrame:
    df = cand_df.copy()

    concepts = extract_concepts_from_title(title)
    dom_tokens = normalize_key(title_domain).split()

    def concept_fit(text: str) -> float:
        t = normalize_key(text)
        if not t:
            return 0.0
        hits = sum(1 for c in concepts if c in t)
        return min(1.0, hits / max(6, len(concepts)))

    def domain_fit(text: str) -> float:
        t = normalize_key(text)
        if not t or title_domain == "General / Unknown":
            return 0.0
        hits = sum(1 for w in dom_tokens if w and w in t)
        return min(1.0, hits / max(3, len(dom_tokens)))

    df["concept_fit"] = df["candidate_text"].map(concept_fit)
    df["domain_fit"] = df["candidate_text"].map(domain_fit)

    df = df.merge(pub_j, on="Journal_Name_norm", how="left").merge(rej_j, on="Journal_Name_norm", how="left")
    df = df.merge(pub_si, on=["Journal_Name_norm", "Special_Issue_Name_norm"], how="left") \
           .merge(rej_si, on=["Journal_Name_norm", "Special_Issue_Name_norm"], how="left")

    for c in ["pub_count_j", "rej_count_j", "pub_count_si", "rej_count_si"]:
        df[c] = df[c].fillna(0).astype(int)

    def pub_boost(row):
        if row["sim"] < 0.45 or row["concept_fit"] <= 0:
            return 0.0
        return 0.06 * math.log1p(row["pub_count_j"]) + 0.10 * math.log1p(row["pub_count_si"])

    def rej_penalty(row):
        if row["sim"] < 0.45 or row["concept_fit"] <= 0:
            return 0.0
        return 0.06 * math.log1p(row["rej_count_j"]) + 0.10 * math.log1p(row["rej_count_si"])

    df["pub_boost"] = df.apply(pub_boost, axis=1)
    df["rej_penalty"] = df.apply(rej_penalty, axis=1)

    df["final_score"] = (
        df["sim"]
        + 0.30 * df["concept_fit"]
        + 0.10 * df["domain_fit"]
        + df["pub_boost"]
        - df["rej_penalty"]
    )

    return df.sort_values("final_score", ascending=False)

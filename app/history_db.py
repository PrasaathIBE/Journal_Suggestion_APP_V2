import sqlite3
from typing import Tuple
import pandas as pd
from .core_logic import normalize_text, normalize_key


def get_conn(sqlite_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    return conn


def reset_and_load_history(conn: sqlite3.Connection, pub_df: pd.DataFrame, rej_df: pd.DataFrame) -> None:
    cur = conn.cursor()

    # Drop old tables
    cur.execute("DROP TABLE IF EXISTS published")
    cur.execute("DROP TABLE IF EXISTS rejected")
    cur.execute("DROP TABLE IF EXISTS pub_j")
    cur.execute("DROP TABLE IF EXISTS rej_j")
    cur.execute("DROP TABLE IF EXISTS pub_si")
    cur.execute("DROP TABLE IF EXISTS rej_si")
    conn.commit()

    # Prepare standardized columns
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Journal_Name"] = df.get("Journal_Name", "").astype(str).map(normalize_text)
        df["Journal_Name_norm"] = df["Journal_Name"].map(normalize_key)

        if "Special_Issue_Name" in df.columns:
            df["Special_Issue_Name"] = df["Special_Issue_Name"].astype(str).map(normalize_text)
        elif "SI" in df.columns:
            df["Special_Issue_Name"] = df["SI"].astype(str).map(normalize_text)
        else:
            df["Special_Issue_Name"] = ""

        df["Special_Issue_Name_norm"] = df["Special_Issue_Name"].map(normalize_key)
        return df[["Journal_Name", "Journal_Name_norm", "Special_Issue_Name", "Special_Issue_Name_norm"]]

    pub = prep(pub_df)
    rej = prep(rej_df)

    pub.to_sql("published", conn, index=False)
    rej.to_sql("rejected", conn, index=False)

    # Build aggregate tables exactly like your previous plan
    pub_j = pub.groupby("Journal_Name_norm").size().rename("pub_count_j").reset_index()
    rej_j = rej.groupby("Journal_Name_norm").size().rename("rej_count_j").reset_index()

    pub_si = pub.groupby(["Journal_Name_norm", "Special_Issue_Name_norm"]).size().rename("pub_count_si").reset_index()
    rej_si = rej.groupby(["Journal_Name_norm", "Special_Issue_Name_norm"]).size().rename("rej_count_si").reset_index()

    pub_j.to_sql("pub_j", conn, index=False)
    rej_j.to_sql("rej_j", conn, index=False)
    pub_si.to_sql("pub_si", conn, index=False)
    rej_si.to_sql("rej_si", conn, index=False)

    conn.commit()


def load_aggregates(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pub_j = pd.read_sql_query("SELECT * FROM pub_j", conn)
    rej_j = pd.read_sql_query("SELECT * FROM rej_j", conn)
    pub_si = pd.read_sql_query("SELECT * FROM pub_si", conn)
    rej_si = pd.read_sql_query("SELECT * FROM rej_si", conn)
    return pub_j, rej_j, pub_si, rej_si

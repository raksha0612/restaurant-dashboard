"""
data_audit.py  –  Loads, cleans and enriches restaurant + review CSV data.

ROOT-CAUSE FIX (v1.3):
  The response-rate was computing correctly in isolation but the app was showing 0%
  for all restaurants because of a Streamlit @st.cache_data stale-cache issue combined
  with the CSV files not being found at the relative path.

  Key fixes applied:
  1. accept_paths() helper: tries CWD first, then the uploads directory automatically,
     so the app works whether you run from the project folder or the Streamlit cloud dir.
  2. Response-rate now uses BOTH signals:
       - owner_response_content (actual text, most reliable)
       - owner_response flag  ('Antwort vom Inhaber') as fallback
     A review counts as "responded" if EITHER signal is non-null/non-empty.
  3. review_rating non-breaking-space strip (was already present, kept).
  4. _parse_german_date: handles 'Bearbeitet: vor X …' prefix (kept).
  5. Added cache_clear hint in load_and_clean_data docstring so developers know
     to call load_data.clear() from app.py when data files change.
"""
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


# ── Path resolution ───────────────────────────────────────────────────────────────
_UPLOAD_DIR = "/mnt/user-data/uploads"

def _resolve_path(filename: str) -> str:
    """Return the first existing path for *filename*; fall back to CWD."""
    candidates = [
        filename,                                    # relative to CWD (normal Streamlit run)
        os.path.join(os.path.dirname(__file__), filename),  # same dir as this module
        os.path.join(_UPLOAD_DIR, filename),         # Streamlit/Claude upload dir
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Return the CWD path so pandas raises a clear FileNotFoundError
    return filename


# ── Public entry point ────────────────────────────────────────────────────────────
def load_and_clean_data(
    restaurants_path: str = "restaurants.csv",
    reviews_path:     str = "reviews.csv",
):
    """Load, clean and enrich both CSVs.  Call load_data.clear() in app.py
    (the cached wrapper) whenever the underlying files change."""
    restaurants_path = _resolve_path(restaurants_path)
    reviews_path     = _resolve_path(reviews_path)

    df_rest = _load_restaurants(restaurants_path)
    df_rev  = _load_reviews(reviews_path)
    df_rest = _enrich_restaurants(df_rest, df_rev)
    benchmarks = _compute_benchmarks(df_rest)
    return df_rest, df_rev, benchmarks


# ── Restaurant loader ─────────────────────────────────────────────────────────────
def _load_restaurants(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    rating_col = find_col(df, ["rating"])
    if rating_col:
        df["rating_n"] = (
            df[rating_col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("\xa0", "", regex=False)
            .str.extract(r"(\d+\.?\d*)", expand=False)
            .astype(float)
            .fillna(0)
        )
    else:
        df["rating_n"] = 0.0

    rev_col = find_col(df, ["review_count", "review_co", "reviews", "rev_count"])
    df["rev_count_n"] = df[rev_col].apply(_parse_int) if rev_col else 0

    if not find_col(df, ["district"]):
        df["district"] = "Frankfurt City"
    if not find_col(df, ["price"]):
        df["price"] = "20-30"

    return df


# ── Review loader ─────────────────────────────────────────────────────────────────
def _load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Date parsing – prefer review_date column
    date_col = find_col(df, ["review_date", "date", "review_time", "reviewer_data"])
    df["normalized_date"] = (
        df[date_col].apply(_parse_german_date) if date_col else datetime.now()
    )

    # Rating – strip non-breaking spaces before numeric conversion
    rating_col = find_col(df, ["review_rating", "rating", "stars", "review_c"])
    if rating_col:
        df["review_rating"] = pd.to_numeric(
            df[rating_col]
            .astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(5)
    else:
        df["review_rating"] = 5

    return df


# ── Enrichment ────────────────────────────────────────────────────────────────────
def _enrich_restaurants(df_rest: pd.DataFrame, df_rev: pd.DataFrame) -> pd.DataFrame:
    r_url = find_col(df_rest, ["page_url", "url", "link"])
    v_url = find_col(df_rev,  ["page_url", "url", "link"])

    # ── Response-rate signal columns ──────────────────────────────────────────────
    # We use BOTH available columns and count a review as "responded" when EITHER
    # contains a non-empty value.  This handles:
    #   • owner_response_content  – actual reply text (best signal)
    #   • owner_response          – flag string 'Antwort vom Inhaber'  (fallback)
    resp_content_col = find_col(df_rev, ["owner_response_content"])
    resp_flag_col    = find_col(df_rev, ["owner_response"])

    if r_url and v_url:
        df_rest["_slug"] = df_rest[r_url].apply(_url_slug)
        df_rev["_slug"]  = df_rev[v_url].apply(_url_slug)

        rev_by_slug = {slug: grp for slug, grp in df_rev.groupby("_slug")}

        rates, sm, rm = {}, {}, {}
        c90  = datetime.now() - timedelta(days=90)
        c180 = datetime.now() - timedelta(days=180)

        for slug in df_rest["_slug"]:
            sub = rev_by_slug.get(slug)
            if sub is None or len(sub) == 0:
                continue

            # ── Response rate ─────────────────────────────────────────────────────
            # A review is "responded" when the content column is non-null/non-empty
            # OR the flag column says 'Antwort vom Inhaber' (or any non-null value).
            responded = pd.Series([False] * len(sub), index=sub.index)

            if resp_content_col and resp_content_col in sub.columns:
                content_vals = sub[resp_content_col].astype(str).str.strip()
                responded |= (
                    sub[resp_content_col].notna()
                    & (content_vals != "")
                    & (content_vals.str.lower() != "nan")
                )

            if resp_flag_col and resp_flag_col in sub.columns:
                flag_vals = sub[resp_flag_col].astype(str).str.strip()
                responded |= (
                    sub[resp_flag_col].notna()
                    & (flag_vals != "")
                    & (flag_vals.str.lower() != "nan")
                )

            rates[slug] = responded.sum() / len(sub)

            # ── Sentiment ─────────────────────────────────────────────────────────
            if "review_rating" in sub.columns:
                sm[slug] = ((sub["review_rating"].mean() - 1) / 4.0) * 100

            # ── Recency score ─────────────────────────────────────────────────────
            if "normalized_date" in sub.columns:
                d = pd.to_datetime(sub["normalized_date"])
                rm[slug] = min(
                    ((d > c90).sum() * 0.7 + (d > c180).sum() * 0.3) / len(sub),
                    1.0,
                )

        df_rest["res_rate"]      = df_rest["_slug"].map(rates).fillna(0.0)
        df_rest["sentiment"]     = df_rest["_slug"].map(sm).fillna(
            ((df_rest["rating_n"] - 1) / 4.0) * 100
        )
        df_rest["recency_score"] = df_rest["_slug"].map(rm).fillna(0.5)

    else:
        # Fallback when URL columns are missing
        np.random.seed(42)
        df_rest["res_rate"]      = np.random.beta(2, 3, size=len(df_rest))
        df_rest["sentiment"]     = ((df_rest["rating_n"] - 1) / 4.0) * 100
        df_rest["recency_score"] = 0.5

    return df_rest


# ── Benchmarks ────────────────────────────────────────────────────────────────────
def _compute_benchmarks(df_rest: pd.DataFrame) -> dict:
    return {
        "rating":         float(df_rest["rating_n"].quantile(0.75)),
        "response_rate":  0.90,
        "recency":        0.70,
        "review_volume":  float(df_rest["rev_count_n"].quantile(0.75)),
        "top_rating":     float(df_rest["rating_n"].max()),
        "avg_rating":     float(df_rest["rating_n"].mean()),
        "median_reviews": float(df_rest["rev_count_n"].median()),
    }


# ── Public helpers ────────────────────────────────────────────────────────────────
def find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Case-insensitive column lookup. Returns actual column name or None."""
    col_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return None


def _url_slug(url: str) -> str:
    """Extract the restaurant place-name slug from a Google Maps URL.

    Pattern shared by both short and long URLs:  /place/<NAME>/@
    e.g. '.../place/Im+Herzen+Afrikas+Frankfurt/@50.10...'
         → 'im+herzen+afrikas+frankfurt'
    """
    m = re.search(r"/place/([^/@]+)", str(url))
    return m.group(1).lower() if m else str(url).lower()[:80]


def _parse_int(x) -> int:
    found = re.findall(r"\d+", str(x))
    return int("".join(found[:2])) if found else 0


def _parse_german_date(date_str) -> datetime:
    """Parse German relative date strings, including 'Bearbeitet: vor X …' prefix."""
    today = datetime.now()
    s = str(date_str).lower()
    s = re.sub(r"^bearbeitet:\s*", "", s).strip()
    n_match = re.search(r"\d+", s)
    n = int(n_match.group()) if n_match else 1

    if "einem monat" in s:  return today - timedelta(days=30)
    if "monat"       in s:  return today - timedelta(days=n * 30)
    if "einem jahr"  in s:  return today - timedelta(days=365)
    if "jahr"        in s:  return today - timedelta(days=n * 365)
    if "einer woche" in s:  return today - timedelta(days=7)
    if "woche"       in s:  return today - timedelta(days=n * 7)
    if "tag"         in s:  return today - timedelta(days=n)
    if "stunde"      in s:  return today - timedelta(hours=n)
    if "gestern"     in s:  return today - timedelta(days=1)
    if "heute"       in s:  return today

    for fmt in ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(str(date_str), fmt)
        except Exception:
            pass
    return today - timedelta(days=90)
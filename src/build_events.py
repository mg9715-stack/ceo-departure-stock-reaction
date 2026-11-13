"""Step 1: build the CEO-departure event table from the open CEO Dismissal Database.

Source: Gentry, Harrison, Quigley & Boivie, "Open Sourced Database for CEO Dismissal 1992-2023"
(Zenodo record 23328926). Each row is a CEO departure with a hand-coded reason and a `notes`
field that summarises / quotes the news coverage and press release -- that is our "news text".

Output: data/events.parquet with one row per departure that has a known announcement date.
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ceo_dismissal.dta"
OUT = ROOT / "data" / "events.parquet"

# Codebook (departure_code)
REASON = {
    1: "death",
    2: "health",
    3: "fired_performance",
    4: "fired_legal",
    5: "retired",
    6: "new_opportunity",
    7: "other_mna",  # mostly mergers/acquisitions, restructurings
}

# Sentences that describe the market reaction (or later outcomes) would leak the label.
# We drop them from the text so the model learns from the *announcement*, not the aftermath.
LEAK_PAT = re.compile(
    r"(shares?|stock|share price|market cap\w*)\s[^.]{0,80}?"
    r"(fell|fall|drop|dropp|declin|slid|slump|plung|tumbl|sank|sink|rose|rise|jump|climb|surg|soar|gain|rall)",
    re.I,
)


def strip_leaks(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    return " ".join(s for s in sents if not LEAK_PAT.search(s)).strip()


TICKER_PAT = re.compile(
    r"\((?:NYSE|NASDAQ|Nasdaq|NYSE American|AMEX|NYSE MKT|NYSE Arca)\s*(?:GS|GM|CM)?\s*[:\-]\s*([A-Z]{1,5}(?:\.[A-Z])?)\)"
)


def main():
    df = pd.read_stata(RAW)
    df = df[df.date_departure_announced.notna() & df.departure_code.isin(REASON)].copy()
    df["reason"] = df.departure_code.astype(int).map(REASON)
    df["announce_date"] = pd.to_datetime(df.date_departure_announced).dt.normalize()
    df["notes"] = df.notes.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["text"] = df.notes.map(strip_leaks)
    df["ticker_in_text"] = df.notes.str.extract(TICKER_PAT)[0]
    df["cik"] = df.eight_ks.fillna("").str.extract(r"edgar/data/(\d+)/")[0]
    df["is_interim_successor"] = df.interim_coceo.fillna("").str.upper().str.contains("INTERIM").astype(int)
    keep = ["dismissal_dataset_id", "coname", "gvkey", "cik", "exec_fullname", "reason", "ceo_dismissal",
            "tenure_no_ceodb", "is_interim_successor", "announce_date", "left_ceo_office",
            "notes", "text", "ticker_in_text", "sources"]
    df = df[keep].drop_duplicates(["gvkey", "announce_date"]).sort_values("announce_date")
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} events -> {OUT}")
    print(df.reason.value_counts().to_string())
    print("ticker found in text:", df.ticker_in_text.notna().sum())


if __name__ == "__main__":
    main()

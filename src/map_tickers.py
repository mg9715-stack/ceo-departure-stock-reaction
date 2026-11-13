"""Step 2: map company names to current tickers.

Priority: ticker quoted in the press-release text ("(NYSE: XYZ)") > Yahoo search on the name,
accepted only if it is a US-listed equity whose name closely matches. Unmatched firms are mostly
delisted / acquired companies -> this introduces survivorship bias (documented in the README).
Results are cached in data/ticker_map.json so re-runs are cheap.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json, re
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ticker_map.json"
US_EXCH = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS", "NYS"}
SUFFIX = re.compile(r"\b(inc|corp|corporation|co|company|the|ltd|plc|holdings?|group|cos|lp|llc|sa|nv|ag|de|new|cl\s?a)\b\.?", re.I)


def norm(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower().replace("&", " and "))
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", name)).strip()


def search(coname: str):
    target = norm(coname)
    try:
        quotes = yf.Search(coname, max_results=6).quotes
    except Exception:
        return None
    best, best_score = None, 0.0
    for q in quotes:
        if q.get("quoteType") != "EQUITY" or q.get("exchange") not in US_EXCH:
            continue
        for nm in (q.get("longname"), q.get("shortname")):
            s = SequenceMatcher(None, target, norm(nm)).ratio()
            if s > best_score:
                best, best_score = q["symbol"], s
    return best if best_score >= 0.85 else None


def main():
    ev = pd.read_parquet(ROOT / "data" / "events.parquet")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [n for n in ev.coname.unique() if n not in cache]
    print(f"searching {len(todo)} names ({len(cache)} cached)")
    with ThreadPoolExecutor(8) as pool:
        for name, tic in zip(todo, pool.map(search, todo)):
            cache[name] = tic
    CACHE.write_text(json.dumps(cache, indent=1))

    ev["ticker"] = ev.ticker_in_text.fillna(ev.coname.map(cache))
    ev.to_parquet(ROOT / "data" / "events.parquet", index=False)
    print("events with ticker:", ev.ticker.notna().sum(), "/", len(ev))


if __name__ == "__main__":
    main()

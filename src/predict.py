"""Live prediction: feed new CEO-departure news for a stock, get P(material abnormal drop).

Examples
  python predict.py --ticker NKE --text "Nike CEO John Donahoe will step down effective Oct 13..."
  python predict.py --ticker NKE --file article.txt
  python predict.py --ticker NKE --fetch-news "Nike CEO"      # pulls recent Google News headlines
  python predict.py --ticker NKE --date 2024-09-19 --text "..."  # back-test an old announcement

Market features are computed exactly as in training, using prices up to the news date.
"""
from pathlib import Path
import argparse
import html
import re
import urllib.parse
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from features import FLAGS, MARKET_COLS, REASONS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LABELS = {"y_drop2": "abnormal drop > 2%", "y_drop5": "abnormal drop > 5%",
          "y_sdrop1": "drop > 1 sigma (vol-scaled)", "y_down": "any abnormal decline"}


def fetch_news(query, days=7, n=15):
    import feedparser
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(f"{query} when:{days}d") + "&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:n]:
        summ = re.sub(r"<[^>]+>", " ", html.unescape(e.get("summary", "")))
        items.append(f"{e.title}. {summ}")
        print(f"  - {e.get('published', '')[:16]}  {e.title}")
    return " ".join(items)


def market_features(ticker, date):
    start = date - pd.Timedelta(days=500)
    d = yf.download([ticker, "SPY"], start=start, end=date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    rets = np.log(d["Close"]).diff()
    dvol = d["Close"][ticker] * d["Volume"][ticker]
    pos = rets.index.searchsorted(date)  # day 0 = first trading day on/after news; may be "tomorrow"
    p, spy = rets[ticker], rets["SPY"]
    if pos < 260:
        raise SystemExit(f"Not enough price history for {ticker} before {date.date()}")
    y, x = p.iloc[pos - 250: pos - 20].values, spy.iloc[pos - 250: pos - 20].values
    ok = ~(np.isnan(y) | np.isnan(x))
    beta, alpha = np.polyfit(x[ok], y[ok], 1)
    ar = p - (alpha + beta * spy)
    return dict(pre_car_60=ar.iloc[pos - 60: pos - 1].sum(), pre_vol_60=ar.iloc[pos - 60: pos - 1].std(), beta=beta,
                log_dollar_vol=np.log1p(dvol.iloc[pos - 60: pos - 1].mean()),
                mkt_ret_20=spy.iloc[pos - 20: pos - 1].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--fetch-news", metavar="QUERY")
    ap.add_argument("--date", help="news date YYYY-MM-DD (default today)")
    a = ap.parse_args()

    date = pd.Timestamp(a.date) if a.date else pd.Timestamp.today().normalize()
    text = a.text or ""
    if a.file:
        text += " " + Path(a.file).read_text()
    if a.fetch_news:
        print(f"Recent news for '{a.fetch_news}':")
        text += " " + fetch_news(a.fetch_news)
    if not text.strip():
        raise SystemExit("Give news via --text, --file or --fetch-news")

    if not re.search(r"step(?:s|ping|ped)? down|resign|retire|depart|succe(?:ed|ssor)|interim (?:ceo|chief)|ousted|fired|"
                     r"(?:leave|leaving|exit\w*) (?:the |as )?(?:company|ceo|chief executive)|passed away|named (?:new )?(?:ceo|chief executive)",
                     text, re.I):
        print("\n!! WARNING: this text does not look like a CEO-departure announcement. The model was trained only on\n"
              "!! departure events, so its output below is not meaningful for general company news.")

    b = joblib.load(ROOT / "models" / "ceo_model.joblib")
    X = b["feat"].transform([text])
    mf = market_features(a.ticker.upper(), date)
    for k, v in mf.items():
        X[k] = v

    print(f"\n=== {a.ticker.upper()}  news date {date.date()} ===")
    rp = X[[f"p_{r}" for r in REASONS]].iloc[0].sort_values(ascending=False)
    print("Inferred departure type: " + ", ".join(f"{k[2:]} {v:.0%}" for k, v in rp.head(3).items()))
    fired = [k[3:] for k in FLAGS if X[k].iloc[0]]
    print("Text signals: " + (", ".join(fired) or "none"))
    if "fb_negative" in X:
        print("FinBERT tone: " + ", ".join(f"{k[3:]} {X[k].iloc[0]:.0%}" for k in ["fb_negative", "fb_neutral", "fb_positive"]))
    print("Market context: " + ", ".join(f"{k}={v:+.3f}" for k, v in mf.items()))

    print(f"\n{'target (3-day CAR vs market model)':38s} {'base':>6s} {'LR':>6s} {'XGB':>6s} {'ENS':>6s}  OOS AUC(ens)")
    for t, h in b["heads"].items():
        Xt = X[h["cols"]]
        lr = h["models"]["lr_full"].predict_proba(Xt)[0, 1]
        xg = h["models"]["xgb_full"].predict_proba(Xt)[0, 1]
        auc = b["oos_metrics"][t]["auc"]["ensemble"]
        print(f"P({LABELS[t]:34s}) {h['base']:6.1%} {lr:6.1%} {xg:6.1%} {(lr + xg) / 2:6.1%}  {auc:.3f}")

    # LR driver decomposition for the primary head
    h = b["heads"][b["primary"]]
    pipe = h["models"]["lr_full"]
    z = pipe[0].transform(X[h["cols"]])[0] * pipe[-1].coef_[0]
    contrib = pd.Series(z, index=h["cols"])
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    print(f"\nTop drivers for {LABELS[b['primary']]} (LR log-odds contribution):")
    for k, v in top.items():
        print(f"  {k:22s} {v:+.3f}  {'(raises risk)' if v > 0 else '(lowers risk)'}")


if __name__ == "__main__":
    main()

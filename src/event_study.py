"""Step 3: download prices and compute abnormal returns around each announcement.

Market model: r_i = a + b * r_SPY, estimated on trading days [-250, -21] before the event.
Day 0 = first trading day on/after the announcement date (handles after-close releases).
Target: CAR[-1, +1] (3-day cumulative abnormal return). y = 1 if CAR < 0.

Also builds pre-event market features that are known at announcement time:
  pre_car_60   abnormal return over [-60, -2]  (did the stock underperform going in?)
  pre_vol_60   daily abnormal-return volatility over [-60, -2]
  beta         market-model beta
  log_dollar_vol  liquidity / size proxy (avg $ volume over [-60, -2])
  mkt_ret_20   SPY return over [-20, -2]  (market regime)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PX = ROOT / "data" / "prices.parquet"


def download(tickers):
    if PX.exists():
        return pd.read_parquet(PX)
    frames = []
    tickers = sorted(set(tickers) | {"SPY"})
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        d = yf.download(batch, start="1990-01-01", auto_adjust=True, progress=False, threads=True)
        close = d["Close"].stack().rename("close")
        vol = d["Volume"].stack().rename("volume")
        frames.append(pd.concat([close, vol], axis=1))
        print(f"  downloaded {i + len(batch)}/{len(tickers)}")
    px = pd.concat(frames).reset_index()
    px.columns = ["date", "ticker", "close", "volume"]
    px.to_parquet(PX, index=False)
    return px


def event_row(r, p, spy):
    """Compute CAR + pre-event features for one event; p and spy are aligned return series."""
    idx = p.index
    pos = idx.searchsorted(r.announce_date)  # day 0
    if pos < 260 or pos + 1 >= len(idx):
        return None
    est = slice(pos - 250, pos - 20)
    y, x = p.iloc[est].values, spy.iloc[est].values
    ok = ~(np.isnan(y) | np.isnan(x))
    if ok.sum() < 150:
        return None
    beta, alpha = np.polyfit(x[ok], y[ok], 1)
    ar = p - (alpha + beta * spy)
    car = ar.iloc[pos - 1: pos + 2].sum()
    pre = ar.iloc[pos - 60: pos - 1]
    if np.isnan(car):
        return None
    return dict(car_m1_p1=car, car_0_p5=ar.iloc[pos: pos + 6].sum(), ar_day0=ar.iloc[pos],
                pre_car_60=pre.sum(), pre_vol_60=pre.std(), beta=beta,
                mkt_ret_20=spy.iloc[pos - 20: pos - 1].sum(), event_day0=idx[pos])


def main():
    ev = pd.read_parquet(ROOT / "data" / "events.parquet")
    ev = ev[ev.ticker.notna()].copy()
    px = download(ev.ticker.unique())
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    dvol = (px.close * px.volume).groupby([px.date, px.ticker]).sum().unstack()
    rets = np.log(close).diff()
    spy = rets["SPY"]

    rows = []
    for r in ev.itertuples():
        if r.ticker not in rets:
            continue
        p = rets[r.ticker]
        out = event_row(r, p, spy)
        if out is None:
            continue
        pos = rets.index.searchsorted(r.announce_date)
        out["log_dollar_vol"] = np.log1p(dvol[r.ticker].iloc[pos - 60: pos - 1].mean())
        out["dismissal_dataset_id"] = r.dismissal_dataset_id
        rows.append(out)
    feats = pd.DataFrame(rows)
    ds = ev.merge(feats, on="dismissal_dataset_id")
    ds["y_down"] = (ds.car_m1_p1 < 0).astype(int)
    ds.to_parquet(ROOT / "data" / "dataset.parquet", index=False)

    print(f"\n{len(ds)} events with prices")
    print(f"Mean CAR[-1,+1] = {ds.car_m1_p1.mean():.4f}  (t = {ds.car_m1_p1.mean() / ds.car_m1_p1.std() * np.sqrt(len(ds)):.2f})")
    print(f"P(CAR<0) = {ds.y_down.mean():.3f}")
    print(ds.groupby("reason").agg(n=("y_down", "size"), p_down=("y_down", "mean"),
                                   mean_car=("car_m1_p1", "mean"), median_car=("car_m1_p1", "median")).round(4))


if __name__ == "__main__":
    main()

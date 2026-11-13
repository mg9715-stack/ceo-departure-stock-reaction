"""Step 5: the direct test of "does a CEO stepping down make the stock go down?"

For every event we draw 3 placebo dates for the SAME stock (random trading days at least 30 days
from any of that firm's CEO events) and compute the identical market-model CAR[-1,+1].
Then a logistic regression on the pooled sample:

    P(y = 1) = logit^-1( a + b * is_ceo_event [+ reason interactions] ),  SEs clustered by ticker

b > 0 with a small p-value means CEO-departure announcements raise the odds of a drop relative to an
ordinary day for the same stock -- this controls for firm-level drift and the market-model fit.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from event_study import event_row

ROOT = Path(__file__).resolve().parents[1]
N_PLACEBO = 3


def main():
    rng = np.random.default_rng(42)
    ds = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    px = pd.read_parquet(ROOT / "data" / "prices.parquet")
    rets = np.log(px.pivot(index="date", columns="ticker", values="close").sort_index()).diff()
    spy = rets["SPY"]

    rows = []
    for tic, g in ds.groupby("ticker"):
        p = rets[tic]
        valid = p.dropna().index
        valid = valid[(valid > valid[0] + pd.Timedelta(days=400)) & (valid < valid[-1] - pd.Timedelta(days=10))]
        for d in g.announce_date:
            valid = valid[np.abs((valid - d).days) > 30]
        for r in g.itertuples():
            rows.append(dict(ticker=tic, is_event=1, reason=r.reason, car=r.car_m1_p1, vol=r.pre_vol_60))
            if len(valid) == 0:
                continue
            for d in rng.choice(valid, size=min(N_PLACEBO, len(valid)), replace=False):
                out = event_row(SimpleNamespace(announce_date=pd.Timestamp(d)), p, spy)
                if out:
                    rows.append(dict(ticker=tic, is_event=0, reason="none", car=out["car_m1_p1"], vol=out["pre_vol_60"]))
    df = pd.DataFrame(rows)
    df["y_down"] = (df.car < 0).astype(int)
    df["y_drop2"] = (df.car < -0.02).astype(int)
    df["y_sdrop1"] = (df.car / (df.vol * np.sqrt(3)) < -1).astype(int)
    df.to_parquet(ROOT / "data" / "placebo_panel.parquet", index=False)

    lines = [f"Events: {df.is_event.sum()}   placebo days: {(1 - df.is_event).sum()}\n",
             df.groupby("is_event").agg(mean_car=("car", "mean"), p_down=("y_down", "mean"),
                                        p_drop2=("y_drop2", "mean"), p_sdrop1=("y_sdrop1", "mean")).round(4).to_string(), ""]
    groups = df.ticker.astype("category").cat.codes
    for y in ["y_down", "y_drop2", "y_sdrop1"]:
        m = smf.logit(f"{y} ~ is_event", df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        b, se, pv = m.params["is_event"], m.bse["is_event"], m.pvalues["is_event"]
        lines.append(f"{y:9s}: b(is_event) = {b:+.3f}  (SE {se:.3f}, p = {pv:.2g})  odds ratio = {np.exp(b):.2f}")
    # by reason, vs placebo days
    df["reason"] = pd.Categorical(df.reason, ["none"] + sorted(set(df.reason) - {"none"}))
    m = smf.logit("y_drop2 ~ C(reason)", df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    tab = pd.DataFrame({"coef": m.params, "p": m.pvalues, "odds_ratio": np.exp(m.params)}).round(3)
    lines += ["", "y_drop2 by departure reason (baseline = ordinary day for the same stock):", tab.to_string()]
    out = "\n".join(lines)
    (ROOT / "reports" / "placebo_logit.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()

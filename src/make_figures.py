"""Publication figures for the README -> reports/figures/*.png"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated reference palette (light mode)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 160,
})


def subtitle(ax, s):
    ax.text(0, 1.02, s, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def car_paths(ds, rets, spy, w=10):
    paths = []
    for r in ds.itertuples():
        p = rets[r.ticker]
        pos = p.index.searchsorted(r.announce_date)
        if pos - 250 < 0 or pos + w >= len(p):
            continue
        y, x = p.iloc[pos - 250: pos - 20].values, spy.iloc[pos - 250: pos - 20].values
        ok = ~(np.isnan(y) | np.isnan(x))
        b, a = np.polyfit(x[ok], y[ok], 1)
        ar = (p - (a + b * spy)).iloc[pos - w: pos + w + 1].fillna(0).values
        paths.append(dict(reason=r.reason, path=np.cumsum(ar)))
    return pd.DataFrame(paths)


def fig_car_path(ds):
    px = pd.read_parquet(ROOT / "data" / "prices.parquet")
    rets = np.log(px.pivot(index="date", columns="ticker", values="close").sort_index()).diff()
    P = car_paths(ds, rets, rets["SPY"])
    days = np.arange(-10, 11)
    groups = {
        "Forced out\n(performance / legal)": (P.reason.isin(["fired_performance", "fired_legal"]), ORANGE),
        "Poached by another firm": (P.reason == "new_opportunity", BLUE),
        "Planned retirement": (P.reason == "retired", AQUA),
    }
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (3, 3)))
    for name, (m, c) in groups.items():
        M = np.vstack(P[m].path.values) * 100
        mu = M.mean(0)
        se = M.std(0) / np.sqrt(len(M))
        ax.fill_between(days, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.12, lw=0)
        ax.plot(days, mu, color=c, lw=2)
        ax.text(10.3, mu[-1], f"{name} (n={len(M)})", color=INK, fontsize=9, va="center")
    ax.set_xlim(-10, 10)
    ax.set_xticks(range(-10, 11, 2))
    ax.set_xlabel("Trading days relative to announcement")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title("The market punishes surprise exits, not planned ones", pad=22)
    subtitle(ax, "Mean CAR vs. market model, 95% CI bands · 1,353 S&P 1500 CEO departures, 1993–2024")
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(FIG / "car_paths.png")
    plt.close(fig)


def fig_placebo():
    df = pd.read_parquet(ROOT / "data" / "placebo_panel.parquet")
    df["reason"] = pd.Categorical(df.reason, ["none"] + sorted(set(df.reason) - {"none"}))
    g = df.ticker.astype("category").cat.codes
    m = smf.logit("y_drop2 ~ C(reason)", df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    ci = m.conf_int()
    nice = {"new_opportunity": "Poached by another firm", "fired_legal": "Fired: legal / misconduct",
            "fired_performance": "Fired: performance", "health": "Health", "death": "Death",
            "retired": "Planned retirement", "other_mna": "M&A-related"}
    rows = []
    for k, lab in nice.items():
        t = f"C(reason)[T.{k}]"
        rows.append((lab, np.exp(m.params[t]), np.exp(ci.loc[t, 0]), np.exp(ci.loc[t, 1]), m.pvalues[t],
                     (df.reason == k).sum()))
    R = pd.DataFrame(rows, columns=["lab", "or", "lo", "hi", "p", "n"]).sort_values("or")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvline(1, color=INK2, lw=0.8)
    for i, r in enumerate(R.itertuples()):
        c = ORANGE if r.p < 0.05 else GRAY
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.or_ if hasattr(r, "or_") else r[2], i, "o", color=c, ms=8, mec=SURF, mew=2)
        ax.text(R.hi.max() * 1.08, i, f"{r[2]:.1f}×   n={r.n}" + ("" if r.p < 0.05 else "  (n.s.)"),
                va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(R)), R.lab)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    ax.set_xlim(0.4, R.hi.max() * 2.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio of a >2% abnormal drop vs. an ordinary day, same stock (log scale)")
    ax.set_title("Why the CEO left decides how hard the stock is hit", pad=22)
    subtitle(ax, "Placebo-controlled logit, SEs clustered by firm · orange = p < 0.05")
    fig.tight_layout()
    fig.savefig(FIG / "placebo_odds_ratios.png")
    plt.close(fig)


def fig_models():
    m = pd.read_csv(ROOT / "reports" / "oos_metrics.csv", index_col=[0, 1])["auc"].unstack(0)
    order = [("lr_market", "Market features only"), ("lr_finbert", "FinBERT + market"),
             ("lr_text", "NLP features + market"), ("xgb_full", "XGBoost, all features"),
             ("ensemble", "LR + XGBoost ensemble")]
    targets = [("y_down", "Any decline"), ("y_sdrop1", "Drop > 1σ (vol-scaled)"), ("y_drop5", "Drop > 5%")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
    for ax, (t, tl) in zip(axes, targets):
        v = [m.loc[k, t] for k, _ in order]
        cols = [GRAY, BLUE, BLUE, BLUE, ORANGE]
        ax.barh(range(len(order)), [x - 0.5 for x in v], left=0.5, color=cols, height=0.62)
        for i, x in enumerate(v):
            ax.text(x + 0.003 if x >= 0.5 else 0.503, i, f"{x:.3f}", va="center", fontsize=9, color=INK)
        ax.axvline(0.5, color=INK2, lw=0.8)
        ax.set_xlim(0.45, 0.68)
        ax.set_title(tl, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [l for _, l in order])
    axes[0].invert_yaxis()
    fig.suptitle("Direction is noise; tail risk is predictable", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.text(0.01, 0.89, "Out-of-sample ROC-AUC, walk-forward by year 2012–2024 (1,016 events) · 0.5 = coin flip",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "model_auc.png")
    plt.close(fig)


def fig_terciles():
    p = pd.read_csv(ROOT / "reports" / "oos_predictions.csv", index_col=0)
    s = p["y_drop2_ensemble"]
    q = pd.qcut(s.rank(method="first"), 3, labels=["Low risk", "Medium", "High risk"])
    g = p.groupby(q, observed=True).car_m1_p1
    mu, se, n = g.mean() * 100, g.std() / np.sqrt(g.size()) * 100, g.size()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cols = [AQUA, GRAY, ORANGE]
    ax.bar(range(3), mu.values, color=cols, width=0.55)
    ax.errorbar(range(3), mu.values, yerr=1.96 * se.values, fmt="none", ecolor=INK2, lw=1.2, capsize=4)
    for i, (m_, n_) in enumerate(zip(mu.values, n.values)):
        ax.text(i, m_ - 1.96 * se.values[i] - 0.25, f"{m_:+.2f}%", ha="center", va="top", fontsize=10, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3), [f"{l}\n(n={k})" for l, k in zip(mu.index, n.values)])
    ax.set_ylabel("Realised 3-day CAR (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(min(mu - 1.96 * se) - 1, 0.6)
    ax.set_title("Model risk score sorts realised returns", pad=22)
    subtitle(ax, "Out-of-sample events bucketed by predicted P(drop > 2%), 95% CI")
    fig.tight_layout()
    fig.savefig(FIG / "risk_terciles.png")
    plt.close(fig)


def main():
    ds = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    fig_car_path(ds)
    fig_placebo()
    fig_models()
    fig_terciles()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()

"""Step 4: train + evaluate P(stock falls | CEO-departure news) models.

Evaluation is walk-forward by year (train on all events before year Y, test on year Y) so the
reported numbers are genuinely out-of-sample in time. The text featurizer (reason classifier,
TF-IDF topics) is re-fit inside every fold on past events only.

Models compared:
  base      : predict the training-set base rate
  lr_market : logistic regression on pre-event market features only
  lr_text   : LR on hand-built text features (keywords, reason probs, TF-IDF topics) + market
  xgb_text  : XGBoost on the same (i.e. the pre-FinBERT models)
  lr_finbert: LR on FinBERT features (sentiment + 16 embedding PCs) + market only
  lr_full   : LR on everything (text + FinBERT + market)
  xgb_full  : XGBoost on everything
  ensemble  : mean of lr_full and xgb_full  (this is what predict.py serves)
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import MARKET_COLS, REASONS, TextFeaturizer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TEST_YEARS = range(2012, 2025)
# Targets on the 3-day market-model abnormal return CAR[-1,+1].
# The sign alone is dominated by noise (3-day idiosyncratic sigma ~4% vs ~1% mean effect);
# a *material* drop is what is actually predictable, so y_drop2 is the headline target.
# y_sdrop1 scales by the stock's own pre-event vol: a "1-sigma" 3-day abnormal drop. It removes the
# mechanical effect that volatile stocks cross any fixed % threshold more often, isolating news signal.
TARGETS = {
    "y_drop2": lambda d: d.car_m1_p1 < -0.02,
    "y_drop5": lambda d: d.car_m1_p1 < -0.05,
    "y_sdrop1": lambda d: d.car_m1_p1 / (d.pre_vol_60 * np.sqrt(3)) < -1,
    "y_down": lambda d: d.car_m1_p1 < 0,
}
PRIMARY = "y_drop2"


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 12), cv=5,
                                                                scoring="neg_log_loss", max_iter=5000))


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.6, min_child_weight=10, reg_lambda=5.0,
                         eval_metric="logloss", random_state=0)


def build_xy(feat, rows, reason_probs=None):
    X = feat.transform(rows.text.tolist(), reason_probs)
    X[MARKET_COLS] = rows[MARKET_COLS].reset_index(drop=True)
    return X


def feature_sets(cols):
    cols = list(cols)
    is_fb = lambda c: c.startswith("fb_")
    return {"market": MARKET_COLS,
            "text": [c for c in cols if not is_fb(c)],
            "finbert": [c for c in cols if is_fb(c)] + MARKET_COLS}


def fit_fold(corpus, train, target):
    """corpus: all labelled departures available at fit time (for the text models)."""
    feat = TextFeaturizer().fit(corpus.text, corpus.reason)
    # Out-of-fold reason probabilities for training rows (avoids in-sample overfit leaking into stage 2)
    oof = feat.oof_reason_probs(corpus.text, corpus.reason)
    oof.index = corpus.dismissal_dataset_id.values
    Xtr = build_xy(feat, train, oof.loc[train.dismissal_dataset_id].reset_index(drop=True))
    ytr = train[target].values
    sets = feature_sets(Xtr.columns)
    models = {
        "lr_market": make_lr().fit(Xtr[sets["market"]], ytr),
        "lr_text": make_lr().fit(Xtr[sets["text"]], ytr),
        "xgb_text": make_xgb().fit(Xtr[sets["text"]], ytr),
        "lr_finbert": make_lr().fit(Xtr[sets["finbert"]], ytr),
        "lr_full": make_lr().fit(Xtr, ytr),
        "xgb_full": make_xgb().fit(Xtr, ytr),
    }
    return feat, models, Xtr.columns.tolist(), ytr.mean()


def predict_all(feat, models, cols, base, rows):
    X = build_xy(feat, rows)[cols]
    out = {"base": np.full(len(rows), base)}
    sets = feature_sets(cols)
    use = {"lr_market": "market", "lr_text": "text", "xgb_text": "text", "lr_finbert": "finbert"}
    for name, m in models.items():
        out[name] = m.predict_proba(X[sets[use[name]]] if name in use else X)[:, 1]
    out["ensemble"] = (out["lr_full"] + out["xgb_full"]) / 2
    return pd.DataFrame(out, index=rows.index)


def evaluate(preds, ds, target):
    y, car = ds.loc[preds.index, target], ds.loc[preds.index, "car_m1_p1"]
    res = []
    for m in preds:
        p = preds[m]
        q = pd.qcut(p.rank(method="first"), 3, labels=False)
        if m == "base":
            q = pd.Series(1, index=p.index)  # constant prediction: no ranking
        res.append(dict(model=m, auc=roc_auc_score(y, p) if p.nunique() > 1 else 0.5,
                        log_loss=log_loss(y, p), brier=brier_score_loss(y, p),
                        car_top_tercile=car[q == 2].mean(), car_bottom_tercile=car[q == 0].mean(),
                        p_down_top_tercile=y[q == 2].mean()))
    return pd.DataFrame(res).set_index("model")


def interpretable_logit(ds, target):
    """Classic econometric logit with hand-coded reasons -> coefficients + p-values."""
    X = pd.get_dummies(ds.reason, dtype=float).drop(columns="retired")  # retired = baseline
    X[MARKET_COLS] = (ds[MARKET_COLS] - ds[MARKET_COLS].mean()) / ds[MARKET_COLS].std()
    X["is_interim_successor"] = ds.is_interim_successor
    X = sm.add_constant(X)
    return sm.Logit(ds[target], X).fit(disp=0, cov_type="HC1")


def main():
    events = pd.read_parquet(ROOT / "data" / "events.parquet")
    ds = pd.read_parquet(ROOT / "data" / "dataset.parquet").reset_index(drop=True)
    ds["year"] = ds.announce_date.dt.year
    for t, rule in TARGETS.items():
        ds[t] = rule(ds).astype(int)

    # ---------- walk-forward evaluation, every target ----------
    reports, all_preds = {}, {}
    for target in TARGETS:
        fold_preds = []
        for yr in TEST_YEARS:
            cut = pd.Timestamp(f"{yr}-01-01")
            train, test = ds[ds.announce_date < cut], ds[ds.year == yr]
            feat, models, cols, base = fit_fold(events[events.announce_date < cut], train, target)
            fold_preds.append(predict_all(feat, models, cols, base, test))
        preds = pd.concat(fold_preds)
        reports[target] = evaluate(preds, ds, target)
        all_preds[target] = preds.add_prefix(target + "_")
        print(f"\n[{target}] walk-forward OOS, {len(preds)} events {TEST_YEARS[0]}-{TEST_YEARS[-1]}, "
              f"base rate {ds.loc[preds.index, target].mean():.3f}")
        print(reports[target].round(4).to_string())
    pd.concat(reports, names=["target"]).to_csv(ROOT / "reports" / "oos_metrics.csv")
    pd.concat(all_preds.values(), axis=1).join(
        ds[["coname", "ticker", "announce_date", "reason", "car_m1_p1"]]).to_csv(ROOT / "reports" / "oos_predictions.csv")

    # ---------- interpretable logit on full sample ----------
    txt = []
    for target in TARGETS:
        logit = interpretable_logit(ds, target)
        tab = logit.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]].round(3)
        txt.append(f"=== {target}  (baseline reason = retired, market vars standardised, HC1 SEs) ===\n"
                   f"{tab.to_string()}\npseudo R2 = {logit.prsquared:.4f}, n = {int(logit.nobs)}\n")
    (ROOT / "reports" / "logit_summary.txt").write_text("\n".join(txt))
    print("\n" + txt[0])

    # ---------- final models on all data (for live predictions) ----------
    bundle = {"targets": list(TARGETS), "primary": PRIMARY, "heads": {}}
    for target in TARGETS:
        feat, models, cols, base = fit_fold(events, ds, target)
        bundle["heads"][target] = dict(models=models, cols=cols, base=base)
        if target == PRIMARY:
            bundle["feat"] = feat
            coefs = pd.Series(models["lr_full"][-1].coef_[0], index=cols).sort_values()
            imp = pd.Series(models["xgb_full"].feature_importances_, index=cols).sort_values(ascending=False)
            coefs.to_csv(ROOT / "reports" / "lr_coefficients.csv")
            imp.to_csv(ROOT / "reports" / "xgb_importance.csv")
            print("LR standardised coefficients (+ = more likely to drop >2%):")
            print(pd.concat([coefs.head(8), coefs.tail(8)]).round(3).to_string())
    bundle["reason_cv_acc"] = bundle["feat"].reason_cv_accuracy(events.text, events.reason)
    bundle["oos_metrics"] = {t: r.to_dict() for t, r in reports.items()}
    joblib.dump(bundle, ROOT / "models" / "ceo_model.joblib")
    print("saved models/ceo_model.joblib")


if __name__ == "__main__":
    main()

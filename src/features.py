"""Shared feature engineering for training and live prediction.

Text pipeline (works on any news article / press release about a CEO departure):
  1. Keyword flags: hand-built regexes for things traders read for (interim CEO? effective
     immediately? investigation? poached by another firm? planned retirement? stays as chair?)
  2. Sentiment counts: small Loughran-McDonald-style negative / uncertainty / positive lists.
  3. Reason classifier: TF-IDF + logistic regression trained on all ~3.3k labelled departures to
     output P(death, health, fired_performance, fired_legal, retired, new_opportunity, other_mna).
     The news never says "code 3 dismissal" -- the model has to infer it from wording.
  4. Latent topics: TF-IDF -> 15-dim truncated SVD.
  5. FinBERT (ProsusAI/finbert): sentiment probabilities (fb_positive/negative/neutral) plus the
     768-dim mean-pooled embedding compressed to 16 PCs (PCA re-fit inside every training fold).
Market features come from event_study.py (pre_car_60, pre_vol_60, beta, log_dollar_vol, mkt_ret_20).
"""
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import finbert
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

REASONS = ["death", "health", "fired_performance", "fired_legal", "retired", "new_opportunity", "other_mna"]
MARKET_COLS = ["pre_car_60", "pre_vol_60", "beta", "log_dollar_vol", "mkt_ret_20"]

FLAGS = {
    "kw_interim": r"\binterim\b|\bacting (?:ceo|chief)",
    "kw_immediate": r"effective immediately|with immediate effect|immediately (?:step|resign)",
    "kw_search": r"(?:search|seek)\w* for a (?:new|permanent|successor)|executive search|search (?:firm|committee)",
    "kw_successor_named": r"(?:named|appointed|elect\w*|promot\w*) (?:as )?(?:the )?(?:new |next )?(?:president and )?(?:ceo|chief executive)|will succeed|successor",
    "kw_planned": r"succession plan|planned (?:transition|retirement)|long[- ]planned|orderly transition|after \d+ years",
    "kw_retire": r"\bretir",
    "kw_stays_chair": r"(?:remain|continue|serve) (?:as|on)\b[^.]{0,30}(?:chair|board)|executive chairman|non-executive chair",
    "kw_resign": r"\bresign",
    "kw_fired": r"\b(?:fired|ousted|terminated|dismissed|removed|forced out|pushed out|step(?:ped)? down at the request)",
    "kw_legal": r"investigat|misconduct|violat|\bSEC\b|subpoena|fraud|lawsuit|allegation|inappropriate|code of conduct|relationship with an employee",
    "kw_health": r"health (?:reasons|issues?|concerns?|problems?)|his health|her health|medical (?:leave|condition|reasons|treatment)|illness|cancer|diagnos|surgery",
    "kw_death": r"passed away|\bdied\b|death|untimely",
    "kw_poached": r"(?:join|accept\w*|take) (?:a |the )?(?:position|role|job|post) (?:as|at|with)|pursue other (?:opportunities|interests)|another company|new opportunity",
    "kw_activist": r"activist|proxy (?:fight|contest|battle)|elliott (?:management|investment)|icahn|starboard value|trian (?:fund|partners)|third point|shareholder pressure",
    "kw_earnings": r"preliminary|guidance|(?:first|second|third|fourth)[- ]quarter|earnings|results|revenue|outlook",
    "kw_cut_guidance": r"(?:lower|cut|reduc|withdr|suspend)\w* (?:its |the )?(?:\d{4} )?(?:financial |full[- ]year |annual )?(?:guidance|outlook|forecast)|below expectations|\bmiss(?:ed|es)?\b",
    "kw_mna": r"merger|acqui|definitive agreement|takeover|spin[- ]?off|transaction (?:closed|close)",
    "kw_mutual": r"mutual(?:ly)? agree|by mutual",
    "kw_personal": r"personal reasons|family reasons|spend more time",
}
FLAG_RE = {k: re.compile(v, re.I) for k, v in FLAGS.items()}

NEG = set("""abrupt abruptly sudden suddenly loss losses decline declined declining weak weakness poor
disappointing disappointed concern concerns problem problems failure failed fail adverse difficult
challenging challenges turmoil crisis downturn missed shortfall impairment restructuring layoffs
investigation misconduct scandal lawsuit violation terminated ousted forced pressure underperform
underperformance struggling struggled troubled resignation unexpected unexpectedly""".split())
UNC = set("""uncertain uncertainty may might possible possibly unclear pending search interim
temporary undetermined unknown evaluate evaluating review reviewing""".split())
POS = set("""strong growth record successful success achieve achieved achievements improve improved
improvement confident confidence excellent outstanding leadership legacy proud grateful seamless
orderly planned thoughtful accomplished momentum transformation""".split())


def keyword_features(texts) -> pd.DataFrame:
    rows = []
    for t in texts:
        t = t or ""
        words = re.findall(r"[a-z]+", t.lower())
        n = max(len(words), 1)
        r = {k: int(bool(rx.search(t))) for k, rx in FLAG_RE.items()}
        r["sent_neg"] = sum(w in NEG for w in words) / n * 100
        r["sent_unc"] = sum(w in UNC for w in words) / n * 100
        r["sent_pos"] = sum(w in POS for w in words) / n * 100
        r["log_len"] = np.log1p(n)
        rows.append(r)
    return pd.DataFrame(rows)


class TextFeaturizer:
    """Fit on a corpus of departure texts; transform any new article into numeric features."""

    def __init__(self, n_topics=15, seed=0, use_finbert=True, n_fb_pcs=16):
        self.n_topics, self.seed = n_topics, seed
        self.use_finbert, self.n_fb_pcs = use_finbert, n_fb_pcs

    def _reason_model(self):
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True, stop_words="english"),
            LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"),
        )

    def fit(self, texts, reasons):
        texts = pd.Series(texts).fillna("")
        self.reason_clf = self._reason_model().fit(texts, reasons)
        self.tfidf = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english").fit(texts)
        self.svd = TruncatedSVD(self.n_topics, random_state=self.seed).fit(self.tfidf.transform(texts))
        if self.use_finbert:
            E, _ = finbert.encode(texts.tolist())
            self.fb_pca = make_pipeline(StandardScaler(), PCA(self.n_fb_pcs, random_state=self.seed)).fit(E)
        return self

    def finbert_features(self, texts):
        E, sent = finbert.encode(list(texts))
        pcs = pd.DataFrame(self.fb_pca.transform(E), columns=[f"fb_pc_{i}" for i in range(self.n_fb_pcs)])
        return pd.concat([sent, pcs], axis=1)

    def reason_cv_accuracy(self, texts, reasons):
        pred = cross_val_predict(self._reason_model(), pd.Series(texts).fillna(""), reasons, cv=5)
        return (pred == np.asarray(reasons)).mean()

    def oof_reason_probs(self, texts, reasons, cv=5):
        """Out-of-fold reason probabilities so the downstream model never sees in-sample fits."""
        p = cross_val_predict(self._reason_model(), pd.Series(texts).fillna(""), reasons, cv=cv,
                              method="predict_proba")
        classes = sorted(set(reasons))
        return pd.DataFrame(p, columns=[f"p_{c}" for c in classes])[[f"p_{c}" for c in REASONS]]

    def reason_probs(self, texts):
        p = self.reason_clf.predict_proba(pd.Series(texts).fillna(""))
        return pd.DataFrame(p, columns=[f"p_{c}" for c in self.reason_clf.classes_])[[f"p_{c}" for c in REASONS]]

    def topics(self, texts):
        z = self.svd.transform(self.tfidf.transform(pd.Series(texts).fillna("")))
        return pd.DataFrame(z, columns=[f"topic_{i}" for i in range(self.n_topics)])

    def transform(self, texts, reason_probs=None):
        rp = self.reason_probs(texts) if reason_probs is None else reason_probs.reset_index(drop=True)
        parts = [keyword_features(texts), rp, self.topics(texts)]
        if self.use_finbert:
            parts.append(self.finbert_features(texts))
        return pd.concat(parts, axis=1)

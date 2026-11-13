# CEO Departures → Stock Reaction

**Question:** when a CEO resigns or steps down, does the stock go down, and can we predict which
announcements will hurt, by reading the news?

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `build_events.py` | 3,332 dated CEO departures (1993–2025) from the open **CEO Dismissal Database** (Gentry et al., Zenodo 23328926). Each departure has a hand-coded reason and a `notes` field that quotes or summarises the press release and news coverage. Sentences describing the share-price reaction are **removed** to prevent label leakage. |
| 2 | `map_tickers.py` | Company → ticker, using tickers quoted in the press release first and otherwise a Yahoo search accepted only on a ≥0.85 name match. 1,595 events matched. |
| 3 | `event_study.py` | Market-model abnormal returns (β estimated on days −250…−21 vs SPY). Target window CAR[−1,+1]. Also builds pre-event features: 60-day abnormal return, idiosyncratic vol, β, liquidity, market momentum. **1,353 events** have usable prices. |
| 4 | `placebo.py` | Logistic regression of the CEO-event days against 3,918 random ordinary days for the **same stocks**, with SEs clustered by ticker. |
| 5 | `features.py` / `train.py` | NLP features (below), then logistic regression and XGBoost, evaluated **walk-forward by year** (2012–2024, 1,016 out-of-sample events). |
| 6 | `predict.py` | Feed it new news plus a ticker and it outputs probabilities, the inferred departure type and the main drivers. |

**NLP features** (`features.py`):
1. **Departure-type classifier:** TF-IDF + logistic regression trained on all 3,332 texts to infer *why* the CEO left (fired for performance or legal issues, retired, poached, health, death, M&A). 5-fold accuracy is 76%, against a 48% majority-class baseline. Out-of-fold probabilities are passed to stage 2, so there is no in-sample leakage.
2. **19 keyword flags** a trader would look for: effective immediately, interim CEO, search firm, successor named, stays as chair, investigation, poached, activist, guidance cut, "personal reasons", and so on.
3. **Sentiment rates** from Loughran-McDonald-style negative, uncertainty and positive word lists.
4. **15 latent topics** (TF-IDF → SVD).
5. **FinBERT** (`finbert.py`, ProsusAI/finbert): sentiment probabilities plus a 768-dim mean-pooled embedding, compressed to 16 PCs. The PCA is re-fit inside each fold. Texts longer than 512 tokens are chunked and averaged, and results are cached in `data/finbert_cache.parquet`.

## Results

### 1. Yes: CEO departures push stocks down on average
- Mean CAR[−1,+1] = **−1.15%** (t = −5.9, n = 1,353).
- Placebo logit, event vs ordinary day for the same stock:

| Outcome | Event-day odds ratio | p |
|---|---|---|
| Any abnormal decline | 1.31 | 1e−5 |
| Abnormal drop > 2% | **1.65** | 3e−14 |
| Drop > 1σ (vol-scaled) | **2.38** | 5e−29 |

- The effect depends heavily on **why** the CEO left (odds of a >2% drop vs an ordinary day): poached by another firm **5.9×**, fired for legal reasons **3.5×**, fired for performance **2.6×**, retired 1.3×, M&A-related 1.1× (not significant).

### 2. Predicting individual events (walk-forward out-of-sample)

| Target | Base rate | LR market-only | LR text+market | XGBoost | Ensemble (all features) |
|---|---|---|---|---|---|
| Any decline (sign) | 59% | 0.47 | 0.51 | 0.49 | 0.50 |
| Drop > 2% | 35% | 0.60 | 0.60 | 0.60 | **0.61** |
| Drop > 5% | 19% | 0.62 | 0.62 | 0.63 | **0.63** |
| Drop > 1σ (vol-scaled) | 28% | 0.51 | **0.56** | 0.56 | 0.56 |

*(AUC)*

What this means:
- **The sign is not predictable.** A 3-day idiosyncratic σ of about 4% swamps a mean effect of about 1%. Even a logit given the *true* hand-coded reason only reaches AUC 0.53.
- **Large drops are predictable.** But for fixed-% thresholds, much of that comes from volatility (volatile stocks cross 2% more often in *either* direction).
- **The vol-scaled target isolates the news signal:** market features alone score 0.51, text features 0.56. The ceiling with the true reason is about 0.57–0.61, and the text model gets most of the way there.
- Ranked by model score, the riskiest third of announcements averaged **−2.4%** CAR, against **−0.9%** for the safest third.

### 3. FinBERT ablation (AUC, walk-forward OOS)

| Model | Drop > 2% | Drop > 5% | Drop > 1σ (vol-scaled) |
|---|---|---|---|
| LR, market only | 0.596 | 0.621 | 0.513 |
| LR, FinBERT + market | 0.595 | 0.609 | 0.532 |
| LR, hand-built text + market | 0.596 | 0.621 | **0.563** |
| LR, everything | 0.595 | 0.610 | 0.560 |
| XGBoost, everything | 0.601 | 0.633 | 0.555 |
| Ensemble (served) | 0.607 | 0.631 | 0.560 |

**FinBERT adds no out-of-sample lift.** On its own it carries signal: its tone tracks departure type (mean P(negative) is 0.72 for legal firings vs 0.14 for retirements), and it beats market-only features on the vol-scaled target. But that information is already captured by the departure-type classifier and keyword flags. Even so, its `fb_negative` and `fb_positive` scores end up among the larger LR coefficients. FinBERT stays in the served model as a stronger tone reader for messy live news, and `predict.py` prints its tone.

## Usage

```bash
cd src
python predict.py --ticker XYZ --text "XYZ said CEO Jane Doe resigned effective immediately..."
python predict.py --ticker XYZ --file article.txt
python predict.py --ticker XYZ --fetch-news "XYZ CEO steps down"     # pulls Google News RSS, last 7 days
python predict.py --ticker UNH --date 2025-05-13 --text "..."        # score a past announcement
```
To rebuild everything from scratch, run `./run.sh` (about 15 minutes).

## Caveats
- **Survivorship bias:** only firms Yahoo still has prices for (mostly survivors). Delisted or acquired firms are dropped, so 1,353 of 3,332 events are used.
- **Training text ≠ live text:** the notes are researcher-written summaries plus press-release excerpts, while live input is raw news. Expect some degradation.
- **Announcement-date precision:** day 0 is the first trading day on or after the reported date, and the [−1,+1] window absorbs most timing error. Same-day earnings or guidance news is mixed in; the `kw_earnings` and `kw_cut_guidance` flags are there to capture it.
- **Small, weak signal:** an AUC of 0.56–0.64 is a real edge for ranking risk, but not a trading signal on its own. There is no transaction-cost or tradability analysis.

## Next ideas
- Pull 8-K Item 5.02 filing timestamps from EDGAR for exact announcement times (needs an SEC User-Agent with contact info).
- Fine-tune FinBERT end-to-end on the reason labels (the frozen embedding + PCA is lossy), or use an LLM extracting structured fields (forced? successor ready? same-day guidance?).
- Recover delisted firms through CRSP/WRDS to remove survivorship bias.
- Regress CAR directly (quantile regression on the left tail) instead of thresholded labels.

#!/usr/bin/env bash
# Rebuild the full pipeline from the raw CEO Dismissal Database.
set -euo pipefail
cd "$(dirname "$0")/src"
pip install -q pandas numpy scikit-learn xgboost statsmodels yfinance feedparser pyarrow joblib torch "transformers>=4.46"
mkdir -p ../data/raw ../models ../reports
[ -f ../data/raw/ceo_dismissal.dta ] || curl -sL -o ../data/raw/ceo_dismissal.dta \
  "https://zenodo.org/records/23328926/files/CEO%20Dismissal%20Dataset%20Stata%2026Nov25.dta?download=1"
python3 build_events.py     # events + cleaned news text
python3 map_tickers.py      # company -> ticker
python3 event_study.py      # prices, market-model CARs, pre-event features
python3 finbert.py          # FinBERT sentiment + embeddings for every text (cached, ~3 min)
python3 placebo.py          # logit: CEO-event days vs ordinary days
python3 train.py | tee ../reports/train_log.txt   # walk-forward LR / XGBoost + final models

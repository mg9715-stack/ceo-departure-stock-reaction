"""FinBERT encoder (ProsusAI/finbert) with an on-disk cache.

For each text returns:
  - 768-dim mean-pooled last-hidden-state embedding (reduced later by PCA inside each CV fold)
  - FinBERT sentiment probabilities: fb_positive, fb_negative, fb_neutral

FinBERT is pre-trained (not fit on our labels), so caching across folds introduces no leakage.
Long texts are split into 512-token chunks; embeddings / probabilities are averaged over chunks.
"""
from pathlib import Path
import hashlib
import os
os.environ.setdefault("USE_TF", "0")  # avoid importing the TensorFlow in the base env
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "finbert_cache.parquet"
MODEL = "ProsusAI/finbert"
EMB_DIM = 768
_model = _tok = _device = None


def _key(t: str) -> str:
    return hashlib.md5((t or "").encode()).hexdigest()


def _load():
    global _model, _tok, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_hidden_states=True).to(_device).eval()
    return _model, _tok, _device


def _encode_raw(texts, batch_size=16):
    import torch
    model, tok, dev = _load()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    # chunk every text into <=510-token windows
    chunks, owner = [], []
    for i, t in enumerate(texts):
        ids = tok((t or " ").strip() or " ", add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
        for s in range(0, len(ids), 510):
            chunks.append(tok.decode(ids[s:s + 510]))
            owner.append(i)
    embs, probs = [], []
    with torch.no_grad():
        for s in range(0, len(chunks), batch_size):
            enc = tok(chunks[s:s + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            out = model(**enc)
            h = out.hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            probs.append(torch.softmax(out.logits, -1).cpu().numpy())
    embs, probs, owner = np.vstack(embs), np.vstack(probs), np.array(owner)
    E = np.vstack([embs[owner == i].mean(0) for i in range(len(texts))])
    P = np.vstack([probs[owner == i].mean(0) for i in range(len(texts))])
    return E, pd.DataFrame(P, columns=[f"fb_{l}" for l in labels])


def encode(texts):
    """Returns (embeddings ndarray [n,768], sentiment DataFrame [n,3]); uses/extends the disk cache."""
    texts = [t or "" for t in texts]
    keys = [_key(t) for t in texts]
    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    have = set(cache.index) if len(cache) else set()
    todo = list(dict.fromkeys(k for k in keys if k not in have))
    if todo:
        tmap = dict(zip(keys, texts))
        E, P = _encode_raw([tmap[k] for k in todo])
        new = pd.concat([pd.DataFrame(E, index=todo, columns=[f"e{i}" for i in range(EMB_DIM)]),
                         P.set_axis(todo)], axis=1)
        cache = pd.concat([cache, new]) if len(cache) else new
        if len(todo) > 5:  # don't churn the cache file for single live predictions
            cache.to_parquet(CACHE)
    rows = cache.loc[keys]
    return rows[[f"e{i}" for i in range(EMB_DIM)]].values, rows[["fb_positive", "fb_negative", "fb_neutral"]].reset_index(drop=True)


if __name__ == "__main__":
    ev = pd.read_parquet(ROOT / "data" / "events.parquet")
    E, P = encode(ev.text.tolist())
    print(E.shape, "\n", P.describe().round(3))
    ev = ev.reset_index(drop=True).join(P)
    print(ev.groupby("reason")[["fb_positive", "fb_negative", "fb_neutral"]].mean().round(3))

# CSINTSY-MCO2-PinoyBot

**PinoyBot** is a word-level language identifier for Filipino code-switched text. Given a tokenized Filipino-English passage, it tags each word as:

| Tag | Meaning |
|-----|---------|
| `FIL` | Filipino / Tagalog word |
| `ENG` | English word |
| `CS`  | Intra-word code-switching (e.g. `nag-march`, `pina-explain`) |
| `OTH` | Names, abbreviations, numbers, punctuation, emojis, onomatopoeia, etc. |

Built for **MCO2** (Machine Learning course, DLSU), using data derived from the Corpus of Historical Filipino English (CoHFiE), courtesy of the Center for Language Technologies (CeLT).

## How it works

Each word is converted into a 30-feature numeric vector (character composition, capitalization, Filipino/English morphological markers, consonant clustering, etc. — see `features.py`) and classified with a `HistGradientBoostingClassifier` from scikit-learn.

**Test accuracy: 91.64%** on 6,126 held-out words (330 sentences), with a 70/15/15 sentence-level train/val/test split to avoid data leakage.

## Repo structure

```
mco2/
├── pinoybot.py          # Main deliverable — tag_language(tokens) function
├── features.py           # Feature extraction (30 features per word)
├── trained_model.pkl      # Pre-trained model + feature column order
├── demo.py                # Interactive CLI to try the tagger
├── dataset.csv            # Annotated training data (not always included — see below)
├── training/
│   └── train_model.py     # Full training pipeline (load → split → train → evaluate → save)
```

## Setup

```bash
pip install scikit-learn joblib pandas numpy
```

## Quick start

```bash
python pinoybot.py
```
```
Tokens: ['Love', 'kita', '.']
Tags: ['ENG', 'FIL', 'OTH']
```

Or use it in your own code:

```python
from pinoybot import tag_language

tokens = ["Gusto", "ko", "ng", "ice", "cream", "."]
tags = tag_language(tokens)
# ['FIL', 'FIL', 'FIL', 'FIL', 'ENG', 'OTH']
```

Or try the interactive demo:

```bash
python demo.py
```

## Re-training the model

If you change `dataset.csv` or `features.py`, re-run training so `trained_model.pkl` stays in sync:

```bash
python training/train_model.py
```

## Restrictions honored

Per the assignment spec: no dictionary lookups to directly infer word language, and no use of pre-existing language identification models — all predictions come from features engineered from the word itself.

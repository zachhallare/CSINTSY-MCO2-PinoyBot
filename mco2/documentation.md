# PinoyBot Documentation

Filipino Code-Switched Language Identifier — tags each word in a passage as **FIL** (Filipino), **ENG** (English), **CS** (Code-Switched), or **OTH** (Other).

---

## Project Files

```
mco2/
├── pinoybot.py          # Main bot — contains the tag_language() function
├── features.py          # Shared feature extraction (30 features)
├── trained_model.pkl    # Saved HistGradientBoosting model (auto-loaded by pinoybot)
├── demo.py              # Interactive CLI for testing
├── dataset.csv          # Annotated training data (41,903 words)
└── training/
    └── train_model.py   # Model training pipeline
```

---

## What Each File Does

### `pinoybot.py`
The core deliverable. Contains `tag_language(tokens)` which:
1. Takes a list of word strings (e.g., `["Love", "kita", "."]`)
2. Extracts 30 numeric features from each word using `features.py`, including whether the word is the first word of the passage
3. Feeds the features into the pre-trained classifier
4. Returns a list of tag strings (e.g., `["ENG", "FIL", "OTH"]`)

The model is loaded once when the module is imported, so repeated calls are fast.

### `features.py`
Extracts 30 linguistic features from a single word token, given its position in the sentence. These features fall into five categories:

| Category | Features |
|----------|----------|
| **Character composition** | `length`, `vowel_ratio`, `conso_ratio`, `upper_ratio`, `special_char_ratio` |
| **Character-type flags** | `has_foreign`, `has_hyphen`, `is_punctuation`, `is_numeric`, `is_all_digit`, `first_is_cap`, `is_all_caps`, `is_all_lower` |
| **Morphological markers** | `has_fil_prefix`, `has_fil_suffix`, `contains_ng`, `has_eng_suffix`, `double_conso`, `has_eng_digraph`, `is_reduplicated`, `ends_with_ng`, `ends_with_n`, `has_cs_pattern` |
| **Pattern features** | `max_conso_cluster`, `max_conso_cluster_ratio`, `starts_with_vowel`, `ends_with_vowel`, `has_repeated_pattern`, `hyphen_with_prefix` |
| **Positional** | `is_first_word` — whether the token is the first word of its sentence |

**Why `is_first_word` matters:** capitalization (`first_is_cap`) is a strong signal for names/proper nouns (tagged OTH) in the training data, since names are always capitalized. But the first word of any sentence is also capitalized regardless of language, which confused the model on ordinary words like "Love" at a sentence start. Splitting out sentence-initial position from mid-sentence capitalization fixed this — see [Model Performance](#model-performance) below.

**Why `_ENG_SUFFIXES` no longer includes `al`, `er`, `ed`, `ly`:** these 2-letter endings turned out to be nearly as common in Filipino words (`bawal`, `tagal`, `opisyal`) as in English ones, so they were adding noise rather than a useful signal. Only longer, more distinctive English suffixes (`tion`, `ment`, `able`, etc.) were kept.

### `training/train_model.py`
Handles the full training pipeline:
1. Loads `dataset.csv` and cleans annotation typos (e.g., `"ENF"` → `"ENG"`, `"OHT"` → `"OTH"`)
2. Splits data **by sentence** (70% train / 15% validation / 15% test) to prevent data leakage
3. Extracts features for all words, including the `is_first_word` flag derived from each word's position within its sentence
4. Trains a `HistGradientBoostingClassifier`
5. Prints classification reports for validation and test sets
6. Saves the model to `trained_model.pkl`

### `demo.py`
Interactive CLI for manually testing the bot. Handles tokenization (splits punctuation from words) and displays results with color-coded tags.

### `trained_model.pkl`
Serialized model package containing:
- The trained `HistGradientBoostingClassifier`
- The ordered list of feature column names (used to keep training and inference feature vectors aligned)

---

## How to Run

### Prerequisites
```bash
pip install scikit-learn joblib pandas numpy
```

### Quick Test
```bash
cd mco2
python pinoybot.py
```
Output:
```
Tokens: ['Love', 'kita', '.']
Tags: ['ENG', 'FIL', 'OTH']
```

### Interactive Demo
```bash
python demo.py
```
Then type any Filipino-English sentence:
```
> Madami ang nag-march sa 13 trillion peso march sa EDSA monument at Luneta Park .

  Word                      Tag
  ───────────────────────────────────
  Madami                    FIL
  ang                       FIL
  nag-march                 CS
  sa                        FIL
  13                        OTH
  trillion                  ENG
  peso                      FIL
  march                     ENG
  sa                        FIL
  EDSA                      OTH
  monument                  ENG
  at                        FIL
  Luneta                    OTH
  Park                      OTH
  .                         OTH
```

> **Note:** this is real output from the current model, not a hand-annotated ground truth. Per the spec's own worked example, `peso` and `Park` should be ENG — the model gets both wrong here. This is expected: overall test accuracy is ~91.6% (see [Model Performance](#model-performance)), and ENG is the weakest-performing class (see table below), so mistakes on individual English loanwords like these are consistent with its measured error rate, not a sign something is broken.

Commands inside the demo:
- Type any sentence → see per-word tags
- `clear` → clear the screen
- `quit` → exit

### Re-Train the Model
Needed if you modify the dataset or `features.py` (e.g., add/remove a feature) — `pinoybot.py` will error or silently mispredict if `trained_model.pkl` was trained on a different feature set than the current `features.py` produces:
```bash
python training/train_model.py
```
This regenerates `trained_model.pkl` and prints accuracy metrics.

### Use in Your Own Code
```python
from pinoybot import tag_language

tokens = ["Gusto", "ko", "ng", "ice", "cream", "."]
tags = tag_language(tokens)
print(tags)
# ['FIL', 'FIL', 'FIL', 'FIL', 'ENG', 'OTH']
```

---

## Model Performance

Model: `HistGradientBoostingClassifier` (scikit-learn), `max_iter=500`, `learning_rate=0.05`, `max_depth=8`, `min_samples_leaf=20`, `l2_regularization=0.1`, `early_stopping=True`, `random_state=42`.

Data split 70/15/15 by sentence (not by word, to avoid leakage): 1,540 train / 330 validation / 330 test sentences, giving 29,228 / 6,549 / 6,126 words respectively.

**Test accuracy: 91.64%** on 6,126 held-out words (330 sentences).

| Tag | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| CS  | 0.59      | 0.32   | 0.42     | 31      |
| ENG | 0.71      | 0.58   | 0.64     | 493     |
| FIL | 0.94      | 0.97   | 0.96     | 4,648   |
| OTH | 0.89      | 0.83   | 0.86     | 954     |

> CS detection is limited due to extreme class imbalance (~198 CS examples out of 41,903 words). ENG is the weakest well-populated class — many English loanwords used in Filipino contexts (`peso`, `Park`) share surface features with Filipino words, so precision/recall lag behind FIL and OTH.
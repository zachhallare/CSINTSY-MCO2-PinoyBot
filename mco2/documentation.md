# PinoyBot Documentation

Filipino Code-Switched Language Identifier — tags each word in a passage as **FIL** (Filipino), **ENG** (English), **CS** (Code-Switched), or **OTH** (Other).

---

## Project Files

```
mco2/
├── pinoybot.py          # Main bot — contains the tag_language() function
├── features.py          # Shared feature extraction (24 features)
├── trained_model.pkl    # Saved Random Forest model (auto-loaded by pinoybot)
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
2. Extracts 24 numeric features from each word using `features.py`
3. Feeds the features into the pre-trained Random Forest model
4. Returns a list of tag strings (e.g., `["ENG", "FIL", "OTH"]`)

The model is loaded once when the module is imported, so repeated calls are fast.

### `features.py`
Extracts 24 linguistic features from a single word token. These features fall into four categories:

| Category | Features |
|----------|----------|
| **Character composition** | `length`, `vowel_ratio`, `conso_ratio`, `upper_ratio`, `special_char_ratio` |
| **Character-type flags** | `has_foreign`, `has_hyphen`, `is_punctuation`, `is_numeric`, `is_all_digit`, `first_is_cap`, `is_all_caps`, `is_all_lower` |
| **Morphological markers** | `has_fil_prefix`, `has_fil_suffix`, `contains_ng`, `has_eng_suffix`, `double_conso` |
| **Pattern features** | `max_conso_cluster`, `max_conso_cluster_ratio`, `starts_with_vowel`, `ends_with_vowel`, `has_repeated_pattern`, `hyphen_with_prefix` |

### `training/train_model.py`
Handles the full training pipeline:
1. Loads `dataset.csv` and cleans annotation typos (e.g., `"FIL "` → `"FIL"`)
2. Splits data **by sentence** (70% train / 15% validation / 15% test) to prevent data leakage
3. Extracts features for all words
4. Trains a Random Forest classifier with balanced class weights
5. Prints classification reports for validation and test sets
6. Saves the model to `trained_model.pkl`

### `demo.py`
Interactive CLI for manually testing the bot. Handles tokenization (splits punctuation from words) and displays results with color-coded tags.

### `trained_model.pkl`
Serialized model package containing:
- The trained Random Forest model (300 trees)
- The ordered list of feature column names

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
  peso                      ENG
  march                     ENG
  sa                        FIL
  EDSA                      OTH
  monument                  ENG
  at                        FIL
  Luneta                    OTH
  Park                      ENG
  .                         OTH
```

Commands inside the demo:
- Type any sentence → see per-word tags
- `clear` → clear the screen
- `quit` → exit

### Re-Train the Model
Only needed if you modify the dataset or features:
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
# ['FIL', 'FIL', 'FIL', 'ENG', 'ENG', 'OTH']
```

---

## Model Performance

**Test accuracy: 92.87%** on 6,287 held-out words (330 sentences).

| Tag | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| CS  | 0.17      | 0.14   | 0.15     | 29      |
| ENG | 0.81      | 0.73   | 0.77     | 560     |
| FIL | 0.96      | 0.96   | 0.96     | 4,843   |
| OTH | 0.87      | 0.87   | 0.87     | 855     |

> CS detection is limited due to extreme class imbalance (~200 CS examples out of 42,000 words). The model still identifies clear CS patterns like "nag-march", "nagreact", and "na-award".

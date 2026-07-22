"""
train_model.py

Training pipeline for the PinoyBot language identification model.
Loads the annotated dataset, cleans labels, extracts features, trains
a Random Forest classifier, evaluates on validation and test splits,
and saves the trained model to disk.

Usage:
    python train_model.py
"""

import os
import sys
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Allow importing the shared features module from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from features import extract_features, get_feature_columns

# Paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_SCRIPT_DIR, '..', 'dataset.csv')
_MODEL_OUTPUT_PATH = os.path.join(_SCRIPT_DIR, '..', 'trained_model.pkl')

# Corrections for annotation typos discovered in the dataset
_TAG_CORRECTIONS = {
    'FIL ': 'FIL',
    'ENF': 'ENG',
    'OHT': 'OTH',
    'EGNG': 'ENG',
    'OT': 'OTH',
    'e': 'ENG',
}

_VALID_TAGS = {'ENG', 'FIL', 'CS', 'OTH'}


def load_and_clean_data(path):
    """
    Load the CSV dataset and normalize annotation labels.

    Fixes known typos, strips whitespace, and drops rows with
    missing or unrecoverable tags.

    Args:
        path: Path to the dataset CSV file.

    Returns:
        Cleaned pandas DataFrame with valid tags only.
    """
    df = pd.read_csv(path)

    if 'answer' in df.columns and 'tag' not in df.columns:
        df = df.rename(columns={'answer': 'tag'})

    df['tag'] = df['tag'].astype(str).str.strip()
    df['tag'] = df['tag'].replace(_TAG_CORRECTIONS)

    # Drop rows where tags could not be corrected
    initial_count = len(df)
    df = df[df['tag'].isin(_VALID_TAGS)].copy()
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with invalid/missing tags")

    print(f"Dataset: {len(df)} words across {df['sentence_id'].nunique()} sentences")
    print(f"Tag distribution:\n{df['tag'].value_counts()}\n")

    return df


def sentence_level_split(df, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split the dataset by sentence_id to prevent data leakage.

    Words from the same sentence always stay in the same split so the
    model cannot exploit partial sentence context during evaluation.

    Args:
        df: DataFrame with a 'sentence_id' column.
        train_ratio: Fraction of sentences allocated to training (0.70).
        val_ratio: Fraction of sentences allocated to validation (0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    sentence_ids = df['sentence_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(sentence_ids)

    n = len(sentence_ids)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_sids = set(sentence_ids[:train_end])
    val_sids = set(sentence_ids[train_end:val_end])
    test_sids = set(sentence_ids[val_end:])

    train_df = df[df['sentence_id'].isin(train_sids)]
    val_df = df[df['sentence_id'].isin(val_sids)]
    test_df = df[df['sentence_id'].isin(test_sids)]

    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(f"Sentences: {len(train_sids)} train / {len(val_sids)} val / {len(test_sids)} test\n")

    return train_df, val_df, test_df


def build_feature_matrix(df, feature_columns):
    """
    Extract features from every word in the DataFrame.

    Args:
        df: DataFrame with a 'word' column.
        feature_columns: Ordered list of feature names.

    Returns:
        Tuple of (X as numpy array, y as numpy array of tag strings).
    """

    is_first = (
        df.groupby('sentence_id')['word_id'].transform('min') == df['word_id']
    )

    feature_rows = []
    for word, first_flag in zip(df['word'], is_first):
        feat_dict = extract_features(word, is_first_word=bool(first_flag))
        feature_rows.append([feat_dict[col] for col in feature_columns])

    X = np.array(feature_rows, dtype=float)
    y = df['tag'].values

    return X, y


def train_and_evaluate():
    """Run the full training pipeline: load, split, train, evaluate, and save."""

    df = load_and_clean_data(_DATASET_PATH)
    feature_columns = get_feature_columns()

    train_df, val_df, test_df = sentence_level_split(df)

    print("Extracting features...")
    X_train, y_train = build_feature_matrix(train_df, feature_columns)
    X_val, y_val = build_feature_matrix(val_df, feature_columns)
    X_test, y_test = build_feature_matrix(test_df, feature_columns)
    print(f"Feature matrix shape: {X_train.shape}\n")

    # Use HistGradientBoostingClassifier to improve accuracy
    # We do not use extreme sample weights here so we can maximize overall accuracy
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42,
    )

    print("Training HistGradientBoosting classifier...")
    model.fit(X_train, y_train)

    # Validation results
    val_pred = model.predict(X_val)
    print("=" * 60)
    print("VALIDATION SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(classification_report(y_val, val_pred, zero_division=0))

    # Test results
    test_pred = model.predict(X_test)
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))

    # Bundle model and feature column ordering for consistent inference
    model_package = {
        'model': model,
        'feature_columns': feature_columns,
    }
    joblib.dump(model_package, _MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {os.path.abspath(_MODEL_OUTPUT_PATH)}")

    # HistGBT doesn't expose feature_importances_ directly so we skip printing them


if __name__ == '__main__':
    train_and_evaluate()
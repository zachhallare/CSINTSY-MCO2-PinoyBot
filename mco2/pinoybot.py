"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

Provides the main tagging function for the PinoyBot project, which identifies
the language of each word in code-switched Filipino-English text.

Tags: ENG (English), FIL (Filipino), CS (intra-code switched), OTH (Other)
"""


import os
from typing import List

# pyrefly: ignore [missing-import]
import joblib
from features import extract_features

# Load trained model once at import time to avoid reloading on every call
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_model_package = joblib.load(os.path.join(_SCRIPT_DIR, "trained_model.pkl"))
_model = _model_package["model"]
_feature_columns = _model_package["feature_columns"]
_VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}


def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.

    Args:
        tokens: List of word tokens (strings).

    Returns:
        List of predicted tags, one per token.
        Each tag is one of: "ENG", "FIL", "CS", "OTH".
        Length of the returned list equals the length of `tokens`.
    """
    if not tokens:
        return []

    # Build feature matrix — one feature vector per token
    feature_matrix = []
    for i, token in enumerate(tokens):
        feat_dict = extract_features(token, is_first_word=(i == 0))
        feature_matrix.append([feat_dict[col] for col in _feature_columns])

    # Run classifier
    predicted = _model.predict(feature_matrix)

    # Ensure every prediction is a valid tag; fall back to OTH if not
    tags = [str(t) if str(t) in _VALID_TAGS else "OTH" for t in predicted]

    return tags


if __name__ == "__main__":
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)
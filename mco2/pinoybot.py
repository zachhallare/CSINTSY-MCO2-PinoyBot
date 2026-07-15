"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

import joblib
from features import extract_features

# Load trained model once at import time to avoid reloading on every call
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_model_package = joblib.load(os.path.join(_SCRIPT_DIR, "trained_model.pkl"))
_model = _model_package["model"]
_feature_columns = _model_package["feature_columns"]
_VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    # Model is pre-loaded at module level above for efficiency.

    if not tokens:
        return []

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    feature_matrix = []
    for token in tokens:
        feat_dict = extract_features(token)
        feature_matrix.append([feat_dict[col] for col in _feature_columns])

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    predicted = _model.predict(feature_matrix)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Ensure every prediction maps to a valid tag label
    tags = [t if t in _VALID_TAGS else "OTH" for t in tags]

    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)
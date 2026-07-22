"""
features.py

Feature extraction module for PinoyBot language identification.
Converts word tokens into numeric feature vectors for classification.
"""

import string


# Letters uncommon in native Filipino orthography (removed 'c' to allow common loanwords like Chinito)
_FOREIGN_LETTERS = set('fjqvxz')

# Common Filipino verb/noun prefixes, ordered longest-first for greedy matching
_FIL_PREFIXES = [
    'nakapag', 'nagpa', 'nakapa', 'pinaka', 'pagka',
    'ipinag', 'ipina',
    'naka', 'nag', 'pag', 'mag', 'mang', 'pang',
    'taga', 'ipa', 'isa', 'pin',
    'ma', 'pa', 'ka', 'na', 'um', 'in', 'i'
]

# Common Filipino word-ending suffixes
_FIL_SUFFIXES = ['uhan', 'ahan', 'ihan', 'hin', 'han', 'an', 'in', 'ng']

# Common English morphological suffixes, ordered longest-first
# NOTE: short 2-letter endings ('al', 'er', 'ed', 'ly') were removed —
# they're too generic and appear about as often in Filipino words
# (e.g. "bawal", "tagal", "opisyal") as in English ones, so they added
# noise rather than signal. Kept only endings that are rare in Filipino.
_ENG_SUFFIXES = [
    'tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ious', 'eous',
    'ous', 'ive', 'ing', 'ful', 'less', 'ist', 'ism',
    'ity', 'ence', 'ance', 'ers', 'est'
]

# Doubled consonants are more frequent in English than Filipino
_DOUBLE_CONSONANTS = [
    'tt', 'ss', 'll', 'dd', 'bb', 'mm', 'nn', 'rr', 'ff', 'pp', 'cc', 'gg'
]

# English consonant digraphs — extremely rare in native Filipino orthography
_ENG_DIGRAPHS = [
    'th', 'sh', 'wh', 'ph', 'gh', 'ck', 'qu'
]

_VOWELS = set('aeiou')


def extract_features(token, is_first_word=False):
    """
    Convert a word token into a dictionary of numeric features.

    Args:
        token: A string representing a single word.
        is_first_word: True if this token is the first word of its
            sentence. Lets the model separate capitalization caused by
            sentence position from capitalization that signals a proper
            noun (e.g. "Love" at a sentence start vs. "EDSA" mid-sentence).

    Returns:
        Dictionary mapping feature names to numeric values (int or float).
    """
    word = str(token)
    lower = word.lower()
    letter_count = sum(1 for c in word if c.isalpha())
    word_len = len(word)

    features = {}

    # Word length and character composition ratios
    features['length'] = word_len
    features['vowel_ratio'] = _ratio_of_set(lower, _VOWELS, letter_count)
    features['conso_ratio'] = (
        1.0 - features['vowel_ratio'] if letter_count > 0 else 0.0
    )
    features['upper_ratio'] = (
        sum(1 for c in word if c.isupper()) / letter_count
        if letter_count else 0.0
    )
    features['special_char_ratio'] = (
        sum(1 for c in word if not c.isalnum()) / word_len
        if word_len else 0.0
    )

    # Character-type indicator flags
    features['has_foreign'] = int(any(c in _FOREIGN_LETTERS for c in lower))
    features['has_hyphen'] = int('-' in word)
    features['is_punctuation'] = int(
        bool(word) and all(c in string.punctuation for c in word)
    )
    features['is_numeric'] = int(any(c.isdigit() for c in word))
    features['is_all_digit'] = int(bool(word) and word.isdigit())
    features['first_is_cap'] = int(bool(word) and word[0].isupper())
    features['is_all_caps'] = int(
        letter_count > 1
        and all(c.isupper() for c in word if c.isalpha())
    )
    features['is_all_lower'] = int(
        letter_count > 0
        and all(c.islower() for c in word if c.isalpha())
    )

    # Filipino morphological markers
    features['has_fil_prefix'] = _has_affix(lower, _FIL_PREFIXES, prefix=True)
    features['has_fil_suffix'] = _has_affix(lower, _FIL_SUFFIXES, prefix=False)
    features['contains_ng'] = int('ng' in lower)

    # English morphological markers
    features['has_eng_suffix'] = _has_affix(lower, _ENG_SUFFIXES, prefix=False)
    features['double_conso'] = int(any(d in lower for d in _DOUBLE_CONSONANTS))

    # Consonant clustering patterns
    features['max_conso_cluster'] = _max_consonant_cluster(lower)
    features['max_conso_cluster_ratio'] = (
        features['max_conso_cluster'] / max(word_len, 1)
    )

    # Vowel position features
    features['starts_with_vowel'] = int(
        bool(lower) and lower[0] in _VOWELS
    )
    features['ends_with_vowel'] = int(
        bool(lower) and lower[-1] in _VOWELS
    )

    # Repetition detection for laughter and exclamations (e.g., "hahaha")
    features['has_repeated_pattern'] = _has_repeated_pattern(lower)

    # Code-switching signal: Filipino prefix joined by hyphen to a root word
    features['hyphen_with_prefix'] = int(
        features['has_hyphen'] and features['has_fil_prefix']
    )

    # English digraph detection — th/sh/ch/wh/ph/gh/ck/qu are rare in Filipino
    features['has_eng_digraph'] = int(
        any(dg in lower for dg in _ENG_DIGRAPHS)
    )

    # Filipino reduplication: hyphenated word where both halves are identical
    # (e.g., "iba-iba", "kain-kain", "bahay-bahay")
    features['is_reduplicated'] = _is_reduplicated(lower)

    # Filipino phonological endings — -ng and -n are highly productive in Filipino
    features['ends_with_ng'] = int(lower.endswith('ng'))
    features['ends_with_n'] = int(
        bool(lower) and lower[-1] == 'n' and not lower.endswith('ng')
    )

    # Strong CS signal: Filipino prefix AND English suffix on the same word
    # (e.g., "naglunch" has nag- prefix + no eng suffix, but "nagmeeting" or
    # "nagreact" may have both signals simultaneously)
    features['has_cs_pattern'] = int(
        features['has_fil_prefix'] == 1 and features['has_eng_suffix'] == 1
    )

    # Positional signal: sentence-initial position explains capitalization
    # that has nothing to do with being a proper noun.
    features['is_first_word'] = int(is_first_word)

    return features


def get_feature_columns():
    """Return the ordered list of feature names produced by extract_features."""
    return list(extract_features("dummy").keys())


def _ratio_of_set(lower_word, char_set, letter_count):
    """Calculate ratio of characters belonging to char_set among alphabetic chars."""
    if letter_count == 0:
        return 0.0
    return sum(1 for c in lower_word if c in char_set) / letter_count


def _has_affix(lower_word, affixes, prefix=True):
    """
    Check if word starts or ends with any affix in the list.

    Only matches when the word is strictly longer than the affix itself,
    preventing false positives on short function words (e.g., "ma", "in").
    """
    for affix in affixes:
        if len(lower_word) <= len(affix):
            continue
        if prefix and lower_word.startswith(affix):
            return 1
        if not prefix and lower_word.endswith(affix):
            return 1
    return 0


def _max_consonant_cluster(lower_word):
    """Find the length of the longest consecutive consonant sequence."""
    max_len = 0
    current = 0
    for c in lower_word:
        if c.isalpha() and c not in _VOWELS:
            current += 1
            if current > max_len:
                max_len = current
        else:
            current = 0
    return max_len


def _has_repeated_pattern(lower_word):
    """
    Detect words formed by repeating a short pattern (e.g., "hahaha", "hehe").

    Checks for 2-char and 3-char repeating units with at least 2 repetitions.
    Tolerates a partial trailing match (e.g., "hahahah" matches pattern "ha").
    """
    if len(lower_word) < 4:
        return 0
    for pat_len in [2, 3]:
        pattern = lower_word[:pat_len]
        repeats = len(lower_word) // pat_len
        remainder = len(lower_word) % pat_len
        if repeats >= 2 and lower_word == pattern * repeats + pattern[:remainder]:
            return 1
    return 0


def _is_reduplicated(lower_word):
    """
    Detect Filipino morphological reduplication via hyphen.

    A reduplicated word contains a hyphen where the substring before the hyphen
    is identical to the substring after it (e.g., "iba-iba", "kain-kain",
    "bahay-bahay"). Both halves must be at least 2 characters long.
    """
    if '-' not in lower_word:
        return 0
    idx = lower_word.index('-')
    left = lower_word[:idx]
    right = lower_word[idx + 1:]
    if len(left) >= 2 and left == right:
        return 1
    return 0
"""
Unified ML Validator - Works for ANY data type

Train once on YOUR data, validate forever.
Works for: phone, email, address, name, or ANY custom column type.

Key: Uses YOUR training data, not pre-trained models.
All training data rows are treated as VALID examples.
"""

import os
import re
import random
import logging
import joblib
import numpy as np
import difflib
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml.feature_extractor import GenericFeatureExtractor

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _rapidfuzz

    def _similarity(a: str, b: str) -> float:
        """String similarity 0.0-1.0 (rapidfuzz, ~50x faster than difflib)."""
        return _rapidfuzz.ratio(a, b) / 100.0
except ImportError:  # pragma: no cover - fallback when rapidfuzz not installed
    def _similarity(a: str, b: str) -> float:
        """String similarity 0.0-1.0 (difflib fallback)."""
        return difflib.SequenceMatcher(None, a, b).ratio()


def _best_match(text: str, candidates) -> Tuple[Optional[str], float]:
    """Find the candidate most similar to text. Returns (match, similarity)."""
    best_val, best_sim = None, 0.0
    for candidate in candidates:
        sim = _similarity(text, candidate)
        if sim > best_sim:
            best_val, best_sim = candidate, sim
    return best_val, best_sim


class UnifiedMLValidator:
    """
    One unified validator for ALL data types.

    Train it on YOUR data CSV (any column names):
    - All rows are treated as valid examples
    - Synthetic invalid examples are generated automatically
    - One model file stores all column validators
    """

    # Numeric columns - don't suggest corrections (numbers don't have "typos")
    NUMERIC_COLUMNS = ['age', 'blood_sugar', 'salary', 'price', 'amount', 'percentage', 'income', 'cost', 'quantity']

    # Columns requiring high similarity for corrections (to avoid random suggestions)
    HIGH_SIMILARITY_COLUMNS = ['phone', 'email']
    HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similarity required

    # Regularization strengths to try during hyperparameter tuning
    C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

    # ML stage only flags a value when P(invalid) reaches this threshold.
    # Precision over recall: a 51% "maybe" from the classifier should not
    # paint a cell red - users stop trusting flags that are coin flips.
    ML_INVALID_THRESHOLD = 0.65

    # Base model validation rules (hardcoded for known columns)
    VALIDATION_RULES = {
        'age': {
            'type': 'numeric',
            'min': 0,
            'max': 120,
        },
        'blood_sugar': {
            'type': 'numeric',
            'min': 0,
            'max': 500,
        },
        'salary': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,  # 100 million max
        },
        'price': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,
        },
        'amount': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,
        },
        'income': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,
        },
        'cost': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,
        },
        'quantity': {
            'type': 'numeric',
            'min': 0,
            'max': 100000000,
        },
        'percentage': {
            'type': 'numeric',
            'min': 0,
            'max': 100,
        },
        'percent': {
            'type': 'numeric',
            'min': 0,
            'max': 100,
        },
        'phone': {
            'type': 'phone',
            'min_digits': 7,
        },
        'mobile': {
            'type': 'phone',
            'min_digits': 7,
        },
        'telephone': {
            'type': 'phone',
            'min_digits': 7,
        },
        'tel': {
            'type': 'phone',
            'min_digits': 7,
        },
        'email': {
            'type': 'email',
        },
        'mail': {
            'type': 'email',
        },
        'country': {
            'type': 'reference',
        },
        'gender': {
            'type': 'reference',
        },
        'currency': {
            'type': 'reference',
        },
        'status': {
            'type': 'reference',
        },
        'name': {
            'type': 'text',
        },
        'date': {
            'type': 'date',
        },
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize unified validator.

        Args:
            model_path: Path to load trained model (optional)
        """
        self.column_models = {}      # {col_name: LogisticRegression}
        self.column_scalers = {}     # {col_name: StandardScaler}
        self.column_vectorizers = {} # {col_name: TfidfVectorizer (char n-grams)}
        self.column_shape_vectorizers = {}  # {col_name: TfidfVectorizer (shape-token n-grams)}
        self.column_numeric_ranges = {}     # {col_name: (min, max)} learned from training data
        self.column_whitelists = {}  # {col_name: set of valid values}
        self.column_examples = {}    # {col_name: list of valid examples} (for corrector)
        self.reference_lists = {}    # {col_name: set of valid values from reference files}
        self.columns = []            # Ordered list of trained columns
        self.training_metrics = {}   # Per-column metrics
        self.categorical_columns = set()  # Columns auto-detected as categorical (finite valid values)
        self.is_trained = False
        self.model_name = "unnamed"
        self.feature_extractor = GenericFeatureExtractor()
        self._ref_list_file_cache = {}  # {col_name: set} cached reference_lists/*.txt loads

        # Try to load if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ── Rule resolution ──────────────────────────────────────

    @staticmethod
    def _column_tokens(column_name: str) -> set:
        """Split a column name into lowercase word tokens ('Customer Email' -> {customer, email})."""
        if not column_name:
            return set()
        return {t for t in re.split(r'[^a-z0-9]+', column_name.lower()) if t}

    @classmethod
    def _rule_key_for_column(cls, column_name: str) -> Optional[str]:
        """
        Resolve which VALIDATION_RULES entry applies to a column, matching
        whole word tokens only. 'customer_email' -> 'email', but
        'mailing_address' -> None and 'percentage' never matches 'age'.
        """
        if not column_name:
            return None
        col_lower = column_name.lower().strip()
        if col_lower in cls.VALIDATION_RULES:
            return col_lower

        tokens = cls._column_tokens(column_name)
        normalized = re.sub(r'[^a-z0-9]+', '', col_lower)

        best = None
        for key in cls.VALIDATION_RULES:
            key_tokens = key.split('_')
            key_normalized = key.replace('_', '')
            if all(t in tokens for t in key_tokens) or key_normalized == normalized:
                if best is None or len(key) > len(best):
                    best = key
        return best

    @staticmethod
    def _parse_numeric(text_str: str) -> Optional[float]:
        """Parse a numeric value, tolerating currency symbols, commas and %."""
        cleaned = text_str.replace(',', '').replace('$', '').replace('£', '').replace('€', '').replace('%', '').strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _apply_rules(self, text_str: str, column_name: str) -> Optional[Tuple[bool, float, Optional[str]]]:
        """
        Apply deterministic validation rules for a column.

        Returns (is_valid, confidence, reason) when a rule decides the value
        (reason is None for positive short-circuits), or None if no rule
        applies (validation continues to whitelist/ML).
        This is the single implementation shared by validate() and validate_batch().
        """
        rule_key = self._rule_key_for_column(column_name)
        if not rule_key:
            return None

        rules = self.VALIDATION_RULES[rule_key]
        rule_type = rules.get('type')
        label = rule_key.replace('_', ' ').title()

        if rule_type == 'numeric':
            num_val = self._parse_numeric(text_str)
            if num_val is None:
                return False, 0.95, f"{label} must be a valid number"
            if 'min' in rules and num_val < rules['min']:
                if num_val < 0:
                    return False, 0.95, f"{label} cannot be negative"
                return False, 0.95, f"{label} is below minimum ({rules['min']})"
            if 'max' in rules and num_val > rules['max']:
                return False, 0.90, f"{label} must be between {rules['min']} and {rules['max']}"

        elif rule_type == 'phone':
            digit_count = sum(1 for c in text_str if c.isdigit())
            if digit_count == 0:
                return False, 0.95, "Phone number must contain digits"
            if digit_count < rules.get('min_digits', 7):
                return False, 0.90, f"Incomplete phone number ({digit_count} digits, minimum {rules.get('min_digits', 7)} required)"
            # Well-formed phone (only digits and separators, plausible length)
            # = valid without consulting the ML model. Precision over recall:
            # a structurally perfect phone number should never be flagged.
            if digit_count <= 15 and re.fullmatch(r'\+?[\d\s().-]+', text_str):
                return True, 0.90, None

        elif rule_type == 'email':
            if '@' not in text_str:
                return False, 0.95, "Email must contain @ symbol"
            if text_str.count('@') > 1:
                return False, 0.95, "Email cannot have multiple @ symbols"
            domain_part = text_str.split('@')[-1]
            if '.' not in domain_part:
                return False, 0.90, "Email must have a valid domain (e.g., .com, .org)"
            # Well-formed address = valid (user@domain.tld with sane characters)
            if re.fullmatch(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_str):
                return True, 0.90, None

        elif rule_type == 'date':
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',      # DD/MM/YYYY or MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',      # DD-MM-YYYY
                r'\d{1,2}/\d{1,2}/\d{2,4}',  # D/M/YY or similar
            ]
            is_date_format = any(re.match(p, text_str) for p in date_patterns)
            if not is_date_format and not text_str.replace('-', '').replace('/', '').isdigit():
                return False, 0.85, "Invalid date format (use: YYYY-MM-DD, DD/MM/YYYY, etc.)"

        return None

    # ── Feature pipeline (char n-grams + shape n-grams + structural features) ──

    @staticmethod
    def _shape_encode(text: str) -> str:
        """
        Encode a string's character-class shape: letters -> x, digits -> d,
        other characters kept. 'P101' -> 'xddd', 'ORD-042' -> 'xxx-ddd'.
        Shape n-grams generalise better than raw characters for ID/code columns.
        """
        return ''.join('x' if c.isalpha() else 'd' if c.isdigit() else c for c in text)

    def _structural_matrix(self, texts: List[str], column_name: str) -> np.ndarray:
        """Extract the hand-crafted structural feature matrix for a list of texts."""
        X = np.array(
            [self.feature_extractor.extract_features(t, column_name) for t in texts],
            dtype=float,
        )
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_features(self, texts: List[str], column_name: str):
        """
        Fit the feature pipeline for a column on training texts and return the
        combined matrix: char n-gram TF-IDF + shape-token n-grams (both learned
        per column) + scaled structural features. Char n-grams learn this
        column's vocabulary, shape n-grams learn its character-class pattern
        (xddd for P101), structural features carry length/ratio/magnitude.
        """
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(1, 3), max_features=300, lowercase=True
        )
        tfidf = vectorizer.fit_transform(texts)
        self.column_vectorizers[column_name] = vectorizer

        shape_vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), max_features=100, lowercase=False
        )
        shapes = [self._shape_encode(t) for t in texts]
        shape_tfidf = shape_vectorizer.fit_transform(shapes)
        self.column_shape_vectorizers[column_name] = shape_vectorizer

        structural = self._structural_matrix(texts, column_name)
        scaler = StandardScaler()
        structural_scaled = np.nan_to_num(
            scaler.fit_transform(structural), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.column_scalers[column_name] = scaler

        return hstack([tfidf, shape_tfidf, csr_matrix(structural_scaled)]).tocsr()

    def _transform_features(self, texts: List[str], column_name: str):
        """Transform texts with the fitted pipeline for a column."""
        structural = self._structural_matrix(texts, column_name)
        scaler = self.column_scalers.get(column_name)
        if scaler is not None:
            structural = np.nan_to_num(
                scaler.transform(structural), nan=0.0, posinf=0.0, neginf=0.0
            )

        vectorizer = self.column_vectorizers.get(column_name)
        if vectorizer is None:
            # Models trained before v3.0 used structural features only
            return structural

        parts = [vectorizer.transform(texts)]

        shape_vectorizer = self.column_shape_vectorizers.get(column_name)
        if shape_vectorizer is not None:
            parts.append(shape_vectorizer.transform([self._shape_encode(t) for t in texts]))

        parts.append(csr_matrix(structural))
        return hstack(parts).tocsr()

    # Keyboard adjacency map for realistic typo simulation (QWERTY neighbours)
    _KEYBOARD_NEIGHBOURS = {
        'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg',
        'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
        'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'q': 'wa', 'r': 'et',
        's': 'ad', 't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
        'y': 'tu', 'z': 'x', '0': '9', '1': '2', '2': '13', '3': '24',
        '4': '35', '5': '46', '6': '57', '7': '68', '8': '79', '9': '80',
    }

    def _generate_invalid_examples(self, valid_examples: List[str], count: int,
                                   column_kind: str = 'open') -> List[str]:
        """
        Generate synthetic invalid data for training, matched to the column type.

        column_kind controls which error families are simulated:
        - 'finite' (categorical): realistic typos (transposed/adjacent-key/dropped
          chars) + structural corruption. A typo of a category IS invalid.
        - 'numeric': numeric entry errors (decimal shift, negation, extreme
          values) + structural corruption.
        - 'open' (names, addresses, IDs, free text): structural corruption ONLY.
          A one-character variation of an open-ended value is usually still a
          legitimate value (names have no canonical spelling), so training on
          near-identical negatives teaches the model to reject valid data.

        Mutations that collide with a real valid value are discarded so the
        classifier is never trained on contradictory labels.

        Args:
            valid_examples: List of valid example strings
            count: Number of invalid examples to generate
            column_kind: 'finite' | 'numeric' | 'open'

        Returns:
            List of synthetic invalid strings
        """
        if not valid_examples:
            return []

        valid_set = {str(v).strip().lower() for v in valid_examples}
        invalid = []

        def _transpose_chars(x):
            # Swap two adjacent characters: "12345" -> "12435", "John" -> "Jhon"
            if len(x) < 3:
                return x + "??"
            i = random.randint(0, len(x) - 2)
            return x[:i] + x[i + 1] + x[i] + x[i + 2:]

        def _keyboard_typo(x):
            # Replace a character with its keyboard neighbour: "gmail" -> "gnail"
            positions = [i for i, c in enumerate(x.lower()) if c in self._KEYBOARD_NEIGHBOURS]
            if not positions:
                return x + "#"
            i = random.choice(positions)
            replacement = random.choice(self._KEYBOARD_NEIGHBOURS[x[i].lower()])
            return x[:i] + replacement + x[i + 1:]

        def _drop_char(x):
            # Delete one character: "Singapore" -> "Singpore"
            if len(x) < 3:
                return ""
            i = random.randint(0, len(x) - 1)
            return x[:i] + x[i + 1:]

        def _double_char(x):
            # Duplicate one character: "email" -> "emaail"
            if not x:
                return "??"
            i = random.randint(0, len(x) - 1)
            return x[:i] + x[i] + x[i:]

        def _decimal_shift(x):
            # Unit/entry error: value 10x or 0.1x off (95.0 -> 950.0)
            return str(round(float(x) * random.choice([10, 100, 0.1]), 4))

        def _numeric_negate(x):
            return str(-abs(float(x)))

        def _numeric_extreme(x):
            return str(float(x) * 1000)

        structural_corruption = [
            lambda x: x[:len(x) // 2] if len(x) > 2 else "",          # Truncate
            lambda x: x + "???",                                       # Garbage suffix
            lambda x: ''.join(random.sample(x, len(x))) if len(x) > 1 else x + "X",  # Shuffle
            lambda x: "",                                              # Empty
            lambda x: "invalid_" + x[:3] if len(x) >= 3 else "invalid_X",  # Prefix garbage
            lambda x: ''.join(c if random.random() > 0.3 else chr(random.randint(33, 126)) for c in x),  # Random noise
        ]
        realistic_typos = [
            _transpose_chars,
            _transpose_chars,
            _keyboard_typo,
            _keyboard_typo,
            _drop_char,
            _double_char,
        ]
        numeric_errors = [
            _decimal_shift,
            _decimal_shift,
            _numeric_negate,
            _numeric_extreme,
        ]

        if column_kind == 'finite':
            mutations = realistic_typos + structural_corruption
        elif column_kind == 'numeric':
            mutations = numeric_errors + structural_corruption
        else:  # 'open'
            mutations = structural_corruption

        attempts = 0
        max_attempts = count * 4
        while len(invalid) < count and attempts < max_attempts:
            attempts += 1
            example = str(random.choice(valid_examples))
            mutation = random.choice(mutations)
            try:
                mutated = mutation(example)
            except (ValueError, TypeError):
                continue  # e.g. numeric mutation on non-numeric value

            # Never label a real valid value as invalid
            if mutated.strip().lower() in valid_set and mutated.strip():
                continue
            invalid.append(mutated)

        # Top up with unambiguous garbage if too many mutations collided
        while len(invalid) < count:
            invalid.append("")

        return invalid

    def train(self, df: pd.DataFrame, model_name: str = "custom",
              exclude_columns: List[str] = None, test_size: float = 0.2) -> Dict[str, dict]:
        """
        Train unified model on entire DataFrame.

        Args:
            df: DataFrame where ALL rows are valid examples
            model_name: Name for this model
            exclude_columns: Columns to skip (e.g., 'id', 'timestamp')
            test_size: Fraction for test split

        Returns:
            dict: Per-column training metrics
        """
        logger.info("Training Unified ML Validator '%s' (%d rows)", model_name, len(df))

        if exclude_columns is None:
            exclude_columns = []

        # Determine columns to train
        columns_to_train = [col for col in df.columns if col not in exclude_columns]
        logger.info("Training on %d columns: %s", len(columns_to_train), columns_to_train)

        if len(df) < 5:
            logger.warning("Very small dataset (< 5 rows). Model may not perform well.")

        self.model_name = model_name
        self.columns = columns_to_train
        self.training_metrics = {}
        self.categorical_columns = set()
        self.column_numeric_ranges = {}

        for col in columns_to_train:
            logger.info("Training column: %s", col)

            # Get valid examples (all rows as valid)
            valid_values = df[col].dropna().astype(str).tolist()
            valid_values = [v.strip() for v in valid_values if v.strip()]

            if not valid_values:
                logger.info("  Skipping '%s': no valid values", col)
                continue

            unique_valid = list(set(valid_values))
            logger.info("  Valid examples: %d (%d unique)", len(valid_values), len(unique_valid))

            # Auto-detect categorical columns (finite set of valid values)
            # Heuristic: fewer than 30% unique ratio AND at most 20 unique values
            unique_ratio = len(unique_valid) / max(len(df), 1)
            if unique_ratio < 0.3 and len(unique_valid) <= 20:
                self.categorical_columns.add(col)
                logger.info("  Auto-detected as categorical (%d unique values, %.1f%% ratio)", len(unique_valid), unique_ratio * 100)

            # Store whitelist and examples for correction
            self.column_whitelists[col] = set(v.lower() for v in unique_valid)
            self.column_examples[col] = unique_valid

            # Pick the synthetic-error family that matches the column type
            numeric_share = sum(1 for v in unique_valid if self._parse_numeric(v) is not None) / len(unique_valid)
            if col in self.categorical_columns:
                column_kind = 'finite'
            elif numeric_share >= 0.8:
                column_kind = 'numeric'
            else:
                column_kind = 'open'

            # For continuous numeric columns, learn the observed value range.
            # Validation rejects values far outside it (range stage).
            if column_kind == 'numeric':
                numeric_vals = [n for n in (self._parse_numeric(v) for v in unique_valid) if n is not None]
                if numeric_vals:
                    self.column_numeric_ranges[col] = (min(numeric_vals), max(numeric_vals))

            # Generate synthetic invalid examples
            invalid_count = max(len(valid_values), 50)  # At least 50 or match valid count
            invalid_values = self._generate_invalid_examples(unique_valid, invalid_count, column_kind)
            logger.info("  Generated %d synthetic invalid examples (%s)", len(invalid_values), column_kind)

            # Prepare training data
            all_texts = valid_values + invalid_values
            all_labels = [1] * len(valid_values) + [0] * len(invalid_values)

            y = all_labels

            col_metrics = {
                'total_samples': len(all_texts),
                'valid_count': len(valid_values),
                'invalid_count': len(invalid_values),
                'unique_valid': len(unique_valid),
            }

            if len(all_texts) >= 10 and min(len(valid_values), len(invalid_values)) >= 4:
                texts_train, texts_test, y_train, y_test = train_test_split(
                    all_texts, y, test_size=test_size, stratify=y, random_state=42
                )

                col_metrics['used_split'] = True
                col_metrics['train_size'] = len(texts_train)
                col_metrics['test_size'] = len(texts_test)

                # Fit feature pipeline (n-gram vectorizer + scaler) on training data only
                X_train_scaled = self._fit_features(texts_train, col)
                X_test_scaled = self._transform_features(texts_test, col)

                # Tune regularization strength C via cross-validation
                # Adaptive folds: at most 5, bounded by smallest class count
                min_class_count = min(
                    sum(1 for yi in y_train if yi == 1),
                    sum(1 for yi in y_train if yi == 0),
                )
                n_cv = min(5, min_class_count)

                if n_cv >= 2:
                    base_lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', solver='lbfgs')
                    grid = GridSearchCV(base_lr, {'C': self.C_GRID}, cv=n_cv, scoring='f1', n_jobs=-1, refit=True)
                    grid.fit(X_train_scaled, y_train)
                    self.column_models[col] = grid.best_estimator_
                    col_metrics['best_C'] = grid.best_params_['C']
                    col_metrics['cv_f1_score'] = round(grid.best_score_, 4)
                    logger.info("  Best C: %s (CV F1: %.1f%%)", grid.best_params_['C'], grid.best_score_ * 100)
                else:
                    # Not enough samples per class for CV - use default C
                    lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', solver='lbfgs')
                    lr.fit(X_train_scaled, y_train)
                    self.column_models[col] = lr
                    col_metrics['best_C'] = 1.0
                    col_metrics['cv_f1_score'] = None

                # Evaluate on held-out test set
                y_train_pred = self.column_models[col].predict(X_train_scaled)
                y_test_pred = self.column_models[col].predict(X_test_scaled)

                col_metrics.update(self._calculate_metrics(y_train, y_train_pred, 'train'))
                col_metrics.update(self._calculate_metrics(y_test, y_test_pred, 'test'))

                logger.info("  Train accuracy: %.1f%%", col_metrics['train_accuracy'] * 100)
                logger.info("  Test accuracy: %.1f%%", col_metrics['test_accuracy'] * 100)

            else:
                # Too small for split - train on all data
                col_metrics['used_split'] = False
                col_metrics['small_dataset_warning'] = True
                col_metrics['best_C'] = 1.0
                col_metrics['cv_f1_score'] = None

                X_scaled = self._fit_features(all_texts, col)
                lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', solver='lbfgs')
                lr.fit(X_scaled, y)
                self.column_models[col] = lr

                y_pred = self.column_models[col].predict(X_scaled)
                col_metrics.update(self._calculate_metrics(y, y_pred, 'train'))

                logger.info("  Train accuracy (no split): %.1f%%", col_metrics['train_accuracy'] * 100)

            self.training_metrics[col] = col_metrics

        self.is_trained = True
        logger.info("Training complete for %d columns", len(self.column_models))

        return self.training_metrics

    def _calculate_metrics(self, y_true, y_pred, prefix='') -> dict:
        """Calculate accuracy, precision, recall, F1, and confusion matrix."""
        metrics = {}
        key_prefix = f"{prefix}_" if prefix else ""

        metrics[f'{key_prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{key_prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    def _match_columns(self, input_columns: List[str]) -> Dict[str, str]:
        """
        Match input columns to trained columns.

        Strategy:
        1. Exact match (case-insensitive)
        2. Normalized match (remove _, -, spaces)
        3. Substring match

        Args:
            input_columns: List of column names from input DataFrame

        Returns:
            {input_col: trained_col} mapping
        """
        mapping = {}
        trained_cols_remaining = set(self.columns)

        def normalize(s):
            return s.lower().replace('_', '').replace('-', '').replace(' ', '')

        # Pass 1: Exact match (case-insensitive)
        for input_col in input_columns:
            for trained_col in trained_cols_remaining:
                if input_col.lower() == trained_col.lower():
                    mapping[input_col] = trained_col
                    trained_cols_remaining.discard(trained_col)
                    break

        # Pass 2: Normalized match
        for input_col in input_columns:
            if input_col in mapping:
                continue
            input_norm = normalize(input_col)
            for trained_col in trained_cols_remaining:
                if input_norm == normalize(trained_col):
                    mapping[input_col] = trained_col
                    trained_cols_remaining.discard(trained_col)
                    break

        # Pass 3: Substring match
        for input_col in input_columns:
            if input_col in mapping:
                continue
            input_norm = normalize(input_col)
            for trained_col in trained_cols_remaining:
                trained_norm = normalize(trained_col)
                if input_norm in trained_norm or trained_norm in input_norm:
                    mapping[input_col] = trained_col
                    trained_cols_remaining.discard(trained_col)
                    break

        return mapping

    def _load_reference_list_for_column(self, column_name: str) -> set:
        """Load reference list for a column from file if available (cached)."""
        if column_name in self._ref_list_file_cache:
            return self._ref_list_file_cache[column_name]

        values = set()
        filepath = os.path.join("reference_lists", f"{column_name}.txt")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            values.add(line.lower())
            except OSError as e:
                logger.warning("Could not read reference list %s: %s", filepath, e)

        self._ref_list_file_cache[column_name] = values
        return values

    def validate(self, text: str, column_name: str) -> Tuple[bool, float]:
        """
        Validate a single value for a specific column.

        Delegates to validate_batch() so single-value and batch validation
        share one implementation and can never disagree.

        Returns:
            (is_valid, confidence) where:
                is_valid: True if valid, False if invalid
                confidence: 0.0-1.0 confidence score
        """
        return self.validate_batch([text], column_name)[0]

    def validate_row(self, row: pd.Series, column_mapping: Dict[str, str] = None) -> Dict[str, Tuple[bool, float]]:
        """
        Validate entire row at once.

        Args:
            row: pandas Series representing a row
            column_mapping: Optional mapping from row columns to trained columns

        Returns:
            {column_name: (is_valid, confidence)}
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        results = {}

        if column_mapping is None:
            column_mapping = self._match_columns(row.index.tolist())

        for input_col, trained_col in column_mapping.items():
            if trained_col in self.column_models:
                value = str(row[input_col])
                results[input_col] = self.validate(value, trained_col)
            else:
                results[input_col] = (True, 0.5)  # Unknown column

        return results

    def validate_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Validate entire DataFrame.

        Args:
            df: DataFrame to validate
            column_mapping: Optional mapping from df columns to trained columns

        Returns:
            DataFrame with validation results per cell
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        if column_mapping is None:
            column_mapping = self._match_columns(df.columns.tolist())

        # Create result structure
        results = {col: [] for col in df.columns}

        for _, row in df.iterrows():
            row_results = self.validate_row(row, column_mapping)
            for col in df.columns:
                if col in row_results:
                    results[col].append(row_results[col])
                else:
                    results[col].append((True, 0.5))  # Unknown column

        return pd.DataFrame(results, index=df.index)

    def _unknown_value_reason(self, column_name: str) -> str:
        """Human-readable reason for a value outside a closed valid set."""
        rule_key = self._rule_key_for_column(column_name)
        if rule_key and self.VALIDATION_RULES[rule_key].get('type') == 'reference':
            return f"{rule_key.replace('_', ' ').title()} not recognized"

        examples = self.column_examples.get(column_name, [])
        if examples:
            sample = ', '.join(sorted(str(v) for v in examples[:8]))
            suffix = f' (+{len(examples) - 8} more)' if len(examples) > 8 else ''
            return f"Not a valid {column_name} (expected: {sample}{suffix})"
        return f"Not a recognized value for {column_name}"

    def validate_batch_detailed(self, texts: List[str], column_name: str) -> List[dict]:
        """
        Validate multiple values for a specific column.

        This is THE validation implementation — validate() and validate_batch()
        delegate here, so the verdict, confidence and explanation always come
        from the same pass.

        Pipeline per value:
          1. Empty / missing check
          2. Deterministic rules (age range, email format, phone digits, ...)
          3. Whitelist exact / numeric-normalized match
          4. Learned numeric range (continuous numeric columns)
          5. Fuzzy typo detection (categorical / reference-list columns only)
          6. Closed-set rejection (unknown category / reference value)
          7. ML classifier (char n-grams + shape n-grams + structural features)

        Returns:
            List of dicts: {'is_valid', 'confidence', 'stage', 'reason'}
            stage is one of: empty, rule, reference, not-trained, whitelist,
            range, typo, unknown-value, ml. reason is None for valid values.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        def entry(is_valid, confidence, stage, reason=None):
            return {'is_valid': is_valid, 'confidence': confidence, 'stage': stage, 'reason': reason}

        results: List[Optional[dict]] = [None] * len(texts)
        texts_for_ml = []
        ml_indices = []

        is_trained_column = column_name in self.column_models

        # Combined whitelist (training data + reference lists)
        whitelist = self._get_combined_whitelist(column_name) if is_trained_column else set()

        # Strict typo detection only for categorical columns or columns with reference lists.
        # High-cardinality columns (order_id, customer_name, etc.) should NOT use fuzzy
        # matching — new unseen values that follow the same pattern are valid, not typos.
        # The ML classifier handles those correctly via learned features.
        use_strict_typo_detection = (
            column_name in self.categorical_columns
            or column_name in self.reference_lists
        )

        numeric_range = self.column_numeric_ranges.get(column_name)

        # For untrained reference-type columns (gender, currency, status, country...)
        # fall back to the bundled reference list files.
        untrained_ref_list = set()
        if not is_trained_column and column_name:
            rule_key = self._rule_key_for_column(column_name)
            if rule_key and self.VALIDATION_RULES[rule_key].get('type') == 'reference':
                untrained_ref_list = self._load_reference_list_for_column(rule_key)

        for i, text in enumerate(texts):
            text_str = str(text).strip()
            text_lower = text_str.lower()

            # 1. Empty / missing value
            if not text_str or text_lower in ('nan', 'none', 'null'):
                results[i] = entry(False, 0.95, 'empty', 'Empty or missing value')
                continue

            # 2. Deterministic rules (run for trained AND untrained columns)
            rule_result = self._apply_rules(text_str, column_name)
            if rule_result is not None:
                is_valid, confidence, reason = rule_result
                results[i] = entry(is_valid, confidence, 'rule', reason)
                continue

            # Untrained column: reference list if available, else uncertain-valid
            if not is_trained_column:
                if untrained_ref_list:
                    if text_lower in untrained_ref_list:
                        results[i] = entry(True, 1.0, 'reference')
                    else:
                        match, best_sim = _best_match(text_lower, untrained_ref_list)
                        if best_sim >= 0.8:
                            results[i] = entry(False, best_sim, 'typo', f"Possible typo - did you mean '{match}'?")
                        else:
                            results[i] = entry(False, 0.7, 'unknown-value', self._unknown_value_reason(column_name))
                else:
                    results[i] = entry(True, 0.5, 'not-trained')  # Column not trained - uncertain
                continue

            # 3. Whitelist exact match
            if text_lower in whitelist:
                results[i] = entry(True, 1.0, 'whitelist')
                continue

            # Numeric-normalized whitelist match (95 == 95.0)
            parsed_numeric = None
            try:
                parsed_numeric = float(text_str)
                if (str(int(parsed_numeric)) in whitelist
                        or str(parsed_numeric) in whitelist
                        or f"{parsed_numeric:.1f}" in whitelist):
                    results[i] = entry(True, 1.0, 'whitelist')
                    continue
            except (ValueError, TypeError, OverflowError):
                pass

            # 4. Learned numeric range - values far outside what training data
            # showed for a continuous numeric column are invalid
            if numeric_range is not None and parsed_numeric is not None:
                lo, hi = numeric_range
                margin = (hi - lo) * 0.1 if hi > lo else abs(hi) * 0.1
                if parsed_numeric < lo - margin or parsed_numeric > hi + margin:
                    results[i] = entry(False, 0.90, 'range',
                                       f"Outside the range seen in training data ({lo:g} to {hi:g})")
                    continue

            # 5. Fuzzy typo detection (finite-valued columns only) - best match
            # so the flagged similarity refers to the same value we'd suggest
            if use_strict_typo_detection:
                match, best_sim = _best_match(text_lower, whitelist)
                if best_sim >= 0.8:  # High similarity but not exact = typo
                    results[i] = entry(False, best_sim, 'typo', f"Possible typo - did you mean '{match}'?")
                    continue

                # 6. Closed set: unknown value that isn't a near-typo is
                # invalid - no ML guessing
                results[i] = entry(False, 0.85, 'unknown-value', self._unknown_value_reason(column_name))
                continue

            # 7. ML classifier
            texts_for_ml.append(text_str)
            ml_indices.append(i)

        # Batch-predict all values that reached the ML stage
        if texts_for_ml:
            try:
                model = self.column_models[column_name]
                X = self._transform_features(texts_for_ml, column_name)
                probabilities = model.predict_proba(X)
                invalid_class_idx = list(model.classes_).index(0)

                for idx, prob in zip(ml_indices, probabilities):
                    p_invalid = float(prob[invalid_class_idx])
                    if p_invalid >= self.ML_INVALID_THRESHOLD:
                        results[idx] = entry(False, p_invalid, 'ml',
                                             "Doesn't match the pattern learned from training data")
                    else:
                        results[idx] = entry(True, 1.0 - p_invalid, 'ml')
            except ValueError as e:
                # Feature-shape mismatch: model was trained with an older
                # feature pipeline. Whitelist/rule stages still work.
                logger.error("ML stage failed for column '%s' (model needs retraining): %s", column_name, e)
                for idx in ml_indices:
                    results[idx] = entry(True, 0.5, 'ml', None)

        return results

    def validate_batch(self, texts: List[str], column_name: str) -> List[Tuple[bool, float]]:
        """
        Validate multiple values for a column. Thin wrapper around
        validate_batch_detailed() returning (is_valid, confidence) tuples.
        """
        return [(r['is_valid'], r['confidence'])
                for r in self.validate_batch_detailed(texts, column_name)]

    def correct(self, value: str, column_name: str) -> Optional[str]:
        """
        Suggest correction using column's valid examples.
        Uses difflib similarity matching with smart thresholds.

        Args:
            value: Value to correct
            column_name: Column name this value belongs to

        Returns:
            Corrected value or None if no good match / not appropriate to suggest
        """
        if not self.is_trained:
            return None

        if column_name not in self.column_examples:
            return None

        # Don't suggest corrections for empty, NaN, or whitespace-only strings
        if not value or str(value).strip() == '' or str(value).lower() in ['nan', 'none', 'null']:
            return None

        text_str = str(value).strip()
        col_tokens = self._column_tokens(column_name)

        # RULE 1: Never suggest corrections for numeric columns
        # Numbers don't have "typos" - a wrong number is just wrong
        # (token match so 'percentage'/'mileage' never match 'age')
        if any(num_col in col_tokens or num_col.replace('_', '') in col_tokens
               for num_col in self.NUMERIC_COLUMNS):
            return None  # No correction for age, blood_sugar, etc.
        rule_key = self._rule_key_for_column(column_name)
        if rule_key and self.VALIDATION_RULES[rule_key].get('type') == 'numeric':
            return None

        # RULE 2: Simple email typo fixes (double @@ -> single @)
        if rule_key in ('email', 'mail'):
            if '@@' in text_str:
                # Fix double @@ to single @
                fixed_email = text_str.replace('@@', '@')
                # Only suggest if the fixed version looks valid
                if '@' in fixed_email and fixed_email.count('@') == 1:
                    domain = fixed_email.split('@')[-1]
                    if '.' in domain:
                        return fixed_email

        # RULE 3: For phone/email, require very high similarity (85%)
        # Prevents suggesting random phone numbers for incomplete inputs
        is_phone_column = rule_key in ('phone', 'mobile', 'telephone', 'tel')
        is_high_similarity_column = is_phone_column or rule_key in ('email', 'mail')

        # RULE 4: For phone numbers, also check digit count similarity
        if is_phone_column:
            input_digits = sum(1 for c in text_str if c.isdigit())
            # If input has very few digits, don't suggest
            if input_digits < 5:
                return None  # Too incomplete to suggest

        valid_examples = self.column_examples[column_name]
        if not valid_examples:
            return None

        text_lower = text_str.lower()

        # Determine similarity threshold based on column type
        if is_high_similarity_column:
            similarity_threshold = self.HIGH_SIMILARITY_THRESHOLD  # 85%
        elif rule_key == 'country':
            similarity_threshold = 0.7  # 70% for countries (typo detection)
        else:
            similarity_threshold = 0.6  # 60% default for names, etc.

        # Find all matches above threshold
        candidates = []
        for valid_example in valid_examples:
            valid_example_str = str(valid_example)
            similarity = difflib.SequenceMatcher(None, text_lower, valid_example_str.lower()).ratio()

            if similarity >= similarity_threshold:
                # For phone numbers, also check digit count is similar
                if is_phone_column:
                    input_digits = sum(1 for c in text_str if c.isdigit())
                    example_digits = sum(1 for c in valid_example_str if c.isdigit())
                    # Require digit counts to be within 2 of each other
                    if abs(input_digits - example_digits) > 2:
                        continue

                candidates.append((valid_example_str, similarity))

        if not candidates:
            return None

        # Sort by similarity (desc), then by canonical form preference
        def canonical_score(s):
            if len(s) <= 3 and s.isupper():
                return 0  # Best for short: "USA", "UK"
            elif s.istitle():
                return 1  # Best for long: "Singapore"
            elif s[0].isupper() if s else False:
                return 2  # OK: "United States"
            elif s.isupper():
                return 3  # Less preferred for long
            else:
                return 4  # Least preferred

        candidates.sort(key=lambda x: (-x[1], canonical_score(x[0]), len(x[0])))
        best_match, best_similarity = candidates[0]

        # Only return if not identical to input and similarity is meaningful
        if text_str.lower() != best_match.lower():
            return best_match

        return None

    def explain_invalidity(self, text: str, column_name: str = None) -> str:
        """
        Explain why a text is considered invalid.

        Args:
            text: Text to analyze
            column_name: Optional column name for context

        Returns:
            Human-readable explanation of what's wrong
        """
        if not self.is_trained:
            return "Model not trained"

        # Check for empty or whitespace first
        if not text or str(text).strip() == '' or str(text).lower() in ['nan', 'none', 'null']:
            return "Empty or missing value"

        text_str = str(text).strip()
        issues = []

        # Check deterministic rules using the same rule resolution as validation
        if column_name:
            rule_key = self._rule_key_for_column(column_name)

            if rule_key:
                rules = self.VALIDATION_RULES[rule_key]
                rule_type = rules.get('type')
                label = rule_key.replace('_', ' ').title()

                # Numeric column validation (age, blood_sugar, salary, ...)
                if rule_type == 'numeric':
                    num_val = self._parse_numeric(text_str)
                    if num_val is None:
                        return f"{label} must be a valid number"
                    if 'min' in rules and num_val < rules['min']:
                        if num_val < 0:
                            return f"{label} cannot be negative"
                        return f"{label} is below minimum ({rules['min']})"
                    if 'max' in rules and num_val > rules['max']:
                        return f"{label} must be between {rules['min']} and {rules['max']}"

                # Phone validation
                elif rule_type == 'phone':
                    digit_count = sum(1 for c in text_str if c.isdigit())
                    if digit_count == 0:
                        return "Phone number must contain digits"
                    if digit_count < rules.get('min_digits', 7):
                        return f"Incomplete phone number ({digit_count} digits, minimum 7 required)"

                # Email validation
                elif rule_type == 'email':
                    if '@' not in text_str:
                        return "Email must contain @ symbol"
                    if text_str.count('@') > 1:
                        return "Email cannot have multiple @ symbols"
                    domain_part = text_str.split('@')[-1]
                    if '.' not in domain_part:
                        return "Email must have a valid domain (e.g., .com, .org)"

                # Name validation
                elif rule_key == 'name':
                    if any(c.isdigit() for c in text_str):
                        return "Name should not contain numbers"
                    if len(text_str) < 2:
                        return "Name is too short"

                # Country validation (check against reference list)
                elif rule_key == 'country':
                    if 'country' in self.reference_lists:
                        # Check for close typos
                        ref_list = self.reference_lists['country']
                        best_match = None
                        best_similarity = 0
                        for valid in ref_list:
                            sim = difflib.SequenceMatcher(None, text_str.lower(), valid.lower()).ratio()
                            if sim > best_similarity:
                                best_similarity = sim
                                best_match = valid
                        if best_similarity > 0.7 and best_similarity < 1.0:
                            return f"Possible typo - did you mean '{best_match}'?"
                        return "Country not recognized"

                # Gender validation (check against reference list)
                elif rule_key == 'gender':
                    valid_genders = ['male', 'female', 'm', 'f', 'other', 'non-binary', 'prefer not to say', 'unknown']
                    if text_str.lower() not in valid_genders:
                        return "Gender not recognized (use: Male, Female, M, F, Other, etc.)"

                # Currency validation (check against reference list)
                elif rule_key == 'currency':
                    return "Currency code not recognized (use: USD, EUR, GBP, SGD, etc.)"

                # Status validation (check against reference list)
                elif rule_key == 'status':
                    return "Status not recognized (use: Active, Inactive, Pending, etc.)"

                # Date validation
                elif rule_type == 'date':
                    return "Invalid date format (use: YYYY-MM-DD, DD/MM/YYYY, etc.)"

        # Categorical column - value not in the valid set seen during training
        if column_name and column_name in self.categorical_columns:
            examples = self.column_examples.get(column_name, [])
            if examples:
                sample = ', '.join(sorted(str(v) for v in examples[:8]))
                suffix = f' (+{len(examples) - 8} more)' if len(examples) > 8 else ''
                return f"Not a valid {column_name} (expected: {sample}{suffix})"
            return f"Not a recognized value for {column_name}"

        # Fallback: generic structural analysis
        text_l = text_str.lower()
        if any(text_l[i] == text_l[i + 1] == text_l[i + 2] for i in range(len(text_l) - 2)):
            issues.append("contains repeated characters")

        # Length checks for unknown columns
        if len(text_str) < 2:
            issues.append("value too short")
        elif len(text_str) > 200:
            issues.append("value too long")

        if not issues:
            issues.append("pattern doesn't match expected format")

        return issues[0].capitalize()

    def save(self, filepath: str):
        """
        Save unified model to single file.

        Args:
            filepath: Path to save model (e.g., "models/my_data_model.pkl")
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model_name': self.model_name,
            'columns': self.columns,
            'column_models': self.column_models,
            'column_scalers': self.column_scalers,
            'column_vectorizers': self.column_vectorizers,
            'column_shape_vectorizers': self.column_shape_vectorizers,
            'column_numeric_ranges': self.column_numeric_ranges,
            'column_whitelists': self.column_whitelists,
            'column_examples': self.column_examples,
            'reference_lists': self.reference_lists,
            'training_metrics': self.training_metrics,
            'categorical_columns': self.categorical_columns,
            'is_trained': self.is_trained,
            'version': '3.1',  # Char + shape n-grams, learned numeric ranges, staged pipeline
        }

        joblib.dump(model_data, filepath)
        logger.info("Model saved to %s", filepath)

    def load(self, filepath: str) -> bool:
        """
        Load unified model from file.

        Args:
            filepath: Path to saved model

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.warning("Model file not found: %s", filepath)
            return False

        try:
            model_data = joblib.load(filepath)

            # Check version
            version = model_data.get('version', '1.0')
            if version != '3.1':
                logger.warning(
                    "Model format v%s predates the current feature pipeline (v3.1). "
                    "Rule/whitelist validation still works but the ML stage may be "
                    "degraded - retraining is recommended.", version,
                )

            self.model_name = model_data.get('model_name', 'unknown')
            self.columns = model_data.get('columns', [])
            self.column_models = model_data.get('column_models', {})
            self.column_scalers = model_data.get('column_scalers', {})
            # Pre-3.0 models have no vectorizers; _transform_features falls back
            # to structural features only for them
            self.column_vectorizers = model_data.get('column_vectorizers', {})
            self.column_shape_vectorizers = model_data.get('column_shape_vectorizers', {})
            self.column_numeric_ranges = model_data.get('column_numeric_ranges', {})
            self.column_whitelists = model_data.get('column_whitelists', {})
            self.column_examples = model_data.get('column_examples', {})
            self.reference_lists = model_data.get('reference_lists', {})
            self.training_metrics = model_data.get('training_metrics', {})
            self.categorical_columns = model_data.get('categorical_columns', set())
            self.is_trained = model_data.get('is_trained', False)

            return True
        except Exception as e:
            logger.error("Error loading model %s: %s", filepath, e)
            return False

    def get_trained_columns(self) -> List[str]:
        """Get list of trained column names."""
        return self.columns.copy()

    def get_column_metrics(self, column_name: str) -> dict:
        """Get training metrics for a specific column."""
        return self.training_metrics.get(column_name, {})

    def add_training_data(self, df: pd.DataFrame, retrain: bool = True) -> Dict[str, dict]:
        """
        Add more training data to an existing model (stacking).

        This allows users to incrementally improve their model by adding
        more valid examples without starting from scratch.

        Args:
            df: DataFrame with additional valid examples
            retrain: Whether to retrain the model after adding data

        Returns:
            dict: Updated training metrics (if retrain=True)
        """
        if not self.is_trained:
            raise ValueError("No existing model to add to. Use train() first.")

        logger.info("Adding %d rows to existing model '%s'", len(df), self.model_name)

        # Add new examples to existing whitelists and examples
        for col in df.columns:
            if col in self.columns:
                # Get new valid values
                new_values = df[col].dropna().astype(str).tolist()
                new_values = [v.strip() for v in new_values if v.strip()]

                if not new_values:
                    continue

                # Add to whitelist (lowercase for matching)
                if col not in self.column_whitelists:
                    self.column_whitelists[col] = set()
                self.column_whitelists[col].update(v.lower() for v in new_values)

                # Add to examples (original case for corrections)
                if col not in self.column_examples:
                    self.column_examples[col] = []
                existing = set(self.column_examples[col])
                for v in new_values:
                    if v not in existing:
                        self.column_examples[col].append(v)

                logger.info("  %s: Added %d examples (total whitelist: %d)", col, len(new_values), len(self.column_whitelists[col]))
            else:
                logger.info("  %s: Skipped (not in trained columns)", col)

        # Optionally retrain ML models with combined data
        if retrain:
            logger.info("Retraining ML models with combined data...")
            # We need to retrain from scratch with all examples
            # Build combined training data from whitelists
            combined_data = {}
            for col in self.columns:
                if col in self.column_examples:
                    combined_data[col] = self.column_examples[col]

            # Create DataFrame from examples (may have different lengths)
            max_len = max(len(v) for v in combined_data.values()) if combined_data else 0
            for col in combined_data:
                # Pad shorter columns by repeating values
                while len(combined_data[col]) < max_len:
                    combined_data[col].extend(combined_data[col][:max_len - len(combined_data[col])])
                combined_data[col] = combined_data[col][:max_len]

            combined_df = pd.DataFrame(combined_data)
            return self.train(combined_df, self.model_name)

        return self.training_metrics

    def load_reference_list(self, column_name: str, filepath: str) -> bool:
        """
        Load a reference list of valid values for a column.

        This supplements the training data whitelist with additional valid values.
        Useful for standardized fields like countries, currencies, etc.

        Args:
            column_name: Column name to associate the reference list with
            filepath: Path to text file with one valid value per line

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.warning("Reference list not found: %s", filepath)
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                values = [line.strip().lower() for line in f if line.strip()]

            if column_name not in self.reference_lists:
                self.reference_lists[column_name] = set()

            self.reference_lists[column_name].update(values)

            # Also add to column_examples for corrections
            if column_name not in self.column_examples:
                self.column_examples[column_name] = []

            # Add original case versions for corrections
            with open(filepath, 'r', encoding='utf-8') as f:
                original_values = [line.strip() for line in f if line.strip()]

            existing = set(self.column_examples[column_name])
            for val in original_values:
                if val not in existing:
                    self.column_examples[column_name].append(val)

            logger.info("Loaded %d values for '%s' from %s", len(values), column_name, filepath)
            return True
        except Exception as e:
            logger.error("Error loading reference list %s: %s", filepath, e)
            return False

    def load_reference_lists_from_dir(self, directory: str) -> Dict[str, int]:
        """
        Load all reference lists from a directory.

        Files should be named {column_name}.txt (e.g., countries.txt, currencies.txt)

        Args:
            directory: Path to directory containing reference list files

        Returns:
            Dict mapping column names to number of values loaded
        """
        results = {}
        if not os.path.exists(directory):
            logger.warning("Reference directory not found: %s", directory)
            return results

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                column_name = filename.replace('.txt', '')
                filepath = os.path.join(directory, filename)

                if self.load_reference_list(column_name, filepath):
                    results[column_name] = len(self.reference_lists.get(column_name, set()))

        return results

    def _get_combined_whitelist(self, column_name: str) -> set:
        """Get combined whitelist from training data and reference lists."""
        combined = set()

        if column_name in self.column_whitelists:
            combined.update(self.column_whitelists[column_name])

        if column_name in self.reference_lists:
            combined.update(self.reference_lists[column_name])

        return combined


# Keep backward compatibility alias
GenericMLValidator = UnifiedMLValidator


if __name__ == "__main__":
    # Example: Train a unified model
    print("=" * 70)
    print("UNIFIED ML VALIDATOR - TEST")
    print("=" * 70)

    # Create sample DataFrame
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.org', 'bob@company.net', 'alice@mail.com'],
        'phone': ['+1234567890', '+65 9123 4567', '+44 20 1234 5678', '+1 555 123 4567'],
    }
    df = pd.DataFrame(sample_data)

    print("\nSample training data:")
    print(df)

    # Create and train validator
    validator = UnifiedMLValidator()
    metrics = validator.train(df, "sample_model")

    # Save
    validator.save("models/sample_model.pkl")

    # Test validation
    print("\n" + "=" * 70)
    print("VALIDATION TEST")
    print("=" * 70)

    test_data = {
        'name': ['John Doe', 'J', 'Invalid123'],
        'email': ['john@example.com', 'invalid', 'missing@at'],
        'phone': ['+1234567890', '123', 'invalid_phone'],
    }
    test_df = pd.DataFrame(test_data)

    print("\nTest data:")
    print(test_df)

    print("\nValidation results:")
    for col in test_df.columns:
        print(f"\n{col}:")
        results = validator.validate_batch(test_df[col].tolist(), col)
        for val, (is_valid, conf) in zip(test_df[col], results):
            status = "VALID" if is_valid else "INVALID"
            print(f"  {val:<25} -> {status} ({conf:.1%})")

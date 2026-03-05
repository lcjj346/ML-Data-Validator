"""
Unified ML Validator - Works for ANY data type

Train once on YOUR data, validate forever.
Works for: phone, email, address, name, or ANY custom column type.

Key: Uses YOUR training data, not pre-trained models.
All training data rows are treated as VALID examples.
"""

import os
import random
import joblib
import difflib
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml.feature_extractor import GenericFeatureExtractor


class UnifiedMLValidator:
    """
    One unified validator for ALL data types.

    Train it on YOUR data CSV (any column names):
    - All rows are treated as valid examples
    - Synthetic invalid examples are generated automatically
    - One model file stores all column validators
    """

    # Columns that are open-ended (no strict typo detection)
    # These columns have unlimited valid values, so fuzzy matching would cause false positives
    OPEN_ENDED_COLUMNS = ['name', 'phone', 'address', 'email', 'age', 'blood_sugar', 'salary', 'price', 'amount', 'percentage', 'date']

    # Numeric columns - don't suggest corrections (numbers don't have "typos")
    NUMERIC_COLUMNS = ['age', 'blood_sugar', 'salary', 'price', 'amount', 'percentage', 'income', 'cost', 'quantity']

    # Columns requiring high similarity for corrections (to avoid random suggestions)
    HIGH_SIMILARITY_COLUMNS = ['phone', 'email']
    HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similarity required

    # Regularization strengths to try during hyperparameter tuning
    C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

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
        'phone': {
            'type': 'pattern',
            'min_digits': 7,
        },
        'email': {
            'type': 'pattern',
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
        self.column_whitelists = {}  # {col_name: set of valid values}
        self.column_examples = {}    # {col_name: list of valid examples} (for corrector)
        self.reference_lists = {}    # {col_name: set of valid values from reference files}
        self.columns = []            # Ordered list of trained columns
        self.training_metrics = {}   # Per-column metrics
        self.categorical_columns = set()  # Columns auto-detected as categorical (finite valid values)
        self.is_trained = False
        self.model_name = "unnamed"
        self.feature_extractor = GenericFeatureExtractor()

        # Try to load if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _generate_invalid_examples(self, valid_examples: List[str], count: int) -> List[str]:
        """
        Generate synthetic invalid data for training.

        Args:
            valid_examples: List of valid example strings
            count: Number of invalid examples to generate

        Returns:
            List of synthetic invalid strings
        """
        if not valid_examples:
            return []

        invalid = []
        mutations = [
            lambda x: x[:len(x)//2] if len(x) > 2 else "",           # Truncate
            lambda x: x + "???",                                      # Add garbage suffix
            lambda x: "???" + x,                                      # Add garbage prefix
            lambda x: ''.join(random.sample(x, len(x))) if len(x) > 1 else x + "X",  # Shuffle
            lambda x: x.replace(x[0], '@') if x else "@",            # Replace first char
            lambda x: "",                                             # Empty
            lambda x: "invalid_" + x[:3] if len(x) >= 3 else "invalid_X",  # Prefix garbage
            lambda x: x[:len(x)//3] if len(x) > 3 else "X",          # Severe truncate
            lambda x: x + x if len(x) < 20 else x[:10] + "###" + x[-10:],  # Duplicate or corrupt middle
            lambda x: ''.join(c if random.random() > 0.3 else chr(random.randint(33, 126)) for c in x),  # Random char replacement
            lambda x: x.upper() + "123" if x.islower() else x.lower() + "ABC",  # Case change + garbage
            lambda x: "   " + x + "   " if random.random() > 0.5 else "\t\t",  # Extra whitespace or just tabs
        ]

        for _ in range(count):
            example = random.choice(valid_examples)
            mutation = random.choice(mutations)
            try:
                mutated = mutation(str(example))
                invalid.append(mutated)
            except Exception:
                invalid.append("")  # Fallback to empty string

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
        print(f"Training Unified ML Validator '{model_name}'...")
        print(f"Total rows: {len(df)}")

        if exclude_columns is None:
            exclude_columns = []

        # Determine columns to train
        columns_to_train = [col for col in df.columns if col not in exclude_columns]
        print(f"Training on {len(columns_to_train)} columns: {columns_to_train}")

        if len(df) < 5:
            print("WARNING: Very small dataset (< 5 rows). Model may not perform well.")

        self.model_name = model_name
        self.columns = columns_to_train
        self.training_metrics = {}
        self.categorical_columns = set()

        for col in columns_to_train:
            print(f"\n--- Training column: {col} ---")

            # Get valid examples (all rows as valid)
            valid_values = df[col].dropna().astype(str).tolist()
            valid_values = [v.strip() for v in valid_values if v.strip()]

            if not valid_values:
                print(f"  Skipping '{col}': no valid values")
                continue

            unique_valid = list(set(valid_values))
            print(f"  Valid examples: {len(valid_values)} ({len(unique_valid)} unique)")

            # Auto-detect categorical columns (finite set of valid values)
            # Heuristic: fewer than 30% unique ratio AND at most 20 unique values
            unique_ratio = len(unique_valid) / max(len(df), 1)
            if unique_ratio < 0.3 and len(unique_valid) <= 20:
                self.categorical_columns.add(col)
                print(f"  Auto-detected as categorical ({len(unique_valid)} unique values, {unique_ratio:.1%} ratio)")

            # Store whitelist and examples for correction
            self.column_whitelists[col] = set(v.lower() for v in unique_valid)
            self.column_examples[col] = unique_valid

            # Generate synthetic invalid examples
            invalid_count = max(len(valid_values), 50)  # At least 50 or match valid count
            invalid_values = self._generate_invalid_examples(unique_valid, invalid_count)
            print(f"  Generated {len(invalid_values)} synthetic invalid examples")

            # Prepare training data
            all_texts = valid_values + invalid_values
            all_labels = [1] * len(valid_values) + [0] * len(invalid_values)

            # Extract features
            X = [self.feature_extractor.extract_features(text, col) for text in all_texts]
            y = all_labels

            # Initialize scaler
            self.column_scalers[col] = StandardScaler()

            col_metrics = {
                'total_samples': len(all_texts),
                'valid_count': len(valid_values),
                'invalid_count': len(invalid_values),
                'unique_valid': len(unique_valid),
            }

            if len(all_texts) >= 10 and min(len(valid_values), len(invalid_values)) >= 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )

                col_metrics['used_split'] = True
                col_metrics['train_size'] = len(X_train)
                col_metrics['test_size'] = len(X_test)

                # Fit scaler on training data only
                X_train_scaled = self.column_scalers[col].fit_transform(X_train)
                X_test_scaled = self.column_scalers[col].transform(X_test)

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
                    print(f"  Best C: {grid.best_params_['C']} (CV F1: {grid.best_score_:.1%})")
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

                print(f"  Train accuracy: {col_metrics['train_accuracy']:.1%}")
                print(f"  Test accuracy: {col_metrics['test_accuracy']:.1%}")

            else:
                # Too small for split - train on all data
                col_metrics['used_split'] = False
                col_metrics['small_dataset_warning'] = True
                col_metrics['best_C'] = 1.0
                col_metrics['cv_f1_score'] = None

                X_scaled = self.column_scalers[col].fit_transform(X)
                lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', solver='lbfgs')
                lr.fit(X_scaled, y)
                self.column_models[col] = lr

                y_pred = self.column_models[col].predict(X_scaled)
                col_metrics.update(self._calculate_metrics(y, y_pred, 'train'))

                print(f"  Train accuracy (no split): {col_metrics['train_accuracy']:.1%}")

            self.training_metrics[col] = col_metrics

        self.is_trained = True
        print(f"\nTraining complete for {len(self.column_models)} columns!")

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
        """Load reference list for a column from file if available."""
        import os
        ref_dir = "reference_lists"
        if not os.path.exists(ref_dir):
            return set()

        # Try exact filename match
        filepath = os.path.join(ref_dir, f"{column_name}.txt")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    values = set()
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            values.add(line.lower())
                    return values
            except Exception:
                pass
        return set()

    def validate(self, text: str, column_name: str) -> Tuple[bool, float]:
        """
        Validate a single value for a specific column.

        Args:
            text: Text to validate
            column_name: Column name this value belongs to

        Returns:
            (is_valid, confidence) where:
                is_valid: True if valid, False if invalid
                confidence: 0.0-1.0 confidence score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        text_str = str(text).strip()
        text_lower = text_str.lower()
        col_lower = column_name.lower() if column_name else ''

        # HARDCODED VALIDATION for known base model columns
        # This catches invalid values BEFORE checking if column is trained
        # Allows validation of untrained columns like salary, percentage, etc.

        # Age validation
        if 'age' in col_lower:
            try:
                age_val = float(text_str)
                if age_val < 0:
                    return False, 0.95  # Negative age is definitely invalid
                if age_val > 120:
                    return False, 0.90  # Age > 120 is invalid
            except (ValueError, TypeError):
                return False, 0.95  # Non-numeric age is invalid

        # Blood sugar validation
        if 'blood_sugar' in col_lower or 'bloodsugar' in col_lower:
            try:
                bs_val = float(text_str)
                if bs_val < 0:
                    return False, 0.95  # Negative blood sugar is invalid
                if bs_val > 500:
                    return False, 0.85  # Extremely high blood sugar is suspicious
            except (ValueError, TypeError):
                return False, 0.95  # Non-numeric blood sugar is invalid

        # Phone validation - minimum digits check
        if 'phone' in col_lower or 'mobile' in col_lower or 'tel' in col_lower:
            digit_count = sum(1 for c in text_str if c.isdigit())
            if digit_count == 0:
                return False, 0.95  # Phone must have digits
            if digit_count < 7:
                return False, 0.90  # Incomplete phone number

        # Email validation - must have @ and domain
        if 'email' in col_lower or 'mail' in col_lower:
            if '@' not in text_str:
                return False, 0.95  # Email must have @
            if text_str.count('@') > 1:
                return False, 0.95  # Multiple @ is invalid
            domain_part = text_str.split('@')[-1]
            if '.' not in domain_part:
                return False, 0.90  # Must have domain extension

        # Salary/Price/Amount/Income/Cost validation - must be non-negative numeric
        if any(kw in col_lower for kw in ['salary', 'price', 'amount', 'income', 'cost', 'quantity']):
            try:
                val = float(text_str.replace(',', '').replace('$', '').replace('£', '').replace('€', ''))
                if val < 0:
                    return False, 0.95  # Cannot be negative
            except (ValueError, TypeError):
                return False, 0.95  # Must be numeric

        # Percentage validation - must be 0-100
        if 'percent' in col_lower or col_lower.endswith('%'):
            try:
                pct_val = float(text_str.replace('%', ''))
                if pct_val < 0:
                    return False, 0.95  # Cannot be negative
                if pct_val > 100:
                    return False, 0.90  # Cannot exceed 100%
            except (ValueError, TypeError):
                return False, 0.95  # Must be numeric

        # Date validation - basic format check
        if 'date' in col_lower:
            import re
            # Check for common date patterns
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                r'\d{1,2}/\d{1,2}/\d{2,4}',  # D/M/YY or similar
            ]
            is_date_format = any(re.match(pattern, text_str) for pattern in date_patterns)
            if not is_date_format and not text_str.replace('-', '').replace('/', '').isdigit():
                return False, 0.85  # Doesn't look like a date

        # Check if column is trained - if not and passed hardcoded rules, return valid with low confidence
        if column_name not in self.column_models:
            # Column not trained in ML model
            # If we got here, it passed hardcoded rules (if any applied)
            # Check reference lists for untrained columns
            if column_name and col_lower in ['gender', 'currency', 'status']:
                # Load reference list if available
                ref_list = self._load_reference_list_for_column(col_lower)
                if ref_list:
                    if text_lower in ref_list:
                        return True, 1.0
                    # Check for typos
                    for valid in ref_list:
                        sim = difflib.SequenceMatcher(None, text_lower, valid.lower()).ratio()
                        if sim >= 0.8:
                            return False, sim  # Likely typo
                    return False, 0.7  # Not in reference list
            return True, 0.5  # Uncertain - column not trained

        # Get combined whitelist (training data + reference lists)
        whitelist = self._get_combined_whitelist(column_name)

        if whitelist:
            # Exact match = definitely valid
            if text_lower in whitelist:
                return True, 1.0

            # For numeric values, try normalized comparison (95 == 95.0)
            try:
                numeric_val = float(text_str)
                # Check both integer and float string representations
                if str(int(numeric_val)) in whitelist or str(numeric_val) in whitelist or f"{numeric_val:.1f}" in whitelist:
                    return True, 1.0
            except (ValueError, TypeError):
                pass  # Not numeric, continue with normal flow

            # Strict typo detection only for categorical columns or columns with reference lists.
            # High-cardinality columns (order_id, customer_name, etc.) should NOT use fuzzy
            # matching — new unseen values that follow the same pattern are valid, not typos.
            # The ML classifier handles those correctly via structural features.
            use_strict_typo_detection = (
                column_name in self.categorical_columns
                or column_name in self.reference_lists
            )

            if use_strict_typo_detection:
                # Fuzzy match to catch typos - if similar to a valid value but not exact, it's likely a typo
                for valid_value in whitelist:
                    similarity = difflib.SequenceMatcher(None, text_lower, valid_value).ratio()

                    # High similarity (>0.8) but not exact = likely typo = INVALID
                    if similarity >= 0.8:
                        return False, similarity  # Invalid - it's a typo of a known value

        # For categorical columns: value not in whitelist and not a known typo = unknown category
        if column_name in self.categorical_columns:
            return False, 0.85

        # For values not similar to whitelist, use ML model
        features = self.feature_extractor.extract_features(text_str, column_name)
        features_scaled = self.column_scalers[column_name].transform([features])
        prediction = self.column_models[column_name].predict(features_scaled)[0]
        probabilities = self.column_models[column_name].predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))

        return bool(prediction), confidence

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

        for idx, row in df.iterrows():
            row_results = self.validate_row(row, column_mapping)
            for col in df.columns:
                if col in row_results:
                    results[col].append(row_results[col])
                else:
                    results[col].append((True, 0.5))  # Unknown column

        return pd.DataFrame(results, index=df.index)

    def validate_batch(self, texts: List[str], column_name: str) -> List[Tuple[bool, float]]:
        """
        Validate multiple values at once for a specific column (faster).

        Args:
            texts: List of texts to validate
            column_name: Column name these values belong to

        Returns:
            List of (is_valid, confidence) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        if column_name not in self.column_models:
            return [(True, 0.5)] * len(texts)

        results = []
        texts_for_ml = []
        ml_indices = []

        # Get combined whitelist (training data + reference lists)
        whitelist = self._get_combined_whitelist(column_name)

        # Strict typo detection only for categorical columns or columns with reference lists.
        # High-cardinality columns (order_id, customer_name, etc.) should NOT use fuzzy
        # matching — new unseen values that follow the same pattern are valid, not typos.
        # The ML classifier handles those correctly via structural features.
        col_lower = column_name.lower()
        use_strict_typo_detection = (
            column_name in self.categorical_columns
            or column_name in self.reference_lists
        )

        # First pass: check whitelist with fuzzy matching
        for i, text in enumerate(texts):
            text_str = str(text).strip()
            text_lower = text_str.lower()

            # Exact match = valid
            if text_lower in whitelist:
                results.append((True, 1.0))
                continue

            # For numeric values, try normalized comparison (95 == 95.0)
            numeric_match = False
            try:
                numeric_val = float(text_str)
                if str(int(numeric_val)) in whitelist or str(numeric_val) in whitelist or f"{numeric_val:.1f}" in whitelist:
                    results.append((True, 1.0))
                    numeric_match = True
            except (ValueError, TypeError):
                pass

            if numeric_match:
                continue

            # Fuzzy match to catch typos (only for columns with finite valid values)
            found_typo = False
            if use_strict_typo_detection:
                for valid_value in whitelist:
                    similarity = difflib.SequenceMatcher(None, text_lower, valid_value).ratio()
                    if similarity >= 0.8:  # High similarity but not exact = typo
                        results.append((False, similarity))  # Invalid - typo
                        found_typo = True
                        break

            if not found_typo:
                if column_name in self.categorical_columns:
                    results.append((False, 0.85))  # Unknown category - whitelist-only validation
                else:
                    results.append(None)  # Placeholder for ML
                    texts_for_ml.append(text_str)
                    ml_indices.append(i)

        # Second pass: ML for entries not matching whitelist
        if texts_for_ml:
            X = [self.feature_extractor.extract_features(text, column_name) for text in texts_for_ml]
            X_scaled = self.column_scalers[column_name].transform(X)
            predictions = self.column_models[column_name].predict(X_scaled)
            probabilities = self.column_models[column_name].predict_proba(X_scaled)

            for idx, pred, prob in zip(ml_indices, predictions, probabilities):
                is_valid = bool(pred)
                confidence = float(max(prob))
                results[idx] = (is_valid, confidence)

        return results

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
        col_lower = column_name.lower() if column_name else ''

        # RULE 1: Never suggest corrections for numeric columns
        # Numbers don't have "typos" - a wrong number is just wrong
        for num_col in self.NUMERIC_COLUMNS:
            if num_col in col_lower:
                return None  # No correction for age, blood_sugar, etc.

        # RULE 2: Simple email typo fixes (double @@ -> single @)
        if 'email' in col_lower or 'mail' in col_lower:
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
        is_high_similarity_column = any(col in col_lower for col in self.HIGH_SIMILARITY_COLUMNS)

        # RULE 4: For phone numbers, also check digit count similarity
        if 'phone' in col_lower:
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
        elif 'country' in col_lower:
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
                if 'phone' in col_lower:
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

        # Check hardcoded rules for known base model columns
        if column_name:
            col_lower = column_name.lower()

            # Find matching rule set (prefer exact match, then word boundary match)
            rule_key = None
            # First try exact match
            if col_lower in self.VALIDATION_RULES:
                rule_key = col_lower
            else:
                # Try word boundary match (avoid 'age' matching 'percentage')
                import re
                for key in self.VALIDATION_RULES:
                    # Match as whole word
                    if re.search(r'\b' + re.escape(key) + r'\b', col_lower):
                        rule_key = key
                        break
                    # Or if column starts with key
                    if col_lower.startswith(key + '_') or col_lower.startswith(key):
                        if rule_key is None or len(key) > len(rule_key):
                            rule_key = key

            if rule_key:
                rules = self.VALIDATION_RULES[rule_key]

                # Numeric column validation (age, blood_sugar)
                if rules.get('type') == 'numeric':
                    try:
                        num_val = float(text_str)
                        if 'min' in rules and num_val < rules['min']:
                            if num_val < 0:
                                return f"{rule_key.replace('_', ' ').title()} cannot be negative"
                            return f"{rule_key.replace('_', ' ').title()} is below minimum ({rules['min']})"
                        if 'max' in rules and num_val > rules['max']:
                            return f"{rule_key.replace('_', ' ').title()} must be between {rules['min']} and {rules['max']}"
                    except (ValueError, TypeError):
                        return f"{rule_key.replace('_', ' ').title()} must be a valid number"

                # Phone validation
                elif rule_key == 'phone':
                    digit_count = sum(1 for c in text_str if c.isdigit())
                    if digit_count == 0:
                        return "Phone number must contain digits"
                    if digit_count < 7:
                        return f"Incomplete phone number ({digit_count} digits, minimum 7 required)"

                # Email validation
                elif rule_key == 'email':
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
                    if 'currency' in self.reference_lists:
                        return "Currency code not recognized (use: USD, EUR, GBP, SGD, etc.)"
                    return "Invalid currency code"

                # Status validation (check against reference list)
                elif rule_key == 'status':
                    if 'status' in self.reference_lists:
                        return "Status not recognized (use: Active, Inactive, Pending, etc.)"
                    return "Invalid status value"

                # Date validation
                elif rule_key == 'date':
                    return "Invalid date format (use: YYYY-MM-DD, DD/MM/YYYY, etc.)"

        # Additional checks for columns not in VALIDATION_RULES but with keywords
        if column_name:
            col_lower = column_name.lower()

            # Salary/Price/Amount validation
            if any(kw in col_lower for kw in ['salary', 'price', 'amount', 'income', 'cost', 'quantity']):
                try:
                    val = float(text_str.replace(',', '').replace('$', '').replace('£', '').replace('€', ''))
                    if val < 0:
                        return f"{col_lower.title()} cannot be negative"
                except (ValueError, TypeError):
                    return f"{col_lower.title()} must be a valid number"

            # Percentage validation
            if 'percent' in col_lower:
                try:
                    pct_val = float(text_str.replace('%', ''))
                    if pct_val < 0:
                        return "Percentage cannot be negative"
                    if pct_val > 100:
                        return "Percentage cannot exceed 100%"
                except (ValueError, TypeError):
                    return "Percentage must be a valid number"

        # Categorical column - value not in the valid set seen during training
        if column_name and column_name in self.categorical_columns:
            examples = self.column_examples.get(column_name, [])
            if examples:
                sample = ', '.join(sorted(str(v) for v in examples[:8]))
                suffix = f' (+{len(examples) - 8} more)' if len(examples) > 8 else ''
                return f"Not a valid {column_name} (expected: {sample}{suffix})"
            return f"Not a recognized value for {column_name}"

        # Fallback: Extract features for generic analysis
        features = self.feature_extractor.extract_features(text, column_name)

        # General pattern issues
        if len(features) > 46 and features[46] > 0:  # triple_chars
            issues.append("contains repeated characters")

        # Length checks for unknown columns
        if features[0] < 2:
            issues.append("value too short")
        elif features[0] > 200:
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
            'column_whitelists': self.column_whitelists,
            'column_examples': self.column_examples,
            'reference_lists': self.reference_lists,
            'training_metrics': self.training_metrics,
            'categorical_columns': self.categorical_columns,
            'is_trained': self.is_trained,
            'version': '2.2',  # Updated with categorical column detection
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Load unified model from file.

        Args:
            filepath: Path to saved model

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False

        try:
            model_data = joblib.load(filepath)

            # Check version
            version = model_data.get('version', '1.0')
            if version not in ['2.0', '2.1', '2.2']:
                print(f"Warning: Loading old model format (v{version}). Some features may not work.")

            self.model_name = model_data.get('model_name', 'unknown')
            self.columns = model_data.get('columns', [])
            self.column_models = model_data.get('column_models', {})
            self.column_scalers = model_data.get('column_scalers', {})
            self.column_whitelists = model_data.get('column_whitelists', {})
            self.column_examples = model_data.get('column_examples', {})
            self.reference_lists = model_data.get('reference_lists', {})
            self.training_metrics = model_data.get('training_metrics', {})
            self.categorical_columns = model_data.get('categorical_columns', set())
            self.is_trained = model_data.get('is_trained', False)

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
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

        print(f"Adding {len(df)} rows to existing model '{self.model_name}'...")

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

                print(f"  {col}: Added {len(new_values)} examples (total whitelist: {len(self.column_whitelists[col])})")
            else:
                print(f"  {col}: Skipped (not in trained columns)")

        # Optionally retrain ML models with combined data
        if retrain:
            print("\nRetraining ML models with combined data...")
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
            print(f"Reference list not found: {filepath}")
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

            print(f"Loaded {len(values)} values for '{column_name}' from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading reference list: {e}")
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
            print(f"Reference directory not found: {directory}")
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

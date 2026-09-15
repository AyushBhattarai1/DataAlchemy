"""Reusable, leak-free ML preprocessing subsystem for DataAlchemy.

Handles mixed tabular data, missing value imputation, categorical one-hot encoding,
datetime extraction, boolean conversion, feature scaling, and automated exclusion of
identifiers, constant columns, all-null features, and high-cardinality columns.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.ingestion.loader import Dataset
from src.ml.models import MLPreprocessingResult
from src.profiling.profiler import DatasetProfile


class MLPreprocessor:
    """Scikit-learn style preprocessor preparing tabular data for unsupervised ML algorithms."""

    def __init__(
        self,
        numeric_imputation: str = "median",
        scaling: Optional[str] = "standard",
        max_categories: int = 50,
        handle_unknown: str = "ignore",
    ) -> None:
        """Initialize preprocessor with configuration parameters.

        Args:
            numeric_imputation: Strategy for missing numeric values ('median', 'mean', 'zero').
            scaling: Feature scaling strategy ('standard', 'minmax', 'robust', or None).
            max_categories: Max unique categories for one-hot encoding before exclusion.
            handle_unknown: Strategy for unseen categories during transform ('ignore').
        """
        self.numeric_imputation = numeric_imputation
        self.scaling_strategy = scaling
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown

        # Fitted state
        self.is_fitted: bool = False
        self.original_columns_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.datetime_columns_: List[str] = []
        self.boolean_columns_: List[str] = []
        self.excluded_columns_: List[str] = []
        self.exclusion_reasons_: Dict[str, str] = {}
        self.generated_features_: List[str] = []
        self.feature_names_: List[str] = []

        # Transformers and parameters
        self.imputation_values_: Dict[str, float] = {}
        self.categorical_imputation_values_: Dict[str, str] = {}
        self.encoder_: Optional[OneHotEncoder] = None
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.encoded_categorical_names_: List[str] = []

    def _extract_dataframe(self, data: Union[Dataset, pd.DataFrame]) -> pd.DataFrame:
        """Extract a clean, unmutated copy of the pandas DataFrame."""
        if isinstance(data, Dataset):
            return data.dataframe.copy()
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        raise TypeError(f"Expected Dataset or pd.DataFrame, got {type(data).__name__}")

    def _is_identifier_column(
        self,
        col: str,
        series: pd.Series,
        n_rows: int,
        profile: Optional[DatasetProfile] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if a column is an identifier based on semantic rules, naming, or uniqueness."""
        if profile and col in profile.columns:
            st = str(profile.columns[col].semantic_type).upper()
            if "ID" in st or "IDENTIFIER" in st:
                return True, "detected_by_semantic_profile"

        # Column naming conventions
        name_lower = col.lower().strip()
        id_patterns = [
            r"^id$",
            r"_id$",
            r"^id_",
            r"^customer_?id",
            r"^user_?id",
            r"^transaction_?id",
            r"^uuid$",
            r"^guid$",
            r"^email$",
            r"_uuid$",
        ]
        for pat in id_patterns:
            if re.search(pat, name_lower):
                return True, f"name_matches_pattern_{pat}"

        # High uniqueness text/object column
        if not pd.api.types.is_numeric_dtype(series) and n_rows >= 5:
            non_null = series.dropna()
            if len(non_null) > 0 and (non_null.nunique() / len(non_null)) >= 0.99:
                return True, "high_uniqueness_text"

        return False, None

    def fit(
        self,
        data: Union[Dataset, pd.DataFrame],
        profile: Optional[DatasetProfile] = None,
    ) -> "MLPreprocessor":
        """Fit preprocessor parameters on dataset without data leakage.

        Args:
            data: Input Dataset or DataFrame.
            profile: Optional pre-computed DatasetProfile.

        Returns:
            self: The fitted MLPreprocessor.
        """
        df = self._extract_dataframe(data)
        n_rows = len(df)
        self.original_columns_ = list(df.columns)

        # Reset states
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.datetime_columns_ = []
        self.boolean_columns_ = []
        self.excluded_columns_ = []
        self.exclusion_reasons_ = {}
        self.generated_features_ = []
        self.imputation_values_ = {}
        self.categorical_imputation_values_ = {}

        # 1. Column analysis and exclusion filtering
        for col in self.original_columns_:
            series = df[col]

            # A. All-null columns
            if series.isna().all() or len(series.dropna()) == 0:
                self.excluded_columns_.append(col)
                self.exclusion_reasons_[col] = "all_null"
                continue

            # B. Identifier columns
            is_id, reason = self._is_identifier_column(col, series, n_rows, profile)
            if is_id:
                self.excluded_columns_.append(col)
                self.exclusion_reasons_[col] = f"identifier ({reason})"
                continue

            # C. Constant columns (zero variance)
            if n_rows > 1 and series.nunique(dropna=True) <= 1:
                self.excluded_columns_.append(col)
                self.exclusion_reasons_[col] = "constant_feature"
                continue

            # D. Datetime columns
            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_columns_.append(col)
                continue
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                # Check if it parses as datetime
                try:
                    sample = series.dropna().iloc[:10]
                    if len(sample) > 0:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().all():
                            self.datetime_columns_.append(col)
                            continue
                except Exception:
                    pass

            # E. Boolean columns
            if pd.api.types.is_bool_dtype(series):
                self.boolean_columns_.append(col)
                continue
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                unique_vals = set(series.dropna().astype(str).str.lower().unique())
                if unique_vals.issubset({"true", "false", "1", "0", "yes", "no"}):
                    self.boolean_columns_.append(col)
                    continue

            # F. Numerical columns
            if pd.api.types.is_numeric_dtype(series):
                self.numeric_columns_.append(col)
                continue

            # G. Categorical columns
            unique_count = series.nunique(dropna=True)
            if unique_count > self.max_categories and n_rows >= 10:
                self.excluded_columns_.append(col)
                self.exclusion_reasons_[col] = (
                    f"high_cardinality ({unique_count} unique > threshold {self.max_categories})"
                )
                continue

            self.categorical_columns_.append(col)

        # 2. Fit Numeric Imputation
        for col in self.numeric_columns_:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.dropna()
            if len(valid) == 0:
                imp_val = 0.0
            elif self.numeric_imputation == "mean":
                imp_val = float(valid.mean())
            elif self.numeric_imputation == "zero":
                imp_val = 0.0
            else:  # default 'median'
                imp_val = float(valid.median())
            self.imputation_values_[col] = imp_val

        # 3. Fit Categorical OneHotEncoder
        if self.categorical_columns_:
            cat_df = pd.DataFrame(index=df.index)
            for col in self.categorical_columns_:
                mode_val = "missing"
                modes = df[col].dropna().mode()
                if len(modes) > 0:
                    mode_val = str(modes.iloc[0])
                self.categorical_imputation_values_[col] = mode_val
                cat_df[col] = df[col].fillna(mode_val).astype(str)

            self.encoder_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=False,
            )
            self.encoder_.fit(cat_df)
            self.encoded_categorical_names_ = list(
                self.encoder_.get_feature_names_out(self.categorical_columns_)
            )

        # 4. Fit Datetime Feature Names
        for col in self.datetime_columns_:
            self.generated_features_.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
            ])

        # 5. Assemble Feature Names
        all_feature_names = (
            list(self.numeric_columns_)
            + list(self.boolean_columns_)
            + list(self.generated_features_)
            + list(self.encoded_categorical_names_)
        )
        self.feature_names_ = all_feature_names

        # 6. Fit Scaler (Transform a sample or fit during fit_transform)
        if self.scaling_strategy and len(self.feature_names_) > 0:
            if self.scaling_strategy == "minmax":
                self.scaler_ = MinMaxScaler()
            elif self.scaling_strategy == "robust":
                self.scaler_ = RobustScaler()
            else:
                self.scaler_ = StandardScaler()

            # Pre-transform unscaled matrix on fit data to fit the scaler
            unscaled = self._transform_unscaled(df)
            self.scaler_.fit(unscaled)

        self.is_fitted = True
        return self

    def _transform_unscaled(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into unscaled numerical feature matrix."""
        blocks: List[np.ndarray] = []

        # A. Numerical
        if self.numeric_columns_:
            num_data = np.zeros((len(df), len(self.numeric_columns_)), dtype=np.float64)
            for idx, col in enumerate(self.numeric_columns_):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").fillna(self.imputation_values_[col])
                    num_data[:, idx] = s.to_numpy(dtype=np.float64)
                else:
                    num_data[:, idx] = self.imputation_values_[col]
            blocks.append(num_data)

        # B. Boolean
        if self.boolean_columns_:
            bool_data = np.zeros((len(df), len(self.boolean_columns_)), dtype=np.float64)
            for idx, col in enumerate(self.boolean_columns_):
                if col in df.columns:
                    s = df[col].astype(str).str.lower()
                    bool_data[:, idx] = s.isin({"true", "1", "yes"}).astype(np.float64)
                else:
                    bool_data[:, idx] = 0.0
            blocks.append(bool_data)

        # C. Datetime
        if self.datetime_columns_:
            dt_cols_count = len(self.datetime_columns_) * 4
            dt_data = np.zeros((len(df), dt_cols_count), dtype=np.float64)
            col_offset = 0
            for col in self.datetime_columns_:
                if col in df.columns:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(df[col], errors="coerce")
                    dt_data[:, col_offset] = parsed.dt.year.fillna(2000).to_numpy(dtype=np.float64)
                    dt_data[:, col_offset + 1] = parsed.dt.month.fillna(1).to_numpy(dtype=np.float64)
                    dt_data[:, col_offset + 2] = parsed.dt.day.fillna(1).to_numpy(dtype=np.float64)
                    dt_data[:, col_offset + 3] = parsed.dt.dayofweek.fillna(0).to_numpy(dtype=np.float64)
                else:
                    dt_data[:, col_offset] = 2000.0
                    dt_data[:, col_offset + 1] = 1.0
                    dt_data[:, col_offset + 2] = 1.0
                    dt_data[:, col_offset + 3] = 0.0
                col_offset += 4
            blocks.append(dt_data)

        # D. Categorical
        if self.categorical_columns_ and self.encoder_ is not None:
            cat_df = pd.DataFrame(index=df.index)
            for col in self.categorical_columns_:
                fill_val = self.categorical_imputation_values_.get(col, "missing")
                if col in df.columns:
                    cat_df[col] = df[col].fillna(fill_val).astype(str)
                else:
                    cat_df[col] = fill_val
            encoded = self.encoder_.transform(cat_df)
            blocks.append(encoded)

        if not blocks:
            return np.zeros((len(df), 0), dtype=np.float64)

        return np.hstack(blocks)

    def transform(self, data: Union[Dataset, pd.DataFrame]) -> np.ndarray:
        """Transform input data using fitted parameters.

        Args:
            data: Dataset or DataFrame to transform.

        Returns:
            np.ndarray: 2D numpy array of preprocessed, scaled features.
        """
        if not self.is_fitted:
            raise RuntimeError("MLPreprocessor must be fitted before calling transform().")

        df = self._extract_dataframe(data)
        unscaled = self._transform_unscaled(df)

        if self.scaler_ is not None and unscaled.shape[1] > 0:
            return self.scaler_.transform(unscaled)

        return unscaled

    def fit_transform(
        self,
        data: Union[Dataset, pd.DataFrame],
        profile: Optional[DatasetProfile] = None,
    ) -> np.ndarray:
        """Fit preprocessor and return transformed feature matrix in one call."""
        return self.fit(data, profile=profile).transform(data)

    def get_preprocessing_result(
        self,
        data: Union[Dataset, pd.DataFrame],
    ) -> MLPreprocessingResult:
        """Produce structured MLPreprocessingResult metadata describing the pipeline.

        Args:
            data: Input dataset evaluated.

        Returns:
            MLPreprocessingResult: Metadata and pipeline documentation.
        """
        df = self._extract_dataframe(data)
        return MLPreprocessingResult(
            feature_names=list(self.feature_names_),
            original_columns=list(self.original_columns_),
            numeric_columns=list(self.numeric_columns_),
            categorical_columns=list(self.categorical_columns_),
            datetime_columns=list(self.datetime_columns_),
            boolean_columns=list(self.boolean_columns_),
            excluded_columns=list(self.excluded_columns_),
            exclusion_reasons=dict(self.exclusion_reasons_),
            generated_features=list(self.generated_features_),
            imputation_metadata={
                "numeric_strategy": self.numeric_imputation,
                "values": self.imputation_values_,
            },
            encoding_metadata={
                "strategy": "one_hot",
                "handle_unknown": self.handle_unknown,
                "encoded_features_count": len(self.encoded_categorical_names_),
            },
            scaling_metadata={
                "strategy": self.scaling_strategy,
                "scaler_type": type(self.scaler_).__name__ if self.scaler_ else "None",
            },
            n_rows=len(df),
            n_features=len(self.feature_names_),
            is_sparse=False,
        )

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz
from typing import List, Set

class RerankerFeatureExtractor:
    """
    Orchestrates the extraction of relevance features from search query and document pairs.

    This class handles text normalization and the computation of various similarity metrics
    including Qdrant scores, fuzzy string matching, header analysis, and domain-specific
    heuristics (e.g., price relevance) to prepare data for the XGBoost reranker.

    Attributes:
        REQUIRED_COLUMNS (List[str]): A list of column names required in the input DataFrame.

    Methods:
        transform(df): Transforms raw input data into a feature set suitable for reranking.
    """

    REQUIRED_COLUMNS: List[str] = ['query_text', 'full_text', 'h1', 'qdrant_score']

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that the input DataFrame contains all necessary columns and is not empty.

        Args:
            df (pd.DataFrame): The input dataframe to validate.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
            TypeError: If the input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}.")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input DataFrame: {missing_cols}")

    @staticmethod
    def _calculate_word_overlap(row: pd.Series) -> float:
        """
        Calculates the intersection over union ratio of tokens between query and document.

        Args:
            row (pd.Series): A row containing 'q_lower' and 'doc_lower' text data.

        Returns:
            float: The overlap ratio (0.0 to 1.0).
        """
        try:
            q_text = str(row.get('q_lower', ''))
            d_text = str(row.get('doc_lower', ''))
            
            q_tokens: Set[str] = set(q_text.split())
            d_tokens: Set[str] = set(d_text.split())

            if not q_tokens:
                return 0.0

            return len(q_tokens.intersection(d_tokens)) / len(q_tokens)
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_price_relevance(row: pd.Series) -> int:
        """
        Determines if a query matches price-related information in the document.

        Args:
            row (pd.Series): A row containing 'q_lower' and 'doc_lower' text data.

        Returns:
            int: 1 if price relevance is established, 0 otherwise.
        """
        try:
            q_text = str(row.get('q_lower', ''))
            d_text = str(row.get('doc_lower', ''))

            price_keywords = {'harga', 'biaya', 'price', 'rp'}
            
            is_price_query = any(w in q_text for w in price_keywords)
            has_price_info = 'rp' in d_text or 'rp.' in d_text

            return 1 if (is_price_query and has_price_info) else 0
        except Exception:
            return 0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw input data into a feature set suitable for reranking models.

        Performs text normalization followed by the extraction of six specific features:
        Vector Score, Lengths, Word Overlap, Header Match, Fuzzy Match, and Price Heuristic.

        Args:
            df (pd.DataFrame): The input dataframe containing raw search data.
                               Must contain 'query_text', 'full_text', 'h1', and 'qdrant_score'.

        Returns:
            pd.DataFrame: A new DataFrame containing only the engineered features.

        Raises:
            ValueError: If validation fails due to missing columns or empty input.
            RuntimeError: If an unexpected error occurs during feature generation.
        """
        logger.info("Starting feature extraction process.")

        try:
            self._validate_schema(df)
            logger.debug(f"Input schema validated. Processing {len(df)} rows.")

            # Initialize output DataFrame
            features = pd.DataFrame()

            # Normalize text (Handle NaNs by filling with empty string)
            # Using copy to avoid SettingWithCopyWarning on the input dataframe
            work_df = df.copy()
            work_df['q_lower'] = work_df['query_text'].fillna("").astype(str).str.lower()
            work_df['doc_lower'] = work_df['full_text'].fillna("").astype(str).str.lower()
            work_df['h1_lower'] = work_df['h1'].fillna("").astype(str).str.lower()

            # 1. Vector Score
            features['qdrant_score'] = work_df['qdrant_score'].astype(float)

            # 2. Lengths
            features['doc_len'] = work_df['doc_lower'].apply(len)
            features['query_len'] = work_df['q_lower'].apply(len)

            # 3. Word Overlap
            features['word_overlap'] = work_df.apply(self._calculate_word_overlap, axis=1)

            # 4. Header Match
            features['match_in_h1'] = work_df.apply(
                lambda x: fuzz.partial_ratio(x['q_lower'], x['h1_lower']), axis=1
            )

            # 5. Fuzzy Match (Truncated to first 500 chars for performance)
            features['fuzzy_ratio'] = work_df.apply(
                lambda x: fuzz.ratio(x['q_lower'], x['doc_lower'][:500]), axis=1
            )

            # 6. Price Heuristic
            features['is_price_match'] = work_df.apply(self._calculate_price_relevance, axis=1)

            logger.info("Feature extraction completed successfully.")
            return features

        except ValueError as ve:
            logger.error(f"Validation error in feature extraction: {ve}")
            raise ve
        except Exception as e:
            logger.exception("Unexpected error during feature extraction transformation.")
            raise RuntimeError(f"Feature extraction failed: {e}") from e
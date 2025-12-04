import pandas as pd
from rapidfuzz import fuzz

class RerankerFeatureExtractor:
    """
    Extracts relevance features from query-document pairs for reranking tasks.

    This class processes a DataFrame containing query and document text data to
    generate numerical features representing similarity, word overlap, and
    specific domain heuristics (e.g., price relevance). It is designed to be
    stateless.

    Attributes:
        None

    Methods:
        transform(df: pd.DataFrame) -> pd.DataFrame:
            Generates feature vectors from the input DataFrame.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw query and document text data into a DataFrame of numerical features.

        This method performs several steps: normalization of text data, extraction
        of vector scores, calculation of length features, computation of word overlap
        ratios, and execution of fuzzy string matching and heuristic checks. It
        ensures that missing values in text columns are treated as empty strings to
        prevent runtime errors during string manipulation.

        Args:
            df (pd.DataFrame): The input DataFrame containing the raw data.
                Must contain the following columns:
                - 'query_text' (str): The search query.
                - 'full_text' (str): The main content of the document.
                - 'h1' (str): The header or title of the document.
                - 'qdrant_score' (float): Pre-computed vector similarity score.

        Returns:
            pd.DataFrame: A DataFrame containing the generated features.
                The columns will include:
                - 'qdrant_score' (float)
                - 'doc_len' (int)
                - 'query_len' (int)
                - 'word_overlap' (float)
                - 'match_in_h1' (float)
                - 'fuzzy_ratio' (float)
                - 'is_price_match' (int)

        Raises:
            TypeError: If the input `df` is not a pandas DataFrame.
            ValueError: If the input `df` is missing required columns.
        """
        # 1. Input Validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        required_columns = {'query_text', 'full_text', 'h1', 'qdrant_score'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        features = pd.DataFrame(index=df.index)

        # Handle empty DataFrame case early
        if df.empty:
            return pd.DataFrame(columns=[
                'qdrant_score', 'doc_len', 'query_len', 'word_overlap',
                'match_in_h1', 'fuzzy_ratio', 'is_price_match'
            ])

        # 2. Data Preprocessing (Robustness)
        # Create copies to avoid SettingWithCopy warnings and ensure isolation
        # Fill NA with empty strings to prevent 'nan' string matching or TypeErrors
        try:
            # We use a temporary internal dataframe for normalized data to keep the logic clean
            temp_df = pd.DataFrame(index=df.index)
            temp_df['q_lower'] = df['query_text'].fillna('').astype(str).str.lower().str.strip()
            temp_df['doc_lower'] = df['full_text'].fillna('').astype(str).str.lower().str.strip()
            temp_df['h1_lower'] = df['h1'].fillna('').astype(str).str.lower().str.strip()
        except Exception as e:
            # Fallback for catastrophic data corruption during normalization
            raise ValueError(f"Error during text normalization: {str(e)}") from e

        # 3. Feature Extraction

        # Feature: Vector Score (Ensure numeric)
        features['qdrant_score'] = pd.to_numeric(df['qdrant_score'], errors='coerce').fillna(0.0)

        # Feature: Lengths
        features['doc_len'] = temp_df['doc_lower'].apply(len)
        features['query_len'] = temp_df['q_lower'].apply(len)

        # Feature: Word Overlap
        def _calculate_word_overlap(row: pd.Series) -> float:
            q_tokens = set(row['q_lower'].split())
            d_tokens = set(row['doc_lower'].split())
            
            if not q_tokens:
                return 0.0
            
            intersection = q_tokens.intersection(d_tokens)
            return len(intersection) / len(q_tokens)

        features['word_overlap'] = temp_df.apply(_calculate_word_overlap, axis=1)

        # Feature: Header Match (Partial Ratio)
        features['match_in_h1'] = temp_df.apply(
            lambda x: fuzz.partial_ratio(x['q_lower'], x['h1_lower']), axis=1
        )

        # Feature: Fuzzy Match (Ratio on truncated document)
        features['fuzzy_ratio'] = temp_df.apply(
            lambda x: fuzz.ratio(x['q_lower'], x['doc_lower'][:500]), axis=1
        )

        # Feature: Price Heuristic
        def _calculate_price_relevance(row: pd.Series) -> int:
            price_keywords = {'harga', 'biaya', 'price', 'rp'}
            
            # Check if query contains price intent
            is_price_query = any(w in row['q_lower'] for w in price_keywords)
            
            # Check if doc contains price indicators (specific currency markers)
            # Using exact substring check for 'rp' markers
            has_price_info = 'rp' in row['doc_lower'] or 'rp.' in row['doc_lower']
            
            return 1 if (is_price_query and has_price_info) else 0

        features['is_price_match'] = temp_df.apply(_calculate_price_relevance, axis=1)

        return features
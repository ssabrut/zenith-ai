import os
from dotenv import load_dotenv
load_dotenv()

import mlflow
import pandas as pd
from fastmcp import FastMCP
from loguru import logger
from qdrant_client import QdrantClient
from rapidfuzz import fuzz
from typing import Union, List, Dict, Set
from langchain_community.embeddings import DeepInfraEmbeddings

mcp = FastMCP("Qdrant-tools-server")

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN")

mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_HOST)
deepinfra_embedding = DeepInfraEmbeddings(
    model_id="Qwen/Qwen3-Embedding-8B",
    query_instruction="",
    embed_instruction="",
    deepinfra_api_token=DEEPINFRA_API_TOKEN
)

@mcp.tool()
async def search_knowledge_base(query: str) -> Union[List[Dict], str]:
    print(f"🔎 MCP Search Query: {query}")

    query_vector = deepinfra_embedding.embed_query(query)

    hits = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        limit=20,
        query=query_vector,
    )

    if not hits.points:
        return "No information found in the knowledge base"

    reranker = get_reranker()

    data_candidates = []
    for point in hits.points:
        payload = point.payload
        data_candidates.append({
            "query_text": query,
            "doc_id": point.id,
            "full_text": payload.get('full_text', ''),
            "h1": payload.get('h1', ''),
            "qdrant_score": point.score,
            "payload": payload
        })

    df_candidates = pd.DataFrame(data_candidates)
    if reranker:
        extractor = RerankerFeatureExtractor()
        X = extractor.transform(df_candidates)
        df_candidates["score"] = reranker.predict(X)
        df_candidates = df_candidates.sort_values(by='score', ascending=False)

    top_results = df_candidates.head(5)
    top_results = top_results.to_dict(orient="records")
    return top_results

def get_reranker():
    global reranker_model
    if not reranker_model:
        print("Loading XGBoost Reranker from MLflow")
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            versions = mlflow_client.get_latest_versions("XGBoostReranker", stages=["Staging"])
            latest_version = versions[0].version
            model_uri = f"models:/XGBoostReranker/{latest_version}"
            reranker_model = mlflow.xgboost.load_model(model_uri)
        except Exception as e:
            print(f"⚠️ Could not load Reranker: {e}. Falling back to raw vector search.")
        return reranker_model

class RerankerFeatureExtractor:
    """
    Orchestrates the extraction of relevance features from search query and document pairs.
    
    This class handles text normalization and the computation of various similarity metrics
    including Qdrant scores, fuzzy string matching, header analysis, and domain-specific
    heuristics (e.g., price relevance).

    Attributes:
        REQUIRED_COLUMNS (list[str]): A list of column names required in the input DataFrame.
    """
    
    REQUIRED_COLUMNS = ['query_text', 'full_text', 'h1', 'qdrant_score']

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
        q_tokens: Set[str] = set(row['q_lower'].split())
        d_tokens: Set[str] = set(row['doc_lower'].split())
        
        if not q_tokens:
            return 0.0
            
        return len(q_tokens.intersection(d_tokens)) / len(q_tokens)

    @staticmethod
    def _calculate_price_relevance(row: pd.Series) -> int:
        """
        Determines if a query matches price-related information in the document.

        Args:
            row (pd.Series): A row containing 'q_lower' and 'doc_lower' text data.

        Returns:
            int: 1 if price relevance is established, 0 otherwise.
        """
        price_keywords = {'harga', 'biaya', 'price', 'rp'}
        # Check if any price keyword exists in the query
        is_price_query = any(w in row['q_lower'] for w in price_keywords)
        
        # Check if price currency indicators exist in the document
        has_price_info = 'rp' in row['doc_lower'] or 'rp.' in row['doc_lower']
        
        return 1 if (is_price_query and has_price_info) else 0

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
            Exception: If an unexpected error occurs during feature generation.
        """
        logger.info("Starting feature extraction process.")
        
        try:
            self._validate_schema(df)
            logger.debug(f"Input schema validated. Processing {len(df)} rows.")

            # Initialize output DataFrame
            features = pd.DataFrame()

            # Normalize text (Handle NaNs by filling with empty string to prevent crashes)
            # We use a temporary working dataframe to avoid modifying the input in place unexpectedly
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
            raise e

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
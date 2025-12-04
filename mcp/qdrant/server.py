import os
import sys
from typing import Union, List, Dict, Set, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from loguru import logger
import mlflow
import pandas as pd
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from rapidfuzz import fuzz
from langchain_community.embeddings import DeepInfraEmbeddings

# Load environment variables
load_dotenv()

# --- Configuration & Validation ---
REQUIRED_ENV_VARS = [
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION",
    "MLFLOW_TRACKING_URI",
    "DEEPINFRA_API_TOKEN"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# --- Constants ---
IS_DOCKER = os.getenv("IS_DOCKER") == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST") if IS_DOCKER else "localhost"
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") if IS_DOCKER else "http://localhost:5050"
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN")

if IS_DOCKER:
    MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
else:
    MLFLOW_S3_ENDPOINT_URL = "http://localhost:9001"

logger.info(f"Docker environment: {IS_DOCKER}")
logger.info(f"Docker environment: {type(IS_DOCKER)}")
logger.info(f"Using '{QDRANT_HOST}' as Qdrant host")
logger.info(f"Tracking MLflow at {MLFLOW_TRACKING_URI}")
logger.info(f"Using S3 at {MLFLOW_S3_ENDPOINT_URL}")

try:
    mlflow_client = mlflow.tracking.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI, 
        registry_uri=MLFLOW_S3_ENDPOINT_URL
    )
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    deepinfra_embedding = DeepInfraEmbeddings(
        model_id="Qwen/Qwen3-Embedding-8B",
        query_instruction="",
        embed_instruction="",
        deepinfra_api_token=DEEPINFRA_API_TOKEN
    )
    logger.info("External clients initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize external clients: {e}")
    sys.exit(1)

reranker: Optional[Any] = None
def load_reranker():
    """Loads the model into the global variable."""
    global reranker
    logger.info("⏳ Loading XGBoost Reranker from MLflow...")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        versions = mlflow_client.get_latest_versions("XGBoostReranker", stages=["Staging"])
        
        if not versions:
            logger.warning("⚠️ No Staging version found for 'XGBoostReranker'.")
            return
            
        latest_version = versions[0].version
        model_uri = f"models:/XGBoostReranker/{latest_version}"
        
        # Load model
        reranker = mlflow.xgboost.load_model(model_uri)
        logger.success(f"✅ Successfully loaded Reranker version {latest_version}.")
    except Exception as e:
        logger.error(f"❌ Could not load Reranker: {e}. Falling back to raw vector search.")

@asynccontextmanager
async def lifespan(server: FastMCP):
    load_reranker()
    yield

# --- Global Client Initialization ---
# We initialize these globally, but actual connection errors will be caught during usage
# to allow the server to start even if downstream services are temporarily blipping.
mcp = FastMCP("Qdrant-tools-server", lifespan=lifespan)

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


@mcp.tool()
async def search_knowledge_base(query: str) -> Union[List[Dict[str, Any]], str]:
    """
    Searches the internal knowledge base for relevant documents based on a user's query.

    This tool performs a semantic vector search using Qdrant and re-ranks the results
    using an XGBoost model to provide the most relevant answers. Use this tool whenever
    the user asks for information, prices, treatments, or specific details contained
    in the documentation.

    Args:
        query (str): The search query or question from the user.

    Returns:
        Union[List[Dict[str, Any]], str]: A list of the top 5 most relevant document records,
                                          or a message indicating no information was found.

    Raises:
        None: Errors are handled internally to return a user-friendly failure message.
    """
    if not query or not isinstance(query, str) or not query.strip():
        logger.warning("Received empty or invalid query.")
        return "Please provide a valid search query."

    logger.info(f"🔎 MCP Search Query: {query}")

    try:
        # 1. Embed Query
        query_vector = deepinfra_embedding.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return "Service temporarily unavailable (Embedding Error)."

    try:
        # 2. Vector Search (Qdrant)
        hits = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            limit=40,
            query=query_vector,
        )
    except Exception as e:
        logger.error(f"Failed to query Qdrant: {e}")
        return "Service temporarily unavailable (Database Error)."

    if not hits.points:
        logger.info("No results found in Qdrant.")
        return "No information found in the knowledge base."

    # 3. Process Candidates
    data_candidates: List[Dict[str, Any]] = []
    for point in hits.points:
        payload = point.payload or {}
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
        try:
            extractor = RerankerFeatureExtractor()
            X = extractor.transform(df_candidates)
            df_candidates["score"] = reranker.predict(X)
            df_candidates = df_candidates.sort_values(by='score', ascending=False)
            logger.info("Reranking completed successfully.")
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning raw vector results.")
            # Fallback is implicitly handled by not sorting if this block fails
            # We assume df_candidates retains the original Qdrant order or we resort by qdrant_score
            df_candidates = df_candidates.sort_values(by='qdrant_score', ascending=False)
    else:
        logger.info("Reranker not available. Returning raw vector results.")
        # Ensure fallback sort
        df_candidates = df_candidates.sort_values(by='qdrant_score', ascending=False)

    # 5. Format Output
    top_results = df_candidates.head(5).to_dict(orient="records")
    return top_results


if __name__ == "__main__":
    logger.info("Starting MCP Server...")
    try:
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
    except Exception as e:
        logger.critical(f"MCP Server crashed: {e}")
        sys.exit(1)
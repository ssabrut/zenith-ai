import mlflow
import pandas as pd
from langchain.tools import tool
from typing import Union, List, Dict, Any, Optional
from loguru import logger

from core.services.deepinfra.factory import make_deepinfra_client
from core.services.qdrant.factory import make_qdrant_client
from core.config import get_settings
from core.utils import RerankerFeatureExtractor
from core.services.mlflow.factory import make_mlflow_service

deepinfra_embedding = make_deepinfra_client("Qwen/Qwen3-Embedding-8B").model
qdrant_client = make_qdrant_client().client
mlflow_client = make_mlflow_service().client
settings = get_settings()

reranker: Optional[Any] = None
def load_reranker():
    """Loads the model into the global variable."""
    global reranker
    logger.info("⏳ Loading XGBoost Reranker from MLflow...")
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
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

reranker = load_reranker()

@tool
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
            collection_name=settings.QDRANT_COLLECTION,
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
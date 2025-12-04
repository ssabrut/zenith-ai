import os
from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.xgboost
import pandas as pd
from typing import List, Dict, Union
from rapidfuzz import fuzz
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from langchain_community.embeddings import DeepInfraEmbeddings

mcp = FastMCP("Qdrant-MCP-Service")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "zenith_collection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN", "")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
deepinfra_embedding = DeepInfraEmbeddings(
    model_id="Qwen/Qwen3-Embedding-8B",
    embed_instruction="",
    query_instruction="",
    deepinfra_api_token=DEEPINFRA_API_TOKEN
)

reranker_model = None

def get_reranker():
    global reranker_model
    if not reranker_model:
        print("Loading XGBoost Reranker from MLflow")
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI)

            versions = client.get_latest_versions("XGBoostReranker", stages=["Staging"])
            latest_version = versions[0].version
            model_uri = f"models:/XGBoostReranker/{latest_version}"
            reranker_model = mlflow.xgboost.load_model(model_uri)
        except Exception as e:
            print(f"⚠️ Could not load Reranker: {e}. Falling back to raw vector search.")
        return reranker_model

class RerankerFeatureExtractor:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        
        # Normalize text
        df['q_lower'] = df['query_text'].astype(str).str.lower()
        df['doc_lower'] = df['full_text'].astype(str).str.lower()
        df['h1_lower'] = df['h1'].astype(str).str.lower()
        
        # 1. Vector Score
        features['qdrant_score'] = df['qdrant_score']
        
        # 2. Lengths
        features['doc_len'] = df['doc_lower'].apply(len)
        features['query_len'] = df['q_lower'].apply(len)
        
        # 3. Word Overlap
        def word_overlap(row):
            q_tokens = set(row['q_lower'].split())
            d_tokens = set(row['doc_lower'].split())
            if not q_tokens: return 0.0
            return len(q_tokens.intersection(d_tokens)) / len(q_tokens)
        features['word_overlap'] = df.apply(word_overlap, axis=1)
        
        # 4. Header Match
        features['match_in_h1'] = df.apply(
            lambda x: fuzz.partial_ratio(x['q_lower'], x['h1_lower']), axis=1
        )
        
        # 5. Fuzzy Match
        features['fuzzy_ratio'] = df.apply(
            lambda x: fuzz.ratio(x['q_lower'], x['doc_lower'][:500]), axis=1
        )
        
        # 6. Price Heuristic
        def price_relevance(row):
            is_price_query = any(w in row['q_lower'] for w in ['harga', 'biaya', 'price', 'rp'])
            has_price_info = 'rp' in row['doc_lower'] or 'rp.' in row['doc_lower']
            return 1 if (is_price_query and has_price_info) else 0
        features['is_price_match'] = df.apply(price_relevance, axis=1)
        
        return features

@mcp.tool
async def search_knowledge_base(query: str) -> Union[List[Dict], str]:
    print(f"🔎 MCP Search Query: {query}")

    query_vector = deepinfra_embedding.embed_query(query)

    hits = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=20
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
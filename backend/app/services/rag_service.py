from typing import List, Dict, Any
# from milvus import MilvusClient
from .bitnet_service import BitNetService
from .embedding_service import EmbeddingService

class RAGService:
    def __init__(self, 
                 milvus_uri: str = "http://localhost:19530",
                 collection_name: str = "documents"):
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.bitnet = BitNetService()
        self.embedding_service = EmbeddingService()

    def search_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant context in Milvus"""
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Search in Milvus
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text"]
        )
        
        return [{"text": hit["text"]} for hit in results[0]]

    def query(self, query: str) -> str:
        """Process a query using RAG"""
        # Get relevant context
        context = self.search_relevant_context(query)
        
        # Generate response using BitNet
        response = self.bitnet.chat_with_context(query, context)
        
        return response 
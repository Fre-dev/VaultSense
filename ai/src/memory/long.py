from typing import Dict, List, Optional, Any
import uuid
import json
import os
from datetime import datetime
from src.memory.vectordb import VectorStore
from src.llm.embed import embed_text
from src.llm.runner import get_llm_model, call_llm, call_llm_sync

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

# Milvus Configuration
MEMORY_COLLECTION_NAME = "ltm"
VECTOR_DIM = 768  # Dimension for text embeddings

class LongTermMemory:
    """
    Long-term memory storage using Milvus vector database.
    
    This class handles storage and retrieval of memories from the vector database,
    including adding new memories, searching for similar memories, and updating
    existing memories.
    """
    
    def __init__(self, customer: str = "default"):
        """
        Initialize the long-term memory for a specific customer.
        
        Args:
            customer: Customer identifier to separate memory spaces
        """
        self.customer = customer
        self.milvus_client = VectorStore.get_vector_store_connection(self.customer)
        self.model = get_llm_model()
        self._init_milvus()
    
    def _init_milvus(self) -> bool:
        """Initialize connection to Milvus and create collection if needed"""
        try:            
            logger.info(f"Initialized Milvus connection for customer {self.customer}")
            return True
        except Exception as e:
            logger.error(f"Error initializing Milvus: {str(e)}")
            return False
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get text embedding using LLM service"""
        try:
            embedding = embed_text(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * VECTOR_DIM
    
    def save_question_answer(
        self, 
        question: str, 
        answer: str, 
        response_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a question-answer pair to long-term memory.
        
        Args:
            question: The user's question
            answer: The system's answer
            response_id: Unique identifier for the response (generated if not provided)
            user_id: Identifier for the user who asked the question
            metadata: Additional metadata to store with the memory
            
        Returns:
            The ID of the saved memory
        """
        try:
            milvus_client = VectorStore.get_vector_store_connection(self.customer)
            if not response_id:
                response_id = str(uuid.uuid4())
            
            # Get embedding for the question
            embedding = self._get_embedding(question)
            
            # Prepare metadata
            metadata_str = "{}"
            if metadata:
                metadata_str = json.dumps(metadata)
            
            # Insert data using the dictionary format expected by the MilvusClient
            milvus_client.insert(
                collection_name=MEMORY_COLLECTION_NAME,
                data={
                    "response_id": response_id,
                    "question": question,
                    "answer": answer,
                    "user_id": user_id or "unknown",
                    "created_at": datetime.now().isoformat(),
                    "metadata": metadata_str,
                    "upvotes": 0,
                    "downvotes": 0,
                    "embedding": embedding
                }
            )
            
            logger.info(f"Saved memory for response ID: {response_id}")
            return response_id
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            return ""
    
    def get_similar_questions(
        self, 
        question: str, 
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar questions in long-term memory.
        
        Args:
            question: The question to search for similar memories
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar memories with their similarity scores
        """
        try:
            milvus_client = VectorStore.get_vector_store_connection(self.customer)
            collection_name = f"{MEMORY_COLLECTION_NAME}_{self.customer}"
            
            # Get embedding for the query
            query_embedding = self._get_embedding(question)
            
            # Perform search using the updated API
            results = milvus_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                output_fields=["question", "answer", "response_id", "created_at", "metadata", "upvotes", "downvotes"]
            )
            
            # Process results
            similar_questions = []
            if results:
                for hit in results:
                    score = hit.get("score", 0.0)
                    if score < min_score:
                        continue
                        
                    # Extract metadata
                    metadata = {}
                    try:
                        metadata_str = hit.get("metadata", "{}")
                        metadata = json.loads(metadata_str)
                    except:
                        logger.error(f"Failed to parse metadata JSON: {hit.get('metadata')}")
                    
                    # Format the result
                    similar_question = {
                        "question": hit.get("question", ""),
                        "answer": hit.get("answer", ""),
                        "response_id": hit.get("response_id", ""),
                        "created_at": hit.get("created_at", ""),
                        "metadata": metadata,
                        "upvotes": hit.get("upvotes", 0),
                        "downvotes": hit.get("downvotes", 0),
                        "similarity": score
                    }
                    similar_questions.append(similar_question)
            
            return similar_questions
        except Exception as e:
            logger.error(f"Error searching for similar questions: {str(e)}")
            return []
    
    def update_votes(self, response_id: str, upvote: bool) -> bool:
        """
        Update the upvotes or downvotes for a memory.
        
        Args:
            response_id: ID of the response to update
            upvote: True for upvote, False for downvote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            milvus_client = VectorStore.get_vector_store_connection(self.customer)
            collection_name = f"{MEMORY_COLLECTION_NAME}_{self.customer}"
            
            # Find the memory by response_id
            results = milvus_client.query(
                collection_name=collection_name,
                filter=f"response_id == '{response_id}'",
                output_fields=["id", "upvotes", "downvotes"],
                limit=1
            )
            
            if not results:
                logger.warning(f"No memory found with response_id {response_id}")
                return False
            
            # Get the memory ID and current votes
            memory = results[0]
            memory_id = memory.get("id")
            upvotes = memory.get("upvotes", 0)
            downvotes = memory.get("downvotes", 0)
            
            # Update the votes
            if upvote:
                upvotes += 1
            else:
                downvotes += 1
            
            # Update the memory
            milvus_client.update(
                collection_name=collection_name,
                filter=f"id == '{memory_id}'",
                data={
                    "upvotes": upvotes, 
                    "downvotes": downvotes
                }
            )
            
            logger.info(f"Updated votes for memory with response_id {response_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating votes: {str(e)}")
            return False
    
    def clear_memories(self) -> bool:
        """
        Clear all memories for the current customer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            milvus_client = VectorStore.get_vector_store_connection(self.customer)
            collection_name = f"{MEMORY_COLLECTION_NAME}_{self.customer}"
            
            # Check if collection exists then drop it
            if milvus_client.has_collection(collection_name):
                milvus_client.drop_collection(collection_name)
                logger.info(f"Cleared all memories for customer {self.customer}")
                return True
            
            logger.info(f"No memories collection found for customer {self.customer}")
            return False
        except Exception as e:
            logger.error(f"Error clearing memories: {str(e)}")
            return False 
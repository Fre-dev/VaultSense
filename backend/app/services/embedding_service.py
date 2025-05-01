from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Dict, Any, List
import json
import os

class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the embedding generator with a specific model"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text input"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings.squeeze().cpu().numpy()
    
    def process_ocr_result(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OCR result and generate embeddings for the extracted text
        
        Args:
            ocr_result: Dictionary containing the OCR output
            
        Returns:
            Dictionary containing the original OCR result and embeddings
        """
        try:
            # Convert the OCR result to a string for embedding
            ocr_text = json.dumps(ocr_result, ensure_ascii=False)
            
            # Generate embedding
            embedding = self.get_embedding(ocr_text)
            
            # Return combined result
            return {
                "ocr_result": ocr_result,
                "embedding": embedding.tolist()
            }
        except Exception as e:
            return {
                "error": f"Error generating embedding: {str(e)}",
                "ocr_result": ocr_result
            } 
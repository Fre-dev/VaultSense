import json
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """Service for generating embeddings using BGE model"""
    
    def __init__(self):
        """Initialize the embedding model"""
        self.model = SentenceTransformer('BAAI/bge-small-en')
        self.embeddings_dir = Path("embeddings")
        self.embeddings_dir.mkdir(exist_ok=True)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            numpy array containing the embedding
        """
        return self.model.encode(text)
    
    def process_json_file(self, json_path: Path) -> None:
        """
        Process a single JSON file and save its embedding
        
        Args:
            json_path: Path to the JSON file
        """
        # Read JSON content
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        
        # Convert JSON to string
        json_str = json.dumps(json_content, sort_keys=True)
        
        # Generate embedding
        embedding = self.generate_embedding(json_str)
        
        # Save embedding
        output_path = self.embeddings_dir / f"{json_path.stem}_embedding.npy"
        np.save(output_path, embedding)
        print(f"Saved embedding to {output_path}")
    
    def process_json_folder(self, json_folder: Union[str, Path]) -> None:
        """
        Process all JSON files in a folder
        
        Args:
            json_folder: Path to folder containing JSON files
        """
        json_path = Path(json_folder)
        
        for json_file in json_path.glob("*.json"):
            print(f"Processing {json_file.name}...")
            self.process_json_file(json_file)
    
    def get_embedding(self, json_path: Path) -> np.ndarray:
        """
        Get embedding for a JSON file if it exists, otherwise generate it
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            numpy array containing the embedding
        """
        embedding_path = self.embeddings_dir / f"{json_path.stem}_embedding.npy"
        
        if embedding_path.exists():
            return np.load(embedding_path)
        else:
            self.process_json_file(json_path)
            return np.load(embedding_path) 
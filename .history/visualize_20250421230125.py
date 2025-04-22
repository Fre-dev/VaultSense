import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Path to your embeddings directory
embeddings_dir = Path("/Users/vijayrajgohil/Documents/VaultSense/embeddings")

# Function to print embedding details
def print_embedding_details(file_path):
    embedding = np.load(file_path)
    print(f"\nFile: {file_path.name}")
    print(f"Shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Min value: {np.min(embedding)}")
    print(f"Max value: {np.max(embedding)}")
    print(f"Mean value: {np.mean(embedding)}")

# Function to visualize embedding
def visualize_embedding(embedding, title):
    plt.figure(figsize=(10, 4))
    plt.plot(embedding)
    plt.title(title)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Process all .npy files in the directory
for npy_file in embeddings_dir.glob("*.npy"):
    # Print details
    print_embedding_details(npy_file)
    
    # Visualize the embedding
    embedding = np.load(npy_file)
    visualize_embedding(embedding, f"Embedding: {npy_file.name}") 
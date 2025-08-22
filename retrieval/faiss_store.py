import os
import json
import numpy as np
import faiss
import pickle # For saving/loading Python objects
import sys # Import sys

# Add the project root to the Python path to enable direct imports from sibling directories.
# This assumes the script is run from the project root or a subdirectory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from embeddings.embedder import EmbeddingModel # Import your EmbeddingModel class
from langchain_core.documents import Document # For consistency with chunker output

def create_and_save_faiss_index(
    chunked_data_path="chunked_data/all_chunks.json",
    faiss_index_path="retrieval/my_index.faiss",
    text_mapping_path="retrieval/text_mapping.pkl"
):
    """
    Loads chunked documents, generates embeddings, creates a FAISS index,
    and saves both the index and a mapping of chunk data.
    
    Args:
        chunked_data_path (str): Path to the JSON file containing chunked documents.
        faiss_index_path (str): Path to save the FAISS index file.
        text_mapping_path (str): Path to save the text mapping (list of chunk objects).
    """
    print("Starting FAISS index creation process...")

    # Ensure the retrieval directory exists for saving
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)

    # 1. Load chunked documents
    if not os.path.exists(chunked_data_path):
        print(f"Error: Chunked data file not found at '{chunked_data_path}'. Please run parsing/chunker.py first.")
        return
    
    try:
        with open(chunked_data_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"Loaded {len(chunks_data)} chunks from '{chunked_data_path}'.")
    except Exception as e:
        print(f"Error loading chunked data: {e}")
        return

    # Convert dictionary-based chunks back to Document objects for easier handling
    # This also prepares the data for text_mapping_pkl
    documents_for_mapping = [
        Document(page_content=item['page_content'], metadata=item['metadata'])
        for item in chunks_data
    ]
    texts_to_embed = [doc.page_content for doc in documents_for_mapping]

    # 2. Initialize embedding model and generate embeddings
    embedder = EmbeddingModel()
    if embedder.model is None:
        print("Embedding model could not be loaded. Aborting FAISS index creation.")
        return

    embeddings = embedder.get_embeddings(texts_to_embed)
    if embeddings is None:
        print("Failed to generate embeddings. Aborting FAISS index creation.")
        return

    # Ensure embeddings are float32, which FAISS often prefers
    embeddings = embeddings.astype('float32')

    # 3. Create FAISS index
    dimension = embeddings.shape[1] # The dimensionality of the embeddings (e.g., 1024 for bge-large)
    
    # Using IndexFlatL2 for simplicity. L2 is Euclidean distance.
    # For very large datasets, you might consider IndexIVFFlat or IndexHNSWFlat for better performance.
    index = faiss.IndexFlatL2(dimension) 
    
    print(f"Created FAISS index with dimension: {dimension}")

    # Add the embeddings to the index
    index.add(embeddings)
    print(f"Added {index.ntotal} embeddings to the FAISS index.")

    # 4. Save the FAISS index
    try:
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to '{faiss_index_path}'.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

    # 5. Save the text mapping using pickle
    try:
        with open(text_mapping_path, 'wb') as f:
            pickle.dump(documents_for_mapping, f)
        print(f"Text mapping saved to '{text_mapping_path}'.")
    except Exception as e:
        print(f"Error saving text mapping: {e}")

    print("\nFAISS index creation and mapping saving complete!")

if __name__ == "__main__":
    create_and_save_faiss_index()
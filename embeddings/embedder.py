import os
import json
from sentence_transformers import SentenceTransformer
import torch # Required for checking CUDA availability and moving models/tensors to GPU

class EmbeddingModel:
    """
    Handles loading the embedding model and generating embeddings for text chunks.
    Uses 'BAAI/bge-large-en-v1.5' by default and leverages CUDA if available.
    """
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing embedding model '{self.model_name}' on device: {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Please ensure you have an internet connection to download the model, or check model name.")
            self.model = None # Set to None if loading fails

    def get_embeddings(self, texts):
        """
        Generates embeddings for a list of text strings.
        
        Args:
            texts (list): A list of strings for which to generate embeddings.
            
        Returns:
            numpy.ndarray: A 2D numpy array of embeddings.
        """
        if self.model is None:
            print("Embedding model not loaded. Cannot generate embeddings.")
            return None
        
        print(f"Generating embeddings for {len(texts)} texts...")
        # Encode_multi_process can be used for larger datasets if needed,
        # but for hackathon scale, direct encode is usually fine.
        embeddings = self.model.encode(texts, 
                                       batch_size=32, # Adjust batch size based on your GPU memory
                                       show_progress_bar=True,
                                       convert_to_tensor=False) # Keep as numpy array for FAISS
        print("Embeddings generated.")
        return embeddings

if __name__ == "__main__":
    # This block demonstrates how to use the EmbeddingModel class.
    # It will load chunks from 'chunked_data/all_chunks.json' and generate embeddings.

    chunked_data_path = os.path.join("chunked_data", "all_chunks.json")

    if not os.path.exists(chunked_data_path):
        print(f"Error: '{chunked_data_path}' not found. Please run parsing/chunker.py first.")
    else:
        try:
            with open(chunked_data_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Extract only the text content for embedding
            texts_to_embed = [chunk['page_content'] for chunk in chunks_data]
            
            # Initialize the embedding model
            embedder = EmbeddingModel()
            
            if embedder.model: # Check if model loaded successfully
                # Generate embeddings
                embeddings = embedder.get_embeddings(texts_to_embed)
                
                if embeddings is not None:
                    print(f"Shape of generated embeddings: {embeddings.shape}")
                    print("Embeddings successfully generated. Ready for FAISS storage.")
                else:
                    print("Failed to generate embeddings.")

        except Exception as e:
            print(f"An error occurred: {e}")

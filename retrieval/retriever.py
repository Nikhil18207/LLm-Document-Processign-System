import os
import faiss
import numpy as np
import pickle # For loading text mapping # Import your DocumentReranker class
from langchain_core.documents import Document # For consistency with chunk objects
from embeddings.embedder import EmbeddingModel 
from embeddings.reranker import DocumentReranker
class DocumentRetriever:
    """
    Handles retrieving and re-ranking relevant documents from a FAISS index based on a query.
    """
    def __init__(
        self,
        faiss_index_path="retrieval/my_index.faiss",
        text_mapping_path="retrieval/text_mapping.pkl"
    ):
        self.faiss_index_path = faiss_index_path
        self.text_mapping_path = text_mapping_path
        self.embedding_model = EmbeddingModel() # Initialize your embedding model
        self.reranker = DocumentReranker()   # Initialize your reranker model

        self.index = None
        self.text_mapping = None

        self._load_index_and_mapping()

    def _load_index_and_mapping(self):
        """Loads the FAISS index and text mapping from disk."""
        print(f"Loading FAISS index from '{self.faiss_index_path}'...")
        if not os.path.exists(self.faiss_index_path):
            print(f"Error: FAISS index file not found at '{self.faiss_index_path}'.")
            print("Please run retrieval/faiss_store.py first to create the index.")
            return

        try:
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"FAISS index loaded successfully with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.index = None

        print(f"Loading text mapping from '{self.text_mapping_path}'...")
        if not os.path.exists(self.text_mapping_path):
            print(f"Error: Text mapping file not found at '{self.text_mapping_path}'.")
            print("Please run retrieval/faiss_store.py first to create the mapping.")
            return

        try:
            with open(self.text_mapping_path, 'rb') as f:
                self.text_mapping = pickle.load(f)
            print(f"Text mapping loaded successfully with {len(self.text_mapping)} entries.")
        except Exception as e:
            print(f"Error loading text mapping: {e}")
            self.text_mapping = None

    def retrieve(self, query: str, initial_top_k: int = 10, final_top_k: int = 5) -> list[Document]:
        """
        Retrieves and re-ranks the most relevant document chunks for a given query.
        
        Args:
            query (str): The user's natural language query.
            initial_top_k (int): The number of top chunks to retrieve from FAISS (before reranking).
            final_top_k (int): The number of final re-ranked chunks to return.
            
        Returns:
            list[Document]: A list of Langchain Document objects, re-ranked by relevance.
        """
        if self.index is None or self.text_mapping is None:
            print("Retriever not fully initialized. Cannot perform retrieval.")
            return []

        # 1. Embed the query
        query_embedding = self.embedding_model.get_embeddings([query])
        
        if query_embedding is None:
            print("Failed to generate embedding for the query.")
            return []

        # Ensure query embedding is float32 and 2D for FAISS search
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # 2. Perform initial similarity search (FAISS)
        print(f"Searching FAISS index for initial top {initial_top_k} relevant chunks...")
        distances, indices = self.index.search(query_embedding, initial_top_k)
        print("Initial FAISS search complete.")

        initial_retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.text_mapping):
                chunk = self.text_mapping[idx]
                chunk.metadata['faiss_score'] = distances[0][i] # Keep original FAISS score
                initial_retrieved_chunks.append(chunk)
            else:
                print(f"Warning: FAISS returned an invalid index: {idx}")

        print(f"Retrieved {len(initial_retrieved_chunks)} chunks from FAISS.")

        # 3. Re-rank the initial retrieved chunks
        if initial_retrieved_chunks:
            reranked_chunks = self.reranker.rerank(query, initial_retrieved_chunks)
            # Take only the final_top_k after reranking
            final_chunks = reranked_chunks[:final_top_k]
            print(f"Returning top {len(final_chunks)} re-ranked chunks.")
            return final_chunks
        else:
            print("No chunks to re-rank.")
            return []

if __name__ == "__main__":
    # Example usage of the DocumentRetriever with reranking
    retriever = DocumentRetriever()

    # Check if the retriever initialized successfully
    if retriever.index and retriever.text_mapping and retriever.reranker.model:
        sample_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        print(f"\nSample Query: '{sample_query}'")
        
        # Retrieve the top relevant chunks after reranking
        # We'll retrieve initial_top_k=10 from FAISS and then re-rank down to final_top_k=5
        relevant_chunks = retriever.retrieve(sample_query, initial_top_k=10, final_top_k=5)

        if relevant_chunks:
            print("\n--- Final Re-ranked Retrieved Chunks ---")
            for i, chunk in enumerate(relevant_chunks):
                print(f"Rank {i+1} (Source: {chunk.metadata.get('source')}, Rerank Score: {chunk.metadata.get('relevance_score'):.4f}):")
                print(f"  Content: {chunk.page_content[:300]}...") # Print first 300 chars
                print(f"  Chunk ID: {chunk.metadata.get('chunk_id')}")
                print("-" * 30)
        else:
            print("No relevant chunks retrieved after reranking.")
    else:
        print("Retriever or Reranker could not be initialized. Please ensure previous steps ran successfully.")


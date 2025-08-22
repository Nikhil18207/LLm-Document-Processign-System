import os
from sentence_transformers import CrossEncoder
import torch
from peft import PeftModel, PeftConfig # Import PeftModel and PeftConfig
from transformers import AutoTokenizer # Needed for loading tokenizer with PEFT model

class DocumentReranker:
    """
    Handles re-ranking of retrieved document chunks based on a query.
    Uses a Cross-Encoder model, with an optional LoRA fine-tuned adapter.
    """
    def __init__(self, 
                 base_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 lora_adapter_path="fine_tuned_reranker_adapters/final_adapter"):
        
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self._load_reranker_model()

    def _load_reranker_model(self):
        """
        Loads the base reranker model and applies the LoRA adapter if available.
        """
        print(f"Initializing base reranker model '{self.base_model_name}' on device: {self.device}")
        try:
            # Load the base CrossEncoder model
            base_reranker = CrossEncoder(self.base_model_name, device=self.device)
            self.model = base_reranker.model # Get the underlying HuggingFace model
            
            # Check if fine-tuned adapter exists and load it
            if os.path.exists(self.lora_adapter_path):
                print(f"Loading LoRA adapter from '{self.lora_adapter_path}'...")
                # Load the PEFT model
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
                print("LoRA adapter loaded and merged successfully.")
                self.model = self.model.merge_and_unload() # Merge LoRA weights into base model for inference
                print("Merged LoRA weights into base model for efficient inference.")
            else:
                print(f"No LoRA adapter found at '{self.lora_adapter_path}'. Using base reranker model.")
            
            self.model.eval() # Set model to evaluation mode
            print("Reranker model (with/without LoRA) loaded successfully.")

        except Exception as e:
            print(f"Error loading reranker model or LoRA adapter: {e}")
            print("Please ensure you have an internet connection, correct paths, and sufficient memory.")
            self.model = None

    def rerank(self, query: str, documents: list) -> list:
        """
        Re-ranks a list of documents based on their relevance to the query.
        
        Args:
            query (str): The user's query.
            documents (list): A list of Langchain Document objects
                                        (initial retrieved chunks).
                                        
        Returns:
            list: The list of Document objects, re-ordered by relevance
                            and with updated 'relevance_score' metadata.
        """
        if self.model is None:
            print("Reranker model not loaded. Cannot perform re-ranking.")
            return documents # Return original order if reranker is not available

        if not documents:
            return []

        # Prepare sentence pairs for the cross-encoder: [(query, doc_content), ...]
        sentence_pairs = [(query, doc.page_content) for doc in documents]
        
        print(f"Re-ranking {len(documents)} documents for the query...")
        
        # Use the base_reranker's predict method which correctly handles tokenization and inference
        # We temporarily create an instance of CrossEncoder just for its predict method,
        # which knows how to use the underlying self.model (which now has LoRA weights merged)
        # This is a bit of a workaround because PeftModel doesn't directly expose .predict() like CrossEncoder
        temp_reranker_instance = CrossEncoder(self.base_model_name, device=self.device)
        temp_reranker_instance.model = self.model # Inject the PEFT-modified model
        
        with torch.no_grad(): # Ensure no gradient calculation during inference
            scores = temp_reranker_instance.predict(sentence_pairs, show_progress_bar=False, batch_size=32)
        
        print("Re-ranking complete.")

        # Pair documents with their new scores
        scored_documents = []
        for i, doc in enumerate(documents):
            doc.metadata['relevance_score'] = float(scores[i])
            scored_documents.append(doc)

        # Sort documents by their new relevance score in descending order
        reranked_documents = sorted(scored_documents, key=lambda x: x.metadata['relevance_score'], reverse=True)
        
        return reranked_documents

if __name__ == "__main__":
    # Example usage:
    reranker = DocumentReranker()

    if reranker.model:
        sample_query = "What are the common side effects of a new medication?"
        
        dummy_documents = [
            # ... (your dummy documents from before, kept for testing)
            # Make sure these are actual Langchain Document objects or dictionaries that can be converted
            {"page_content": "This medication helps with pain, but it might cause mild headaches.", "metadata":{"source": "med_guide.txt", "chunk_id": "med_chunk_1"}},
            {"page_content": "The new drug has been approved for public use after successful trials.", "metadata":{"source": "news_article.txt", "chunk_id": "news_chunk_5"}},
            {"page_content": "Common side effects include nausea, dizziness, and fatigue. Consult your doctor if symptoms persist.", "metadata":{"source": "med_guide.txt", "chunk_id": "med_chunk_2"}},
            {"page_content": "Patients over 65 should take a reduced dosage.", "metadata":{"source": "dosage_info.txt", "chunk_id": "dosage_chunk_3"}},
            {"page_content": "The medication comes in pill and liquid form.", "metadata":{"source": "packaging.txt", "chunk_id": "packaging_chunk_1"}}
        ]
        
        # Convert dummy_documents to Langchain Document objects for consistency with expected input type
        from langchain_core.documents import Document as LCDocument
        dummy_documents_lc = [LCDocument(page_content=d['page_content'], metadata=d['metadata']) for d in dummy_documents]

        print(f"\nSample Query: '{sample_query}'")
        print("\n--- Initial (Unranked) Documents ---")
        for i, doc in enumerate(dummy_documents_lc):
            print(f"Doc {i+1} (Source: {doc.metadata.get('source')}): {doc.page_content[:50]}...")

        reranked_documents = reranker.rerank(sample_query, dummy_documents_lc)

        if reranked_documents:
            print("\n--- Reranked Documents ---")
            for i, chunk in enumerate(reranked_documents):
                print(f"Rank {i+1} (Source: {chunk.metadata.get('source')}, Score: {chunk.metadata.get('relevance_score'):.4f}):")
                print(f"  Content: {chunk.page_content[:100]}...")
                print("-" * 30)
    else:
        print("Reranker could not be initialized for testing.")

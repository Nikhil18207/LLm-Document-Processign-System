import os
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_processed_texts(processed_text_dir="processed_text") -> List[Dict[str, str]]:
    """
    Loads text from all .txt files in the specified directory.

    Args:
        processed_text_dir (str): The directory containing the processed text files.

    Returns:
        list: A list of dictionaries, where each dictionary contains the text
              and original source filename.
    """
    texts = []
    print(f"Loading texts from '{processed_text_dir}'...")
    if not os.path.exists(processed_text_dir):
        print(f"Error: Directory '{processed_text_dir}' not found. Please run pdf_parser.py first.")
        return texts

    for filename in os.listdir(processed_text_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(processed_text_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                texts.append({"text": content, "source": filename})
                print(f"Loaded '{filename}' for chunking.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts

def chunk_documents_hierarchically(documents: List[Dict[str, str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Chunks a list of documents using a hierarchical approach to preserve semantic units.
    
    Args:
        documents (list): A list of dictionaries, each with 'text' and 'source' keys.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.
        
    Returns:
        list: A list of Document objects, each representing a chunk with metadata.
    """
    # Define a list of separators that indicate logical breaks in your documents.
    # This is customized for the provided insurance policy documents.
    # We prioritize larger structural breaks first.
    separators = [
        "\n\nSection", 
        "\n\nSECTION",
        "\n\nClause",
        "\n\nDef.",
        "\n\nSl. No.",
        "\n\n-",
        "\n",  # Fallback to single newlines
        " ",   # Final fallback to spaces
    ]

    # Initialize the text splitter with the custom separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )

    all_chunks = []
    print(f"\nChunking documents using a hierarchical approach (Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap})...")
    for doc in documents:
        # Create a Langchain Document object for consistent processing
        lc_doc = Document(page_content=doc['text'], metadata={"source": doc['source']})
        
        # Split the document into chunks
        chunks = text_splitter.split_documents([lc_doc])
        
        # Add a unique chunk ID and append to the list
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{os.path.splitext(doc['source'])[0]}_chunk_{i}"
            all_chunks.append(chunk)
        print(f"Chunked '{doc['source']}' into {len(chunks)} pieces.")
            
    return all_chunks

if __name__ == "__main__":
    # Define the directory where processed texts are stored
    processed_text_directory = "processed_text"
    
    # Define the directory to save the chunked data
    chunked_data_output_dir = "chunked_data"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(chunked_data_output_dir):
        os.makedirs(chunked_data_output_dir)

    # 1. Load the processed text files
    loaded_docs = load_processed_texts(processed_text_directory)

    if loaded_docs:
        # 2. Chunk the loaded documents with the new hierarchical logic
        # Adjust chunk_size and chunk_overlap as needed
        chunked_documents = chunk_documents_hierarchically(loaded_docs, chunk_size=1000, chunk_overlap=200)

        print(f"\nTotal chunks created: {len(chunked_documents)}")

        # 3. Save the chunked documents to a JSON file
        output_filepath = os.path.join(chunked_data_output_dir, "all_chunks.json")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Convert Document objects to dictionaries for JSON serialization
            json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in chunked_documents], f, indent=4)
        print(f"Saved all chunks to '{output_filepath}'")
        
        # Optional: Print a sample chunk to verify
        if chunked_documents:
            print("\n--- Sample of a new chunk ---")
            print(f"Content: {chunked_documents[0].page_content[:300]}...")
            print(f"Metadata: {chunked_documents[0].metadata}")
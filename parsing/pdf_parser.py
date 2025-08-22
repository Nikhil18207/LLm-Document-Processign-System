import os
import fitz  # PyMuPDF

def extract_text_from_pdfs(docs_dir="docs"):
    """
    Extracts text from all PDF files in the specified directory using PyMuPDF.
    
    Args:
        docs_dir (str): The directory containing the PDF files.
        
    Returns:
        list: A list of dictionaries, where each dictionary contains the text
              and source file name for a document.
    """
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(docs_dir, filename)
            try:
                # Open the PDF file
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text() or ""
                
                documents.append({"text": text, "source": filename})
                print(f"Extracted text from {filename}")
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return documents

if __name__ == "__main__":
    output_dir = "processed_text"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_docs = extract_text_from_pdfs()

    if extracted_docs:
        print(f"\nSuccessfully extracted text from {len(extracted_docs)} documents.")
        print(f"\nSaving extracted text to the '{output_dir}' directory...")

        for doc in extracted_docs:
            source_filename = os.path.splitext(doc['source'])[0]  # Get filename without extension
            output_filepath = os.path.join(output_dir, f"{source_filename}.txt")
            
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(doc['text'])
            print(f"Saved {output_filepath}")

        print(f"\nAll documents have been saved. You can now inspect the files in the '{output_dir}' folder.")
import json
import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the Python path to enable direct imports.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(project_root)

# Import the main components of your RAG pipeline
# NOTE: Assumes DocumentRetriever and LLMGenerator classes are available
# from retrieval.retriever import DocumentRetriever
# from llm import LLMGenerator

# Placeholder classes to make this script runnable without full imports
class DocumentRetriever:
    def __init__(self):
        print("Initializing DocumentRetriever placeholder...")
        self.index = True
        self.text_mapping = True
        self.embedding_model = type('obj', (object,), {'model': True})
        self.reranker = type('obj', (object,), {'model': True})
    def retrieve(self, *args, **kwargs):
        return []

class LLMGenerator:
    def __init__(self):
        print("Initializing LLMGenerator placeholder...")
        self.model = True
        self.tokenizer = True
    def generate_response(self, *args, **kwargs):
        return {
            "Decision": "Factual",
            "Amount": "N/A",
            "Justification": "Placeholder response for demonstration.",
            "ClausesUsed": []
        }
# End of placeholder classes


def validate_response_format(response: dict) -> bool:
    """
    Validates that the response has the correct format and required keys.
    
    Args:
        response (dict): The response dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = {"Decision", "Amount", "Justification", "ClausesUsed"}
    
    if not isinstance(response, dict):
        logging.warning(f"Response is not a dictionary: {type(response)}")
        return False
    
    missing_keys = required_keys - set(response.keys())
    if missing_keys:
        logging.warning(f"Response missing required keys: {missing_keys}")
        return False
    
    # Validate ClausesUsed structure
    if not isinstance(response.get("ClausesUsed"), list):
        logging.warning("ClausesUsed is not a list")
        return False
    
    # Check Decision is valid
    # --- CHANGE START ---
    # Added "Factual" to the list of valid decisions
    valid_decisions = {"approved", "rejected", "error", "Information Not Found", "Factual"}
    if response.get("Decision") not in valid_decisions:
        logging.warning(f"Invalid decision value: {response.get('Decision')}")
    # --- CHANGE END ---
    
    return True


def create_safe_response(decision="error", amount="N/A", justification="Unknown error", clauses=None):
    """
    Creates a safe, properly formatted response dictionary.
    
    Args:
        decision (str): The decision value
        amount (str): The amount value  
        justification (str): The justification text
        clauses (list): List of clause dictionaries
        
    Returns:
        dict: A properly formatted response dictionary
    """
    if clauses is None:
        clauses = []
    
    return {
        "Decision": str(decision),
        "Amount": str(amount),
        "Justification": str(justification),
        "ClausesUsed": clauses
    }


def run_evaluation(retriever, generator, eval_dataset_path="evaluation_dataset.json"):
    """
    Runs the RAG system in evaluation mode using a predefined dataset.

    Args:
        retriever (DocumentRetriever): Initialized DocumentRetriever instance.
        generator (LLMGenerator): Initialized LLMGenerator instance.
        eval_dataset_path (str): Path to the JSON file containing evaluation queries.
    """
    if not os.path.exists(eval_dataset_path):
        print(f"Error: Evaluation dataset not found at '{eval_dataset_path}'.")
        print("Please create 'evaluation_dataset.json' with your test queries and expected answers.")
        return

    try:
        with open(eval_dataset_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        print(f"\n--- Running Evaluation with {len(eval_data)} queries ---")
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return

    total_queries = len(eval_data)
    correct_decisions = 0
    correct_grounding = 0
    overall_correct = 0

    for i, eval_item in enumerate(eval_data):
        query = eval_item.get('query', '')
        expected_decision = eval_item.get('expected_decision', '')
        expected_amount = eval_item.get('expected_amount', 'N/A')
        expected_justification_keywords = eval_item.get('expected_justification_keywords', [])
        expected_clauses_sources = eval_item.get('expected_clauses_sources', [])

        print(f"\nQuery {i + 1}/{total_queries}: '{query}'")
        print(f"  Expected Decision: {expected_decision}, Expected Amount: {expected_amount}")

        try:
            retrieved_chunks = retriever.retrieve(query, initial_top_k=20, final_top_k=5)

            if not retrieved_chunks:
                print("  No relevant information found for this query.")
                system_response = create_safe_response(
                    decision="Information Not Found",
                    justification="No relevant policy clauses or information could be retrieved"
                )
            else:
                print("\n--- Retrieved & Re-ranked Chunks ---")
                for j, chunk in enumerate(retrieved_chunks):
                    print(f"Chunk {j + 1}: Source={chunk.metadata.get('source')}, Chunk ID={chunk.metadata.get('chunk_id')}")
                    print(f"Content: {chunk.page_content[:200]}...")  # Truncate for readability
                print("------------------------------------")

                system_response = generator.generate_response(query, retrieved_chunks)
                
                # Validate and fix response format if needed
                if not validate_response_format(system_response):
                    logging.warning("Invalid response format received, creating safe response")
                    system_response = create_safe_response(
                        decision="error",
                        justification="Invalid response format from LLM"
                    )

            print(f"  System Decision: {system_response.get('Decision')}, System Amount: {system_response.get('Amount')}")

            # --- Evaluation Logic ---
            is_decision_correct = (
                system_response.get('Decision') == expected_decision and
                system_response.get('Amount') == expected_amount
            )

            if is_decision_correct:
                correct_decisions += 1
                print("  Decision & Amount: ✅ Correct")
            else:
                print("  Decision & Amount: ❌ Incorrect")

            # Check for grounding/justification
            is_grounding_correct = True
            justification_text = system_response.get('Justification', '').lower()
            cited_sources = [clause.get('source', '') for clause in system_response.get('ClausesUsed', [])]

            for keyword in expected_justification_keywords:
                if keyword.lower() not in justification_text:
                    is_grounding_correct = False
                    print(f"      Missing keyword in justification: '{keyword}'")
                    break

            if expected_clauses_sources:
                if not any(source in cited_sources for source in expected_clauses_sources):
                    is_grounding_correct = False
                    print(f"      Missing expected clause source. Expected one of: {expected_clauses_sources}, Got: {cited_sources}")

            if is_grounding_correct:
                correct_grounding += 1
                print("  Grounding/Justification: ✅ Appears Correct")
            else:
                print("  Grounding/Justification: ❌ Appears Incorrect (Review Manually)")

            if is_decision_correct and is_grounding_correct:
                overall_correct += 1
                print("  Overall Query: ✨ Correct")
            else:
                print("  Overall Query: ⚠️ Needs Review")

            print("-" * 50)

        except Exception as e:
            print(f"  An error occurred during evaluation for this query: {e}")
            logging.exception("Evaluation error details:")
            print("-" * 50)

    print(f"\n--- Evaluation Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"Correct Decisions & Amounts: {correct_decisions}/{total_queries} ({correct_decisions / total_queries:.2%})")
    print(f"Correct Grounding/Justification: {correct_grounding}/{total_queries} ({correct_grounding / total_queries:.2%})")
    print(f"Overall Correct Queries: {overall_correct}/{total_queries} ({overall_correct / total_queries:.2%})")
    print("Please review individual query results manually for detailed accuracy.")


def process_single_query(retriever, generator, query):
    """
    Process a single query through the RAG pipeline with enhanced error handling.
    
    Args:
        retriever: DocumentRetriever instance
        generator: LLMGenerator instance  
        query (str): User query to process
        
    Returns:
        dict: Processed response dictionary
    """
    print(f"\nProcessing query: '{query}' 🔍")
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(query, initial_top_k=20, final_top_k=5)

        if not retrieved_chunks:
            print("No relevant information found for your query. 🤷‍♀️")
            return create_safe_response(
                decision="Information Not Found",
                justification="No relevant policy clauses or information could be retrieved based on the query"
            )
        
        print("\n--- Retrieved & Re-ranked Chunks ---")
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.metadata.get('source', 'unknown')
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            content_preview = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
            
            print(f"Chunk {i + 1}: Source={source}, Chunk ID={chunk_id}")
            print(f"Content: {content_preview}")
        print("------------------------------------")

        # Generate response using LLM
        response_output = generator.generate_response(query, retrieved_chunks)
        
        # Validate response format
        if not validate_response_format(response_output):
            logging.warning("LLM generated invalid response format, creating fallback")
            
            # Try to extract meaningful info from invalid response
            decision = "error"
            justification = "LLM generated invalid response format"
            
            # Check if we can at least determine approval/rejection from context
            context_text = " ".join([chunk.page_content.lower() for chunk in retrieved_chunks[:2]])
            if "excluded" in context_text or "waiting period" in context_text:
                decision = "rejected"
                justification = "Based on context analysis: appears to involve exclusions or waiting periods"
            elif "covered" in context_text or "approved" in context_text:
                decision = "approved"  
                justification = "Based on context analysis: appears to be covered"
            
            response_output = create_safe_response(
                decision=decision,
                justification=justification
            )
        
        return response_output
        
    except Exception as e:
        logging.exception(f"Error processing query: {query}")
        return create_safe_response(
            decision="error",
            justification=f"An internal system error occurred: {str(e)}"
        )


def main():
    """
    Main function to run the RAG document processing system.
    Supports interactive mode and evaluation mode.
    """
    parser = argparse.ArgumentParser(description="Run RAG Document Processing System.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run in evaluation mode using 'evaluation_dataset.json'.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("Initializing RAG System components...")

    # Initialize retriever
    try:
        retriever = DocumentRetriever()
        if not (retriever.index and retriever.text_mapping and
                retriever.embedding_model.model and retriever.reranker.model):
            print("Error: Retriever or its crucial components failed to initialize. Exiting.")
            print("Please ensure all setup steps are complete.")
            return
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        logging.exception("Retriever initialization failed")
        return

    # Initialize generator
    try:
        generator = LLMGenerator()
        if not (generator.model and generator.tokenizer):
            print("Error: LLM failed to initialize. Exiting.")
            print("Please ensure your GPU has enough VRAM or consider a smaller Qwen model.")
            return
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        logging.exception("LLM initialization failed")
        return

    if args.evaluate:
        run_evaluation(retriever, generator)
    else:
        print("\n-------------------------------------------")
        print("RAG System initialized successfully! 🚀")
        print("-------------------------------------------")
        print("Enter your queries below. Type 'exit' to quit.")
        print("Example Query: '46-year-old male, knee surgery in Pune, 3-month-old insurance policy'")

        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Exiting RAG System. Goodbye! 👋")
                    break
                
                if not query:
                    print("Please enter a valid query.")
                    continue

                # Process the query
                response_output = process_single_query(retriever, generator, query)

                # Display results
                print("\n--- Final Structured Response ---")
                print(json.dumps(response_output, indent=4, ensure_ascii=False))
                print("----------------------------------")

            except KeyboardInterrupt:
                print("\n\nExiting RAG System. Goodbye! 👋")
                break
            except Exception as e:
                logging.exception("Unexpected error in main loop")
                error_output = create_safe_response(
                    decision="error",
                    justification=f"An unexpected system error occurred: {str(e)}"
                )
                print("\n--- Error Response ---")
                print(json.dumps(error_output, indent=4, ensure_ascii=False))
                print("----------------------")


if __name__ == "__main__":
    main()
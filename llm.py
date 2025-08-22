import json
import re
import logging
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.documents import Document


REQUIRED_KEYS = ["Decision", "Justification", "ClausesUsed"]


def _normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"```(?:json)?", "", t)
    t = t.replace("```", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _tokenize_words(t: str) -> List[str]:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return [w for w in t.split() if w]


def _justification_overlap(expected: str, actual: str) -> float:
    """
    Simple fuzzy match: Jaccard overlap of word sets.
    """
    ew = set(_tokenize_words(expected))
    aw = set(_tokenize_words(actual))
    if not ew:
        return 0.0
    inter = len(ew & aw)
    union = len(ew | aw)
    return inter / union if union else 0.0


class LLMGenerator:
    """
    Handles loading the Large Language Model and generating structured responses.
    """

    def __init__(self, model_name: str = "Qwen/Qwen1.5-7B-Chat", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing LLM '{self.model_name}' on device: {self.device}")
        self.tokenizer = None
        self.model = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Ensure a pad token exists (some causal models lack one)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.tokenizer = None
            self.model = None

    def create_dynamic_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Strict instruction to force JSON output with mandatory keys.
        """
        context_text = ""
        for chunk in context_chunks:
            source = chunk.get("source", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            content = chunk.get("content", "")
            context_text += f"Source: {source}, Chunk ID: {chunk_id}\nContent: {content}\n\n"

        prompt = f"""You are an insurance policy assistant.
Analyze the user's query and the provided policy clauses, and return ONLY a JSON object with EXACTLY these fields:

{{
  "Decision": "correct" | "incorrect" | "Factual" | "Insufficient Information",
  "Justification": "A clear, concise explanation or factual answer. Cite clause ids/names inline where helpful.",
  "ClausesUsed": ["source:chunk_id", "source:chunk_id"]
}}

Rules:
- Output MUST be a single valid JSON object. No markdown. No code fences. No extra text.
- Always include all three fields.
- If the context has no relevant information, set:
  "Decision": "Insufficient Information",
  "Justification": "No context available.",
  "ClausesUsed": []

Query:
{query}

Context:
{context_text}

JSON object:
"""
        return prompt

    def create_fallback_response(self, error_message: str) -> dict:
        """Create a valid fallback response when parsing fails."""
        return {
            "Decision": "error",
            "Justification": f"Error: {error_message}",
            "ClausesUsed": [],
        }

    def extract_json_aggressively(self, text: str) -> dict:
        """
        Extract JSON robustly. Strips code fences, finds the first balanced JSON object,
        sanitizes minor trailing-comma issues, and ensures REQUIRED_KEYS exist.
        """
        try:
            if not text or not text.strip():
                return self.create_fallback_response("Empty model output")

            # Strip common markdown wrappers pre-emptively
            text = _normalize_text(text)

            # Find first '{' and last matching '}' (balanced)
            first_brace = text.find("{")
            if first_brace == -1:
                return self.create_fallback_response("No JSON object found")

            text = text[first_brace:]

            brace_count = 0
            end_pos = -1
            for i, ch in enumerate(text):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            if end_pos == -1:
                return self.create_fallback_response("Incomplete JSON object")

            json_str = text[:end_pos]

            # Light sanitization (do NOT remove backslashes blindly — it can break JSON)
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)

            parsed = json.loads(json_str)

            # Ensure required fields exist with correct types
            for key in REQUIRED_KEYS:
                if key not in parsed:
                    parsed[key] = [] if key == "ClausesUsed" else "error"

            if not isinstance(parsed["ClausesUsed"], list):
                parsed["ClausesUsed"] = []

            # Normalize Justification whitespace
            if isinstance(parsed["Justification"], str):
                parsed["Justification"] = parsed["Justification"].strip()

            return parsed

        except Exception as e:
            logging.warning(f"JSON extraction failed: {e}")
            return self.create_fallback_response(f"JSON parsing failed: {str(e)}")

    def generate_response(self, query: str, retrieved_documents: List[Document]) -> dict:
        """
        Generates a structured JSON response based on the query and retrieved documents.
        """
        if self.tokenizer is None or self.model is None:
            print("LLM not loaded. Cannot generate response.")
            return self.create_fallback_response("LLM not initialized")

        try:
            context_chunks = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "content": doc.page_content.strip(),
                }
                for doc in retrieved_documents[:3]
            ]

            prompt = self.create_dynamic_prompt(query, context_chunks)
            print(f"Generated prompt length: {len(prompt)}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=300,
                    do_sample=False,  # deterministic for eval
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            print(f"Raw LLM Response (new tokens only): {response_text}")
            return self.extract_json_aggressively(response_text)

        except Exception as e:
            logging.exception("LLM generation error")
            return self.create_fallback_response(f"LLM generation failed: {str(e)}")


def evaluate_prediction(
    expected: Dict[str, Any],
    predicted: Dict[str, Any],
    just_overlap_threshold: float = 0.15,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates a single prediction against expected.
    - Decision must match (case-insensitive).
    - Justification must have a minimum overlap with expected (fuzzy Jaccard).
      If expected justification is very short (< 6 words), also allow substring match.
    """
    expected_dec = (expected.get("Decision") or "").strip().lower()
    expected_jus = (expected.get("Justification") or "").strip()
    pred_dec = (predicted.get("Decision") or "").strip().lower()
    pred_jus = (predicted.get("Justification") or "").strip()

    decision_match = (expected_dec == pred_dec)

    # Fuzzy match for justification
    overlap = _justification_overlap(expected_jus, pred_jus)
    short_expected = len(_tokenize_words(expected_jus)) < 6
    substring_ok = expected_jus.lower() in pred_jus.lower() if expected_jus else False
    justification_match = overlap >= just_overlap_threshold or (short_expected and substring_ok)

    is_correct = decision_match and (justification_match or expected_dec in ["insufficient information", "incorrect"])
    details = {
        "decision_match": decision_match,
        "overlap": round(overlap, 3),
        "substring_ok": substring_ok,
        "expected_decision": expected_dec,
        "predicted_decision": pred_dec,
    }
    return is_correct, details


if __name__ == "__main__":
    generator = LLMGenerator()

    if generator.model and generator.tokenizer:
        # Load evaluation dataset
        try:
            with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
                evaluation_data = json.load(f)
        except Exception as e:
            print(f"Error loading evaluation_dataset.json: {e}")
            evaluation_data = []

        correct = 0
        total = len(evaluation_data)

        for i, entry in enumerate(evaluation_data):
            query = entry.get("query", "").strip()

            # Expecting structured target under "expected_answer"
            expected = entry.get("expected_answer", {})
            if not expected:
                print(f"\nSkipping item {i+1}: missing 'expected_answer'")
                continue

            # Build retrieved docs from any provided contexts (optional)
            retrieved_docs: List[Document] = []
            contexts = entry.get("positive_contexts") or entry.get("contexts") or []
            for j, content in enumerate(contexts[:3]):
                retrieved_docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": f"{i+1}.txt", "chunk_id": f"{i+1}_chunk_{j+1}"},
                    )
                )

            print(f"\nEvaluating Query {i+1}/{total}: '{query}'")
            print("\n--- Generating LLM Response ---")

            response = generator.generate_response(query, retrieved_docs)

            # Score
            is_correct, details = evaluate_prediction(expected, response)

            if is_correct:
                correct += 1

            expected_j = expected.get("Justification", "") or ""
            actual_j = response.get("Justification", "") or ""

            print("\n--- Evaluation Result ---")
            print(f"Correct: {is_correct}")
            print(f"Decision Match: {details['decision_match']}")
            print(f"Justification Overlap: {details['overlap']}")
            print(f"Expected Decision: {details['expected_decision']}")
            print(f"Actual Decision: {details['predicted_decision']}")
            print(f"Expected Justification (partial): {expected_j[:120]}...")
            print(f"Actual Justification (partial): {actual_j[:120]}...")
            print(f"Full Response: {json.dumps(response, indent=2)}")

        accuracy = (correct / total * 100) if total > 0 else 0
        print("\n--- Evaluation Summary ---")
        print(f"Total Queries: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("LLMGenerator could not be initialized for testing.")

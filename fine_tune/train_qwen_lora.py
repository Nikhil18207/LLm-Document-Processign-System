import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

def load_dataset_from_json(json_path):
    """Loads a dataset from a JSON file into HuggingFace Dataset."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found at: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_list(data)

def train_qwen_lora(
    model_name="Qwen/Qwen1.5-7B-Chat",
    dataset_path="fine_tune/training_dataset.json",
    output_dir="fine_tuned_qwen_lora_adapters",
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_seq_length=128,
):
    """
    Performs LoRA fine-tuning on the Qwen model.
    Consumes 'query', 'positive_contexts', and 'negatives' fields from training_dataset.json.
    Optimized for RTX 4060 (lower memory footprint & reduced heat).
    """
    print(f"Starting LoRA fine-tuning for '{model_name}'...")
    print(f"Loading dataset from: {dataset_path}")

    # Clear GPU memory
    torch.cuda.empty_cache()

    # 1. Load the dataset
    try:
        dataset = load_dataset_from_json(dataset_path)
        print(f"Dataset loaded successfully with {len(dataset)} examples.")
        required_fields = ["query", "positive_contexts"]
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Dataset must contain '{field}' column.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Tokenizer and Base Model
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,  # Simplified quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    print("Base model loaded and prepared for k-bit training.")

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print("PEFT (LoRA) model created.")
    model.print_trainable_parameters()

    # 4. Define a formatting function to create the 'text' column
    def formatting_func(example):
        pos_context = "\n".join(str(x) for x in example["positive_contexts"] if x is not None)
        neg_context = "\n".join(str(x) for x in example.get("negatives", []) if x is not None)

        user_prompt = f"Relevant Context:\n{pos_context}"
        if neg_context.strip():
            user_prompt += f"\n\nIrrelevant Context (ignore):\n{neg_context}"

        user_prompt += f"\n\nQuestion:\n{example['query']}"

        messages = [
            {"role": "system", "content": "You are a helpful and accurate assistant. Only use relevant context when answering and ignore irrelevant context."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 5. Prepare dataset with 'text' column and apply truncation
    print("Preparing dataset with formatted text...")
    formatted_dataset = dataset.map(
        lambda example: {"text": formatting_func(example)},
        desc="Applying formatting function to train dataset"
    )

    # Tokenize dataset with truncation to respect max_seq_length
    def tokenize_function(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        # Add labels for causal language modeling (same as input_ids for SFT)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        desc="Tokenizing dataset with truncation",
        remove_columns=formatted_dataset.column_names  # Remove original columns, keep tokenized ones
    )

    # Debug: Print sample dataset entries
    for i in range(min(2, len(formatted_dataset))):
        print(f"Sample {i}: {formatted_dataset[i]['text']}")

    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=50,
        save_steps=200,
        optim="adamw_torch",  # Use standard AdamW
        save_total_limit=1,
        fp16=False,  # Disable fp16 to avoid GradScaler issues
        bf16=True,   # Use bf16 to match quantization dtype
        fp16_full_eval=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
    )

    # 7. Initialize SFTTrainer and Start Training
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 8. Save the fine-tuned adapter weights
    final_output_dir = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"\nFine-tuning complete! LoRA adapter saved to: {final_output_dir}")

if __name__ == "__main__":
    train_qwen_lora(
        dataset_path="fine_tune/training_dataset.json",
        output_dir="fine_tuned_qwen_lora_adapters",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=128
    )
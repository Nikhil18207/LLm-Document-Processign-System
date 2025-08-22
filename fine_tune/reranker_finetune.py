import os
import json
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
import logging
from typing import Dict, Any

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset_from_json(json_path):
    """Loads a dataset from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found at: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate dataset
    valid_data = []
    for i, item in enumerate(data):
        if not item.get('query') or not item.get('positive_contexts') or not item.get('negatives'):
            logger.warning(f"Invalid entry at index {i}: {item}")
        else:
            valid_data.append(item)
    
    logger.info(f"Loaded {len(valid_data)} valid entries out of {len(data)} total")
    return valid_data

class RerankerDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for reranker fine-tuning.
    Prepares (query, positive_context) and (query, negative) pairs and tokenizes them.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = []
        
        # Balance positive and negative examples
        positive_count = 0
        negative_count = 0

        for item in data:
            query = item['query']
            positive_contexts = item['positive_contexts']
            negatives = item['negatives']

            # Validate inputs
            if not query or not positive_contexts or not negatives:
                logger.warning(f"Skipping invalid entry: {item}")
                continue

            # Positive pairs
            for positive in positive_contexts:
                if not positive.strip():  # Skip empty positive contexts
                    continue
                    
                encoding = self.tokenizer(
                    query, positive,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',  # Changed to max_length padding
                    return_tensors='pt'
                )
                
                self.features.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': torch.tensor(1.0, dtype=torch.float)
                })
                positive_count += 1

            # Negative pairs (limit to balance dataset)
            for neg in negatives[:len(positive_contexts)]:  # Balance pos/neg ratio
                if not neg.strip():  # Skip empty negatives
                    continue
                    
                encoding = self.tokenizer(
                    query, neg,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',  # Changed to max_length padding
                    return_tensors='pt'
                )
                
                self.features.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': torch.tensor(0.0, dtype=torch.float)
                })
                negative_count += 1
        
        logger.info(f"Dataset created with {positive_count} positive and {negative_count} negative examples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class CustomTrainer(Trainer):
    """Custom trainer with better loss computation for binary classification."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # Get logits and apply sigmoid for binary classification
        logits = outputs.logits
        if logits.dim() > 1:
            logits = logits.view(-1)  # Flatten if needed
            
        # Use BCEWithLogitsLoss for numerical stability
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def train_reranker_lora(
    model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
    dataset_path="fine_tune/training_dataset.json",
    output_dir="fine_tuned_reranker_adapters",
    learning_rate=5e-5,  # Slightly higher learning rate
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Increased batch size
    gradient_accumulation_steps=1,   # Reduced since we increased batch size
    warmup_ratio=0.1,                # Add warmup
    weight_decay=0.01,               # Add weight decay
):
    """
    Performs LoRA fine-tuning on a Cross-Encoder reranker model.
    """
    logger.info(f"Starting LoRA fine-tuning for reranker '{model_name}'...")
    logger.info(f"Loading dataset from: {dataset_path}")

    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("Tokenizer loaded for dataset processing.")

    # 2. Load and validate the dataset
    try:
        raw_data = load_dataset_from_json(dataset_path)
        logger.info(f"Raw dataset loaded successfully with {len(raw_data)} examples.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Create the custom dataset
    train_dataset = RerankerDataset(raw_data, tokenizer, max_length=512)
    logger.info(f"Prepared training dataset with {len(train_dataset)} tokenized examples.")

    if len(train_dataset) == 0:
        logger.error("No valid training examples found. Please check your dataset.")
        return

    # 3. Load Base Model directly (not through CrossEncoder)
    logger.info("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,  # Binary classification
        torch_dtype=torch.float32  # Use FP32 to avoid gradient issues
    )
    
    # Resize token embeddings if necessary
    model.resize_token_embeddings(len(tokenizer))

    # 4. Configure LoRA with better target modules
    lora_config = LoraConfig(
        r=16,  # Increased rank
        lora_alpha=32,  # Increased alpha
        target_modules=["query", "key", "value", "dense"],  # More target modules
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    peft_model = get_peft_model(model, lora_config)
    logger.info("PEFT (LoRA) model created for reranker.")
    peft_model.print_trainable_parameters()

    # 5. Define Training Arguments with better settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_strategy="steps",
        eval_strategy="no",  # No eval set provided
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        optim="adamw_torch",
        save_total_limit=2,
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Use BF16 if available
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # Disable to avoid gradient issues
        max_grad_norm=1.0,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Disable multiprocessing
    )

    # 6. Initialize Custom Trainer
    logger.info("Initializing Custom Trainer...")
    trainer = CustomTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # Updated parameter name
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        ),
    )

    # 7. Start Training
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 8. Save the fine-tuned adapter weights
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    peft_model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)  # Also save tokenizer
    
    logger.info(f"Fine-tuning complete! LoRA reranker adapter saved to: {final_adapter_dir}")
    
    # 9. Save training info
    training_info = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "training_examples": len(train_dataset),
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),  # Convert set to list
            "lora_dropout": lora_config.lora_dropout,
        }
    }
    
    with open(os.path.join(final_adapter_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

if __name__ == "__main__":
    train_reranker_lora(
        dataset_path="fine_tune/training_dataset.json",
        output_dir="fine_tuned_reranker_adapters",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
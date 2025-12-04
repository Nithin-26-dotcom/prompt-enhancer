# ===========================================
# Step 0: Install Dependencies
# ===========================================
!pip install -q datasets transformers accelerate sentencepiece

# ===========================================
# Step 1: Imports & Device Setup
# ===========================================
import torch
import os
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import warnings
import inspect
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# This is the correct device setup for Google Colab
# It will automatically use the free T4 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device.upper()}")
if torch.cuda.is_available():
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not available. Running on CPU. This will be very slow.")
    print("Go to Runtime > Change runtime type > and select T4 GPU.")

# ===========================================
# Step 2: Load & Prepare Dataset
# ===========================================
print("\nLoading dataset...")
# Using 5k samples from your notebook for a quick run
dataset = load_dataset('vera365/lexica_dataset', split='train')
dataset = dataset.shuffle(seed=42).select(range(5000))

# remove unused fields
drop_cols = ['image', 'id', 'promptid', 'width', 'height', 'seed', 'grid',
             'model', 'nsfw', 'modifier10_vector']
dataset = dataset.remove_columns(drop_cols)

def preprocess(example):
    # Input format from your notebook: "enhance: {subject}"
    example['input_text'] = f"enhance: {example['subject']}"
    example['target_text'] = example['prompt']
    return example

dataset = dataset.map(preprocess)

train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_data = train_test['train']
test_data = train_test['test']

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print("Dataset loaded successfully!")

# ===========================================
# Step 3: Tokenizer
# ===========================================
model_name = "t5-base"
print(f"\nLoading tokenizer: {model_name}")
# Using legacy=True from your notebook
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)

def tokenize(batch):
    # Tokenize inputs
    model_inputs = tokenizer(
        batch['input_text'],
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Tokenize labels
    labels = tokenizer(
        batch['target_text'],
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Replace pad tokens with -100 for loss calculation
    labels_ids = []
    for seq in labels["input_ids"]:
        labels_ids.append([token if token != tokenizer.pad_token_id else -100 for token in seq])

    model_inputs["labels"] = labels_ids
    return model_inputs

print("Tokenizing train dataset...")
tokenized_train = train_data.map(tokenize, batched=True, batch_size=32, remove_columns=train_data.column_names)
print("Tokenizing test dataset...")
tokenized_test = test_data.map(tokenize, batched=True, batch_size=32, remove_columns=test_data.column_names)

# Set format to PyTorch tensors (from your notebook)
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Tokenization complete!")

# ===========================================
# Step 4: Load Model
# ===========================================
print(f"\nLoading model: {model_name}")
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device) # Move model to the correct device (GPU)
print("Model loaded successfully!")

# ===========================================
# Step 5: Training Args
# ===========================================
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True, # This enables mixed-precision training (requires GPU)
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none", # Disables wandb/etc.
    remove_unused_columns=True,
    seed=42,
    dataloader_pin_memory=True,
    dataloader_num_workers=0, # Keep at 0 in Colab
)

print("Training arguments configured!")

# ===========================================
# Step 6: Trainer & Train
# ===========================================
print("\nCreating trainer...")

# Custom data collator from your notebook
# This is needed because you set the dataset format to 'torch'
def custom_data_collator(batch):
    batch_dict = {}
    for key in ['input_ids', 'attention_mask', 'labels']:
        if key in batch[0]:
            batch_dict[key] = torch.stack([item[key] for item in batch if key in item])
    return batch_dict

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=custom_data_collator,
)

print("\n" + "="*60)
print("=== STARTING TRAINING ===")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("=== TRAINING COMPLETE ===")
print("="*60)

# ===========================================
# Step 7: Generation & Evaluation
# ===========================================
print("\n" + "="*60)
print("=== TESTING MODEL ON SAMPLE PROMPTS ===")
print("="*60 + "\n")

model.eval()

def generate_prompt(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=120,
            num_beams=4,
            temperature=0.8,
            top_p=0.9,
            early_stopping=True,
            do_sample=True # do_sample=True was in your notebook
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Testing model on 5 random test samples:\n")
for i in range(min(5, len(test_data))):
    ex = test_data[i]
    result = generate_prompt(ex['input_text'])

    print(f"Sample {i+1}:")
    print(f"INPUT:     {ex['input_text'][:100]}...")
    print(f"EXPECTED:  {ex['target_text'][:100]}...")
    print(f"GENERATED: {result[:100]}...")
    print("\n" + "-"*60 + "\n")

print("Evaluation complete!")

# ===========================================
# Step 8: Custom Inference
# ===========================================
print("\n" + "="*60)
print("=== CUSTOM INFERENCE ===")
print("="*60 + "\n")

custom_subject = "a beautiful portrait of a woman"
custom_input = f"enhance: {custom_subject}"

print(f"Custom Input: {custom_input}\n")

result = generate_prompt(custom_input)
print(f"Generated Output: {result}")

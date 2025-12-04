🚀 Project Overview

This project fine-tunes a T5-Base transformer model to generate enhanced text prompts based on simpler subjects.
It uses the Lexica Dataset from HuggingFace (vera365/lexica_dataset) containing subjects and their corresponding high-quality prompts.

This repository includes:

Dataset loading & preprocessing

Tokenization

Fine-tuning T5-Base using HuggingFace Trainer

Mixed-precision training (FP16)

Evaluation on test samples

Custom prompt generation

🧩 Objective

Given an input like:

enhance: a beautiful portrait of a woman


The model learns to generate an enhanced prompt such as:

A cinematic ultra-detailed portrait of a woman with soft lighting, realistic skin texture…


This is useful for:

Prompt engineering

AI art generation (Stable Diffusion, Midjourney, Lexica)

Creative writing and caption enhancement

🛠 Installation

Run this before usage (Colab recommended):

pip install -q datasets transformers accelerate sentencepiece

🔥 Features

✔ Fine-tunes T5-Base on 5000 samples
✔ Uses FP16 mixed precision (fast & memory-efficient)
✔ Clean preprocessing pipeline
✔ Custom data collator
✔ Automatic GPU detection
✔ Evaluation across test samples
✔ Easy custom inference

📂 Project Structure
├── train_t5.py / notebook code
├── README.md
└── results/          # Model checkpoints will be saved here

🧵 Workflow Summary

Below is a simple explanation of each block from your code.

1️⃣ Device Setup

Detects GPU (T4 in Colab):

device = "cuda" if torch.cuda.is_available() else "cpu"

2️⃣ Dataset Loading

Uses Lexica Dataset, selects 5000 samples:

dataset = load_dataset('vera365/lexica_dataset', split='train')
dataset = dataset.shuffle(seed=42).select(range(5000))


Removes unused metadata fields.

Each sample is converted into:

input_text → "enhance: {subject}"
target_text → prompt

3️⃣ Tokenization

Tokenizes both input & output (max length = 256):

tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)


-100 is used for padded labels.

4️⃣ Model Loading
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

5️⃣ Training Arguments

Configured for fast but stable training:

Batch size: 2

Gradient Accumulation: 8

LR: 3e-5

Epochs: 3

FP16 enabled

Saves best model based on eval loss

6️⃣ Training

Custom data collator ensures proper tensor batching.

trainer = Trainer(...)
trainer.train()

7️⃣ Evaluation

The script generates predictions for 5 samples:

INPUT: enhance: a cyberpunk street...
EXPECTED: high-quality prompt from dataset
GENERATED: model output

8️⃣ Custom Inference

Example:

custom_subject = "a beautiful portrait of a woman"
generate_prompt(f"enhance: {custom_subject}")

🧪 How to Use the Model
Inference Example
text = "enhance: a futuristic robot warrior"
result = generate_prompt(text)
print(result)

📊 Training Tips

The more samples you use → better output

Increase epochs if GPU is strong

Try Low-Rank Adapters (LoRA) for improved speed

Use beam search for stable generation

⚠️ Common Issues & Fixes
Issue	Reason	Fix
GPU not detected	Wrong Colab runtime	Runtime → Change runtime type → GPU (T4)
CUDA OOM	Sequence too long	Reduce max_length to 128
Poor generation	Too few samples	Increase dataset size to 20k+
📎 Future Improvements

Add WandB training visualizations

Add T5-Large experiments

Add LoRA fine-tuning support

Deploy model with FastAPI

🏁 Conclusion

This project provides a complete pipeline for fine-tuning a T5 model to enhance text prompts using real-world Lexica data.
The workflow is optimized for both beginners and advanced ML users.

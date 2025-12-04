# 🎨 T5 Prompt Optimizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A fine-tuned **T5-Base** transformer model designed to act as an automated "Prompt Engineer." It takes simple, short subjects and upscales them into highly detailed, descriptive prompts suitable for AI art generators like **Stable Diffusion**, **Midjourney**, or **DALL-E**.

---

## 🚀 Project Overview

Generating high-quality images requires complex, detailed prompts. This project solves the "writer's block" of prompt engineering by training a model on real-world data.

- **Model:** Google's T5-Base (Text-to-Text Transfer Transformer).
- **Dataset:** `vera365/lexica_dataset` (Stable Diffusion prompts).
- **Task:** Text Generation / Prompt Enhancement.
- **Training Environment:** Optimized for Google Colab (T4 GPU) using Mixed Precision (FP16).

---

## 🧩 How It Works

The model is trained to map a simple input string to a complex output string:

| Input (User) | Output (Model Generation) |
| :--- | :--- |
| `enhance: a cyberpunk street` | `A futuristic cyberpunk street at night, neon lights reflecting on wet pavement, cinematic lighting, highly detailed, 8k resolution, unreal engine 5 render...` |
| `enhance: portrait of a woman` | `A beautiful portrait of a woman, soft studio lighting, intricate details, realistic skin texture, artstation, digital painting by greg rutkowski...` |

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/t5-prompt-optimizer.git](https://github.com/yourusername/t5-prompt-optimizer.git)
cd t5-prompt-optimizer
2. Install Dependencies
This project relies on the Hugging Face ecosystem and PyTorch.

Bash
pip install -q datasets transformers accelerate sentencepiece torch
💻 Usage
Training the Model
The script train.py (or your notebook) handles the entire pipeline: loading data, tokenizing, training, and saving the model.

Python
# To run the training script
python train.py
Note: The script automatically detects if a GPU is available. It is highly recommended to run this on a machine with a CUDA-enabled GPU (like Google Colab T4).

Inference (Generating Prompts)
Once trained, you can use the model to generate prompts. Here is a minimal example:

Python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load your fine-tuned model
model_path = "./results/checkpoint-final" # Update with your path
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")

def enhance_prompt(text):
    input_ids = tokenizer(f"enhance: {text}", return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=256, num_beams=4, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(enhance_prompt("a magical forest"))
⚙️ Training Configuration
The model was fine-tuned with the following hyperparameters to balance speed and quality on limited hardware:
Parameter,Value,Reason
Batch Size,2,Fits in T4 GPU VRAM
Gradient Accumulation,8,Simulates a batch size of 16
Learning Rate,3e-5,Standard for fine-tuning T5
Epochs,3,Prevents overfitting on small subsets
Precision,FP16,Reduces memory usage & speeds up training
Optimizer,AdamW,Standard optimizer for Transformers
📂 Project Structure
Bash
├── train.py           # Main training script (Data loading -> Training -> Eval)
├── results/           # Directory where model checkpoints are saved
├── README.md          # Project documentation
└── requirements.txt   # List of dependencies
📊 Dataset
We use the Lexica Dataset from Hugging Face.

Preprocessing: Removed metadata (width, height, seed) to focus purely on the text.

Filtering: Sampled 5,000 high-quality image-text pairs for training.

🤝 Contributing
Contributions are welcome! If you want to try different T5 sizes (Small/Large) or integrate Low-Rank Adaptation (LoRA), feel free to fork and submit a PR.

📜 License
This project is open-source and available under the MIT License.

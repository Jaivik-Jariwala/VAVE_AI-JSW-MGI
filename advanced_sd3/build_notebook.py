import json
import os

notebook_path = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\advanced_sd3\stable_diffusion_3_medium_diffusers.ipynb"

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "VAVE_SD3_FineTuning.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": []
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": os.urandom(6).hex()},
        "source": [line + "\n" if i < len(text.split("\n")) - 1 else line for i, line in enumerate(text.split("\n"))]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": os.urandom(6).hex()},
        "outputs": [],
        "source": [line + "\n" if i < len(text.split("\n")) - 1 else line for i, line in enumerate(text.split("\n"))]
    })


add_md("# VAVE AI: Simplified Stable Diffusion 3 Medium Training\nThis notebook handles dataset extraction, simplified LoRA fine-tuning, testing, and validation of custom automotive components.")

add_md("### 1. Setup Environment & HuggingFace Login\nEnsure you are using a GPU Runtime in Colab (T4, L4, or A100). Use your `HF_TOKEN` from HuggingFace to login.")
add_code("!pip install -q -U diffusers transformers accelerate peft huggingface_hub\n\nimport os\nfrom huggingface_hub import login\n\n# NOTE: Replace with your actual HuggingFace Read Token!\n# Make sure you accepted the License on the model page first.\nlogin(token=\"YOUR_HF_TOKEN_HERE\")")

add_md("### 2. Extract Dataset\nUpload your `dataset.zip` file using the folder icon on the left, then run this block to unzip it.")
add_code("!unzip -q dataset.zip -d dataset/")

add_md("### 3. Simplified SD3 LoRA Fine-Tuning\nWe use the official diffusers dreambooth script with minimal essential parameters.")
add_code("!git clone https://github.com/huggingface/diffusers.git\n%cd diffusers/examples/dreambooth\n\n# Install the specific training dependencies\n!pip install -r requirements_sd3.txt\n\n!accelerate launch train_dreambooth_lora_sd3.py \\\n  --pretrained_model_name_or_path=\"stabilityai/stable-diffusion-3-medium-diffusers\" \\\n  --dataset_name=\"/content/dataset\" \\\n  --instance_prompt=\"A realistic automotive engineering photo\" \\\n  --output_dir=\"/content/sd3-vave-lora\" \\\n  --mixed_precision=\"fp16\" \\\n  --resolution=1024 \\\n  --train_batch_size=1 \\\n  --gradient_accumulation_steps=1 \\\n  --learning_rate=1e-4 \\\n  --max_train_steps=500")

add_md("### 4. Download Weights to PC\nZip and download the resulting fine-tuned weights.")
add_code("!zip -r sd3-vave-lora.zip /content/sd3-vave-lora\n\nfrom google.colab import files\nfiles.download(\"sd3-vave-lora.zip\")")

add_md("### 5. Test Inference\nTest the generated LoRA weights visually.")
add_code("import torch\nfrom diffusers import StableDiffusion3Pipeline\nfrom IPython.display import display\n\npipeline = StableDiffusion3Pipeline.from_pretrained(\n    \"stabilityai/stable-diffusion-3-medium-diffusers\",\n    torch_dtype=torch.float16,\n    text_encoder_3=None,\n    tokenizer_3=None\n)\npipeline.load_lora_weights(\"/content/sd3-vave-lora\")\npipeline.to(\"cuda\")\n\nprompt = \"A realistic automotive engineering photo showing a Brake Caliper. Cost reduction idea: use cast aluminum.\"\nimage = pipeline(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]\nimage.save(\"/content/test_output.png\")\ndisplay(image)")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated Google Colab Notebook at {notebook_path}")

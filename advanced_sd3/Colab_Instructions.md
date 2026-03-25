# Colab Training Setup for Stable Diffusion 3 Medium LoRA

To avoid Windows environment and GPU VRAM issues, you can train your SD3 LoRA for the VAVE project on Google Colab (using an A100 or L4 GPU). 

Follow these exact steps by pasting the code below into Colab cells.

### Step 1: Upload Your Dataset
1. Compress your `advanced_sd3/dataset` folder into a `.zip` file on your PC (`dataset.zip`).
2. Open [Google Colab](https://colab.research.google.com/) and create a new Notebook.
3. In the menu, go to **Runtime > Change runtime type** and select a **T4 GPU** or higher (L4/A100 recommended for SD3).
4. On the left sidebar, click the **Folder icon** and upload the `dataset.zip` file.

### Step 2: Install Libraries & Unzip Dataset
Create a code cell and run this to install the required libraries and unpack your dataset:
```python
!pip install diffusers accelerate transformers peft bitsandbytes huggingface_hub
!unzip -q dataset.zip -d dataset/
```

### Step 3: Login to HuggingFace
Create a new code cell and run this to authenticate. 
*Note: Make sure you have agreed to the license at https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers first.*
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Step 4: Clone the Diffusers Repo
Create a new code cell and run this to download the official training script from HuggingFace:
```bash
!git clone https://github.com/huggingface/diffusers.git
%cd diffusers/examples/dreambooth
```

### Step 5: Start the LoRA Training (Dreambooth)
Create a new code cell and run this single command. It will download the base model, allocate the GPU using accelerate, and train your custom automotive weights!

```bash
!accelerate launch --num_processes=1 train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
  --dataset_name="/content/dataset" \
  --instance_prompt="A realistic automotive engineering photo" \
  --output_dir="/content/sd3-vave-lora" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed=42
```

### Step 6: Download Your Weights!
Once the training finishes (it will show a progress bar to 500 steps), the LoRA weights will be saved in `/content/sd3-vave-lora`. You can download that folder by zipping it up:
```bash
!zip -r sd3-vave-lora.zip /content/sd3-vave-lora
from google.colab import files
files.download("sd3-vave-lora.zip")
```

Once downloaded, you can use these files with the `test_sd3.py` or TensorRT `validate_sd3.py` scripts on your local machine!

import os
import subprocess
import sys
from huggingface_hub import login

def run_simplified_local_training():
    print("==================================================")
    print(" VAVE AI - SIMPLIFIED LOCAL SD3 LORA TRAINING     ")
    print("==================================================")
    
    # Check if user passed a token argument
    if len(sys.argv) < 2:
        print("\nERROR: You must provide your HuggingFace Token!")
        print("Usage: python run_local_sd3.py hf_your_token_here")
        print("Grab a token from https://huggingface.co/settings/tokens")
        sys.exit(1)
        
    hf_token = sys.argv[1]
    
    # Login
    print("\n1. Authenticating with HuggingFace...")
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)
        
    # Dataset and Output paths
    base_path = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\advanced_sd3"
    diffusers_script = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\diffusers\examples\dreambooth\train_dreambooth_lora_sd3.py"
    dataset_dir = os.path.join(base_path, "dataset")
    output_dir = os.path.join(base_path, "sd3-vave-lora")
    
    # Formulate simplified training command using direct Python executable
    # The user has multiple python environments, so we MUST force "python3" which holds the accelerate module
    command = [
        "python3", "-m", "accelerate.commands.launch", 
        "--num_processes=1",
        diffusers_script,
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers",
        f"--dataset_name={dataset_dir}",
        "--instance_prompt=\"A realistic automotive engineering photo\"",
        f"--output_dir={output_dir}",
        "--mixed_precision=fp16",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--learning_rate=1e-4",
        "--max_train_steps=500"
    ]
    
    print("\n2. Launching Simplified SD3 Dreambooth Training...")
    print("Command:", " ".join(command))
    
    # Execute the training without shell=True but passing the explicit list
    try:
        subprocess.run(command, check=True)
        print("\nSUCCESS! Training completed.")
        print(f"Your model weights are saved at: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining crashed with error code {e.returncode}")
        print("Ensure 'accelerate' is correctly installed and your GPU has sufficient VRAM.")

if __name__ == "__main__":
    run_simplified_local_training()

import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_image
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

# Add the MasaCtrl directory to path
masactrl_path = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\MasaCtrl"
sys.path.append(masactrl_path)

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl

def load_image_tensor(image_path, device):
    image = read_image(image_path)
    # read_image returns [C, H, W] in 0-255
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3]
    image = image.unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "runwayml/stable-diffusion-v1-5" # common base, SD 1.4 or 1.5
    
    # Initialize Pipeline
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    
    print("Loading MasaCtrlPipeline...")
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    
    dataset_dir = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\advanced_sd3\masactrl_dataset"
    metadata_file = os.path.join(dataset_dir, "metadata.jsonl")
    
    out_dir = os.path.join(dataset_dir, "output_masactrl")
    os.makedirs(out_dir, exist_ok=True)
    
    # parameters
    STEP = 4
    LAYER = 10
    
    # Process just the first 2 images as a test
    num_samples = 2
    count = 0
    
    seed = 42
    seed_everything(seed)
    
    print("Processing dataset...")
    
    with open(metadata_file, 'r') as f:
        for line in f:
            if count >= num_samples:
                break
                
            rec = json.loads(line)
            idea_id = rec['idea_id']
            # image_path from jsonl is something like 'images/train_img_0000.jpeg'
            # we extracted the zip directly into 'masactrl_dataset', so the path is 'masactrl_dataset/images/...'
            img_path = os.path.join(dataset_dir, rec['image_path'])
            target_prompt = rec['prompt']
            
            # Use empty string since we don't know the exact prompt describing the original image
            source_prompt = "" 
            prompts = [source_prompt, target_prompt]
            
            if not os.path.exists(img_path):
                print(f"Skipping {idea_id}, image not found: {img_path}")
                continue
                
            try:
                print(f"[{count+1}/{num_samples}] Inverting {idea_id} ({img_path})")
                source_image = load_image_tensor(img_path, device)
                
                # invert
                start_code, latents_list = model.invert(
                    source_image,
                    source_prompt,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    return_intermediates=True
                )
                start_code = start_code.expand(len(prompts), -1, -1, -1)
                
                # hijack attention
                editor = MutualSelfAttentionControl(STEP, LAYER)
                regiter_attention_editor_diffusers(model, editor)
                
                print(f"[{count+1}/{num_samples}] Generating edited output for: '{target_prompt}'")
                
                # synthesize
                image_masactrl = model(
                    prompts,
                    latents=start_code,
                    guidance_scale=7.5,
                    # ref_intermediate_latents=latents_list # enable if VRAM permits
                )
                
                # save
                out_path = os.path.join(out_dir, f"{idea_id}_masactrl.jpg")
                save_image(image_masactrl[-1:], out_path)
                print(f"[{count+1}/{num_samples}] Saved output to {out_path}")
                count += 1
                
            except Exception as e:
                print(f"Failed to process {idea_id}: {e}")

if __name__ == "__main__":
    main()

import torch
from diffusers import StableDiffusion3Pipeline

def test_sd3_lora(model_id, lora_weights_dir, prompt, output_image_path="test_output.png"):
    """
    Tests the basic PyTorch Diffusers pipeline for SD3 Medium with LoRA weights loaded.
    This does NOT use TensorRT (yet) - it's for immediate visual validation post-training.
    """
    print(f"Loading Base SD3 Model: {model_id}...")
    
    # Load base SD3 pipeline
    # Note: access to the stabilityai/stable-diffusion-3-medium-diffusers requires HF token
    try:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            text_encoder_3=None, # T5XXL takes massive VRAM, can optionally disable for speed/test
            tokenizer_3=None
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # Load custom LoRA weights
    print(f"Loading LoRA weights from {lora_weights_dir}...")
    try:
        pipeline.load_lora_weights(lora_weights_dir)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Note: If the directory is missing, the model will run without fine-tuning.")

    pipeline = pipeline.to("cuda")

    print(f"Generating image for prompt: '{prompt}'")
    
    # Generation
    image = pipeline(
        prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        height=1024,
        width=1024
    ).images[0]

    image.save(output_image_path)
    print(f"Saved successful generation to {output_image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test SD3 LoRA")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--lora", type=str, default="sd3-vave-lora")
    parser.add_argument("--prompt", type=str, default="A realistic automotive engineering photo showing a Brake Caliper. Cost reduction idea: use cast aluminum.")
    
    args = parser.parse_args()
    
    test_sd3_lora(args.model, args.lora, args.prompt)

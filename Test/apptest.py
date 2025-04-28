import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GPT2Tokenizer,GPT2LMHeadModel

def load_model(model_dir):
    """Load model and tokenizer from the given directory."""
    print(f"Loading model from {model_dir}...")
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(
        model_dir,
        device_map="auto",  # Use available GPU or CPU
    )
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text based on the prompt."""
    print(f"Generating text for prompt: '{prompt}'")
    
    # Encode the prompt and move tensors to device
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move each tensor individually to the model's device
    inputs = {
        k: v.to(model.device) 
        for k, v in inputs.items()
    }
    
    # Ensure input_ids are long integers (required for model)
    inputs["input_ids"] = inputs["input_ids"].long()
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Set the correct model directory path
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return
    
    # Check if config.json exists in the model directory
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Error: config.json not found in {model_dir}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_dir)
    
    print("\n===== MG MOTOR AI TEXT GENERATOR =====")
    print("Type 'exit' to quit the program")
    
    # Interactive loop
    while True:
        user_prompt = input("\nEnter your prompt: ")
        
        if user_prompt.lower() == 'exit':
            print("Exiting program...")
            break
        
        # Generate and display text
        try:
            generated_text = generate_text(model, tokenizer, user_prompt)
            print("\n--- Generated Text ---")
            print(generated_text)
            print("---------------------")
        except Exception as e:
            print(f"Generation error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
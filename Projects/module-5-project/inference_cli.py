#!/usr/bin/env python3
"""
CLI Inference Script for Fine-tuned TinyLlama Banking FAQ Model

Usage:
    python inference_cli.py "How do I activate my credit card?"
    python inference_cli.py --interactive
"""

import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = "./tinyllama-banking-finetuned"

def load_model():
    """Load the fine-tuned model with QLoRA quantization"""
    print("Loading fine-tuned model...")
    print("This may take a moment...\n")
    
    # Configure 4-bit quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully!\n")
    return model, tokenizer

def generate_response(model, tokenizer, query, max_new_tokens=256, temperature=0.3):
    """
    Generate response from the fine-tuned model
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        query: Customer query string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        Generated response string
    """
    # Format prompt for TinyLlama
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Get model device and move inputs there
    if hasattr(model, 'device'):
        model_device = next(model.parameters()).device
    else:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (remove prompt)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    elif "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

def interactive_mode(model, tokenizer):
    """Run in interactive mode (chat-like interface)"""
    print("=" * 80)
    print("Banking FAQ Chatbot - Interactive Mode")
    print("=" * 80)
    print("Enter your banking questions. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("Bot: ", end="", flush=True)
            response = generate_response(model, tokenizer, query)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="CLI Inference for Fine-tuned TinyLlama Banking FAQ Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_cli.py "How do I activate my credit card?"
  python inference_cli.py --interactive
  python inference_cli.py -i
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Customer query to get answer for'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (chat interface)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (default: 0.3, lower = more deterministic)'
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please make sure you have fine-tuned the model first.")
        sys.exit(1)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    if args.interactive or not args.query:
        interactive_mode(model, tokenizer)
    else:
        print("=" * 80)
        print("Banking FAQ Chatbot")
        print("=" * 80)
        print(f"\nQuery: {args.query}\n")
        print("Response:")
        print("-" * 80)
        response = generate_response(
            model, 
            tokenizer, 
            args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)
        print("=" * 80)

if __name__ == "__main__":
    main()



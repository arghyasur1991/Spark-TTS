import argparse
from transformers import AutoTokenizer
import os

def main():
    parser = argparse.ArgumentParser(description="Test Hugging Face Tokenizer")
    parser.add_argument("--text", type=str, default="Hello, world!", 
                        help="Text to tokenize")
    parser.add_argument("--tokenizer_path", type=str, 
                        default=os.path.join("onnx_models", "LLM_onnx"),
                        help="Path to the tokenizer directory (containing tokenizer.json)")
    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    
    # Check if the path exists
    if not os.path.isdir(args.tokenizer_path):
        print(f"Error: Tokenizer path does not exist: {args.tokenizer_path}")
        # Try to provide a more specific path if in the script's directory context
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, args.tokenizer_path)
        if os.path.isdir(alt_path):
            print(f"Attempting alternative path: {alt_path}")
            args.tokenizer_path = alt_path
        else:
            print(f"Alternative path also not found: {alt_path}")
            return

    try:
        # Load the tokenizer
        # `trust_remote_code=True` might be needed if the tokenizer uses custom code.
        # `use_fast=True` is generally recommended if available.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Attempting to load without use_fast=True...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
            print("Tokenizer loaded successfully (without use_fast=True).")
        except Exception as e_slow:
            print(f"Error loading tokenizer (without use_fast=True either): {e_slow}")
            return

    print(f"\nTokenizing text: '{args.text}'")

    # Tokenize the text
    # `return_tensors="pt"` returns PyTorch tensors. We just need lists for comparison.
    # `add_special_tokens=True` is default, but being explicit.
    encoding = tokenizer(args.text, add_special_tokens=True, return_attention_mask=True)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print(f"Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")

    # You can also decode to see how the tokenizer breaks down the string
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"Decoded Tokens: {decoded_tokens}")
    
    # Print special tokens for reference
    print(f"\nSpecial tokens map:")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"  CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  MASK token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# Copyright (c) 2025 SparkAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
import torch
import os

# Ensure optimum is installed: pip install optimum[onnxruntime] or optimum[onnxruntime-gpu]
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    # Use the main_export function for more direct control
    from optimum.exporters.onnx import main_export
    from optimum.exporters.onnx.model_configs import LlamaOnnxConfig # Assuming Qwen2 might use a Llama-like config or has its own
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print("Please install Hugging Face Optimum: pip install optimum[onnxruntime]")
    print("or for GPU: pip install optimum[onnxruntime-gpu]")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Export SparkTTS Main LLM to ONNX.")
    parser.add_argument(
        "--hf_model_path", 
        type=str, 
        required=True,
        help="Path to the Hugging Face LLM model directory (e.g., ./pretrained_models/Spark-TTS-0.5B/LLM)."
    )
    parser.add_argument(
        "--output_onnx_dir", 
        type=str, 
        required=True, 
        help="Directory to save the exported LLM ONNX model and associated files (e.g., ./onnx_models/LLM_onnx)."
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.") # Renamed from opset_version for main_export
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export the model in FP16 precision (requires GPU for export and inference typically).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation-with-past", # task for main_export, implies use_cache
        help="The task for which the model is exported. Use 'text-generation-with-past' for KV caching."
    )

    args = parser.parse_args()
    
    if args.fp16 and args.device == "cpu":
        print("Warning: FP16 export is selected but device is CPU. This might not be optimal or supported for all operations.")

    print(f"Exporting LLM from: {args.hf_model_path}")
    print(f"Output directory for ONNX LLM: {args.output_onnx_dir}")
    print(f"Task: {args.task}")
    print(f"Opset: {args.opset}")
    print(f"Device: {args.device}")
    print(f"FP16: {args.fp16}")

    output_path = Path(args.output_onnx_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nAttempting export using optimum.exporters.onnx.main_export")
        
        main_export(
            model_name_or_path=args.hf_model_path,
            output=Path(args.output_onnx_dir),
            task=args.task, 
            opset=args.opset,
            device=args.device,
            fp16=args.fp16,
            # trust_remote_code=True, 
        )

        print(f"ONNX LLM export attempt with main_export completed. Check {args.output_onnx_dir}")

        # --- Verification Step --- #
        print("\n--- Verifying exported ONNX LLM (after main_export) ---")
        
        # 1. Load original PyTorch model and tokenizer
        print(f"Loading original PyTorch model from: {args.hf_model_path}")
        pt_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
        print(f"Tokenizer EOS token ID: {pt_tokenizer.eos_token_id}")
        pt_model = AutoModelForCausalLM.from_pretrained(args.hf_model_path)
        pt_model.eval()
        if args.device != "cpu":
            pt_model.to(args.device)

        # Ensure tokenizer is also saved in the ONNX model directory for easy loading by ORTModelForCausalLM
        pt_tokenizer.save_pretrained(args.output_onnx_dir)

        # 2. Load ONNX model and its tokenizer (which should be the same as pt_tokenizer now)
        if args.device == "cuda:0" and not torch.cuda.is_available():
            print("Verification step requesting CUDA but not available, testing ONNX on CPU.")
            verify_device_onnx = "cpu"
        else:
            verify_device_onnx = args.device
        
        print(f"Loading ONNX LLM from {args.output_onnx_dir} for verification on device: {verify_device_onnx}")
        onnx_llm = ORTModelForCausalLM.from_pretrained(args.output_onnx_dir) 
        if verify_device_onnx != "cpu":
             print(f"ONNX model loaded. Note: Device placement for ORTModelForCausalLM is usually via execution providers.")

        test_prompt = "Once upon a time, in a land far away,"
        max_new_tokens_test = 30
        print(f"Test prompt: '{test_prompt}', max_new_tokens: {max_new_tokens_test}")
        
        # Tokenize for PyTorch model
        pt_inputs = pt_tokenizer(test_prompt, return_tensors="pt")
        if args.device != "cpu":
            pt_inputs = {k: v.to(args.device) for k, v in pt_inputs.items()}

        # Generate with PyTorch model
        print("\nGenerating with PyTorch model...")
        generated_ids_pt = pt_model.generate(
            **pt_inputs, 
            max_new_tokens=max_new_tokens_test, 
            pad_token_id=pt_tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.7,
            top_k=50
        )
        generated_text_pt = pt_tokenizer.batch_decode(generated_ids_pt, skip_special_tokens=True)
        print(f"PyTorch generated token IDs: {generated_ids_pt}")
        print(f"PyTorch generated text: {generated_text_pt}")

        # Tokenize for ONNX model (using the same tokenizer instance)
        onnx_inputs = pt_tokenizer(test_prompt, return_tensors="pt") 

        # Generate with ONNX model
        print("\nGenerating with ONNX model...")
        generated_ids_onnx = onnx_llm.generate(
            **onnx_inputs, 
            max_new_tokens=max_new_tokens_test, 
            pad_token_id=pt_tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.7,
            top_k=50
        )
        generated_text_onnx = pt_tokenizer.batch_decode(generated_ids_onnx, skip_special_tokens=True)
        print(f"ONNX LLM generated token IDs: {generated_ids_onnx}")
        print(f"ONNX LLM generated text: {generated_text_onnx}")
        
        print("\nONNX LLM verification attempt complete.")

    except Exception as e:
        print(f"An error occurred during LLM export or verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
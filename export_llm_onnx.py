"""
Exports a Hugging Face Transformer LLM to ONNX using Hugging Face Optimum.

This script utilizes `optimum.exporters.onnx.main_export` to convert a 
Hugging Face LLM (e.g., from the SparkTTS model) into ONNX format. 
It supports features like FP16 precision and KV caching (via the 'text-generation-with-past' task).

After export, a verification step is performed:
1. Load the original PyTorch model and the exported ONNX model.
2. Generate text from a sample prompt using both models.
3. Print the outputs for a qualitative comparison.
   (Exact match is not guaranteed if sampling is used in generation).
"""

import argparse
from pathlib import Path
import torch # For device checks and PyTorch model loading
import traceback

# Hugging Face Optimum and Transformers imports
try:
    from optimum.exporters.onnx import main_export
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("[ERROR] Hugging Face Optimum or Transformers not found. Please install them:")
    print("        pip install optimum[onnxruntime] transformers")
    print("        For GPU support: pip install optimum[onnxruntime-gpu] transformers")
    exit(1)

def export_llm_to_onnx(
    hf_model_path: Path,
    output_onnx_dir: Path,
    opset: int,
    device: str,
    fp16: bool,
    task: str,
    trust_remote_code: bool = False, # Added for flexibility with custom models
):
    """
    Exports a Hugging Face LLM to ONNX and verifies its basic functionality.

    Args:
        hf_model_path: Path to the Hugging Face LLM model directory.
        output_onnx_dir: Directory to save the exported ONNX model and associated files.
        opset: ONNX opset version.
        device: Device for export (e.g., 'cpu', 'cuda:0'). 
                Note: Optimum might handle device placement internally for export.
        fp16: Whether to export the model in FP16 precision.
        task: The task for Optimum export (e.g., 'text-generation-with-past' for KV cache).
        trust_remote_code: Whether to trust remote code when loading models/configs.
    """
    print(f"[INFO] Starting LLM ONNX export process...")
    print(f"[INFO]   Hugging Face model path: {hf_model_path}")
    print(f"[INFO]   Output ONNX directory: {output_onnx_dir}")
    print(f"[INFO]   Task: {task}")
    print(f"[INFO]   Opset: {opset}")
    print(f"[INFO]   Device for export: {device}")
    print(f"[INFO]   FP16 export: {fp16}")
    print(f"[INFO]   Trust remote code: {trust_remote_code}")

    if fp16 and device == "cpu":
        print("[WARNING] FP16 export is selected, but the export device is CPU. This might not be optimal or fully supported.")

    output_onnx_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] Attempting LLM export using optimum.exporters.onnx.main_export...")
        main_export(
            model_name_or_path=str(hf_model_path),
            output=output_onnx_dir,
            task=task,
            opset=opset,
            device=device,
            fp16=fp16,
            trust_remote_code=trust_remote_code,
            # Other potential args: no_post_process, optimize, etc. if needed
        )
        print(f"[INFO] ONNX LLM export attempt with main_export completed. Files should be in {output_onnx_dir}")

    except Exception as e:
        print(f"[ERROR] An error occurred during LLM export via optimum.exporters.onnx.main_export: {e}")
        traceback.print_exc()
        return # Stop if export fails

    # --- Verification Step --- #
    print("\n[INFO] --- Starting ONNX LLM Verification (Qualitative) ---")
    try:
        # 1. Load original PyTorch model and tokenizer
        print(f"[INFO] Loading original PyTorch model and tokenizer from: {hf_model_path}")
        pt_tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
        # Save the tokenizer to the output ONNX directory so ORTModelForCausalLM can find it easily.
        # main_export usually does this, but good to ensure.
        pt_tokenizer.save_pretrained(output_onnx_dir)
        print(f"[INFO] Tokenizer (EOS token ID: {pt_tokenizer.eos_token_id}) saved to {output_onnx_dir}")

        # Determine device for PyTorch model verification
        pt_device_verify = device
        if pt_device_verify == "cuda:0" and not torch.cuda.is_available():
            print("[WARNING] Requested CUDA for PyTorch verification, but not available. Using CPU.")
            pt_device_verify = "cpu"
        
        pt_model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
        pt_model.eval().to(pt_device_verify)
        print(f"[INFO] PyTorch model loaded to {pt_device_verify} for verification.")

        # 2. Load ONNX model using ORTModelForCausalLM
        # ORTModelForCausalLM will use the tokenizer from the same directory.
        # Device for ONNX model is typically handled by ONNX Runtime providers.
        print(f"[INFO] Loading ONNX LLM from {output_onnx_dir} for verification.")
        # For ORTModel, device placement is usually via providers when creating the InferenceSession.
        # If `device` was 'cuda:0', ORTModelForCausalLM might try to use CUDAExecutionProvider.
        onnx_llm = ORTModelForCausalLM.from_pretrained(output_onnx_dir) 
        print(f"[INFO] ONNX LLM loaded. It will use available providers (e.g., CPUExecutionProvider or CUDAExecutionProvider if configured).")

        # 3. Prepare sample prompt and generation parameters
        test_prompt = "Once upon a time, in a land far away,"
        max_new_tokens_test = 30
        # Common generation parameters - keep them same for both models for fairer comparison
        generation_params = {
            "max_new_tokens": max_new_tokens_test,
            "pad_token_id": pt_tokenizer.eos_token_id, # Ensure pad_token_id is set
            "eos_token_id": pt_tokenizer.eos_token_id,
            "do_sample": False, # Use greedy decoding for more deterministic output for verification
            # "temperature": 0.7, # Only relevant if do_sample=True
            # "top_k": 50,        # Only relevant if do_sample=True
        }
        print(f"[INFO] Test prompt: '{test_prompt}', Generation parameters: {generation_params}")
        
        # 4. Generate with PyTorch model
        print("\n[INFO] Generating with PyTorch model...")
        pt_inputs = pt_tokenizer(test_prompt, return_tensors="pt").to(pt_device_verify)
        generated_ids_pt = pt_model.generate(**pt_inputs, **generation_params)
        generated_text_pt = pt_tokenizer.batch_decode(generated_ids_pt, skip_special_tokens=True)
        print(f"[INFO] PyTorch generated token IDs: {generated_ids_pt[0].tolist()}") # Show first sequence
        print(f"[INFO] PyTorch generated text: {generated_text_pt}")

        # 5. Generate with ONNX model
        print("\n[INFO] Generating with ONNX model...")
        # ORTModelForCausalLM expects inputs on CPU if it manages its own device placement via providers.
        onnx_inputs = pt_tokenizer(test_prompt, return_tensors="pt") # Keep on CPU
        generated_ids_onnx = onnx_llm.generate(**onnx_inputs, **generation_params)
        generated_text_onnx = pt_tokenizer.batch_decode(generated_ids_onnx, skip_special_tokens=True)
        print(f"[INFO] ONNX LLM generated token IDs: {generated_ids_onnx[0].tolist()}") # Show first sequence
        print(f"[INFO] ONNX LLM generated text: {generated_text_onnx}")
        
        # Basic check: Did both produce output?
        if generated_text_pt and generated_text_onnx:
            print("\n[INFO] Both PyTorch and ONNX models produced output. Verification primarily qualitative.")
            # For greedy search, outputs should ideally be very close or identical if all ops are mapped perfectly.
            if generated_ids_pt[0].tolist() == generated_ids_onnx[0].tolist():
                print("[INFO] Generated token IDs from PyTorch and ONNX are IDENTICAL.")
            else:
                print("[WARNING] Generated token IDs from PyTorch and ONNX DIFFER. This can happen due to minor numerical differences or op variations.")
        else:
            print("[ERROR] One or both models failed to generate text.")

        print("\n[INFO] ONNX LLM verification attempt complete.")

    except Exception as e:
        print(f"[ERROR] An error occurred during LLM verification: {e}")
        traceback.print_exc()
    
    print(f"[INFO] LLM ONNX export and verification process complete. Models are in: {output_onnx_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face Transformer LLM to ONNX using Optimum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hf_model_path", 
        type=Path, 
        required=True,
        help="Path to the Hugging Face LLM model directory (e.g., ./pretrained_models/Spark-TTS-0.5B/LLM)."
    )
    parser.add_argument(
        "--output_onnx_dir", 
        type=Path, 
        required=True, 
        help="Directory to save the exported LLM ONNX model and associated files (e.g., ./onnx_models_exported/SparkTTS/LLM)."
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version for the export.")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device hint for Optimum export (e.g., 'cpu', 'cuda:0'). Optimum handles actual placement."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export the model in FP16 precision. Typically requires GPU for efficient export and inference.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation-with-past",
        help="The task for which the model is exported via Optimum. 'text-generation-with-past' enables KV caching and is recommended for inference."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set this flag if the Hugging Face model requires trusting remote code for loading (e.g., custom architectures)."
    )

    args = parser.parse_args()
    
    export_llm_to_onnx(
        hf_model_path=args.hf_model_path,
        output_onnx_dir=args.output_onnx_dir,
        opset=args.opset,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        trust_remote_code=args.trust_remote_code
    )

if __name__ == "__main__":
    main() 
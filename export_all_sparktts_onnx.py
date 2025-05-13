import os
import subprocess
import argparse
import shutil
import sys
from pathlib import Path

def run_export_script(script_name: str, script_args: list[str], verbose: bool = False):
    """Runs a given export script with the provided arguments."""
    command = [sys.executable, script_name] + script_args
    print(f"\n[INFO] Executing: {' '.join(command)}")
    try:
        # Using text=True for Python 3.7+ for automatic decoding of stdout/stderr
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"[INFO] Successfully executed {script_name}.")
        if verbose:
            if process.stdout:
                print(f"  [VERBOSE] stdout from {script_name}:\n{process.stdout.strip()}")
            if process.stderr:
                print(f"  [VERBOSE] stderr from {script_name}:\n{process.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error executing {script_name}:")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  stdout:\n{e.stdout}")
        if e.stderr:
            print(f"  stderr:\n{e.stderr}")
        raise # Re-raise the exception to stop the main script
    except FileNotFoundError:
        print(f"[ERROR] Script {script_name} not found. Make sure it's in the correct path and sys.executable is correct.")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate SparkTTS ONNX model exports for all components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_model_dir",
        type=Path,
        default=Path("./pretrained_models/Spark-TTS-0.5B"),
        help="Path to the base SparkTTS model directory (e.g., ./pretrained_models/Spark-TTS-0.5B)."
    )
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        default=Path("./onnx_models_exported"), 
        help="Base directory for the final ONNX model structure (e.g., ./onnx_models_exported)."
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="Global ONNX opset version for export scripts. Some scripts might have their own defaults or specific needs if this is not suitable."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0"], # Common choices
        help="Device for export processes where applicable (e.g., 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        "--fp16_llm",
        action="store_true",
        help="Export the Main LLM component in FP16 precision."
    )
    parser.add_argument(
        "--sample_audio_for_vocoder",
        type=str, # Kept as string, sub-script handles Path conversion
        default="./example/prompt_audio.wav",
        help="Path to a sample WAV file to derive token shapes for BiCodec Vocoder dummy inputs."
    )
    parser.add_argument(
        "--trust_remote_code_llm",
        action="store_true",
        help="Allow trusting remote code for the LLM export, if required by the Hugging Face model."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout and stderr from successful sub-script executions."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output_sparktts_dir (onnx_models_exported/SparkTTS by default) before starting exports."
    )

    args = parser.parse_args()

    # Define output paths using pathlib
    final_output_root = args.output_base_dir
    output_sparktts_dir = final_output_root / "SparkTTS"
    output_llm_dir = output_sparktts_dir / "LLM"

    # --- 0. Clean output directory if requested ---
    if args.clean:
        if output_sparktts_dir.exists():
            print(f"[INFO] --clean flag detected. Removing directory: {output_sparktts_dir}")
            try:
                shutil.rmtree(output_sparktts_dir)
                print(f"[INFO] Successfully removed {output_sparktts_dir}")
            except OSError as e:
                print(f"[ERROR] Failed to remove {output_sparktts_dir}: {e}")
                sys.exit(1) # Exit if cleaning fails as it might lead to unexpected results
        else:
            print(f"[INFO] --clean flag detected, but directory {output_sparktts_dir} does not exist. Nothing to remove.")

    # Create directories if they don't exist
    output_llm_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured output directory structure exists at: {output_sparktts_dir}")

    common_export_args = {
        "opset_version": str(args.opset_version),
        "device": args.device
    }

    # --- 1. Export Wav2Vec2 ---
    # Note: Refactored export_wav2vec2_onnx.py does not take a --device argument.
    # It performs CPU-based export internally.
    wav2vec2_hf_path = str(args.base_model_dir / "wav2vec2-large-xlsr-53")
    wav2vec2_output_path = str(output_sparktts_dir / "wav2vec2_model.onnx")
    run_export_script("export_wav2vec2_onnx.py", [
        "--hf_model_path", wav2vec2_hf_path,
        "--output_path", wav2vec2_output_path,
        "--opset_version", str(args.opset_version)
    ], verbose=args.verbose)

    # --- 2. Export Mel Spectrogram ---
    mel_spec_output_path = str(output_sparktts_dir / "mel_spectrogram.onnx")
    run_export_script("export_mel_spectrogram_onnx.py", [
        "--model_dir", str(args.base_model_dir), # Used for BiCodec config for mel_params
        "--output_path", mel_spec_output_path,
        "--opset_version", common_export_args["opset_version"],
        "--device", common_export_args["device"]
    ], verbose=args.verbose)

    # --- 3. Export BiCodec Encoder + Quantizer ---
    # This script's internal default opset is 17 if not specified, but here we pass the global one.
    bicodec_enc_quant_output_path = str(output_sparktts_dir / "bicodec_encoder_quantizer.onnx")
    run_export_script("export_bicodec_encoder_quantizer_onnx.py", [
        "--model_dir", str(args.base_model_dir),
        "--output_path", bicodec_enc_quant_output_path,
        "--opset_version", common_export_args["opset_version"],
        "--device", common_export_args["device"]
    ], verbose=args.verbose)

    # --- 4. Export Speaker Encoder Tokenizer ---
    speaker_enc_tok_output_path = str(output_sparktts_dir / "speaker_encoder_tokenizer.onnx")
    run_export_script("export_speaker_encoder_tokenizer_onnx.py", [
        "--model_dir", str(args.base_model_dir),
        "--output_path", speaker_enc_tok_output_path,
        "--opset_version", common_export_args["opset_version"],
        "--device", common_export_args["device"]
    ], verbose=args.verbose)

    # --- 5. Export BiCodec Vocoder ---
    bicodec_voc_output_path = str(output_sparktts_dir / "bicodec_vocoder.onnx")
    run_export_script("export_bicodec_vocoder_onnx.py", [
        "--model_dir", str(args.base_model_dir),
        "--output_path", bicodec_voc_output_path,
        "--sample_audio_for_shapes", args.sample_audio_for_vocoder,
        "--opset_version", common_export_args["opset_version"],
        "--device", common_export_args["device"]
    ], verbose=args.verbose)
    
    # --- 6. Export Main LLM ---
    llm_hf_path = str(args.base_model_dir / "LLM")
    llm_export_args_list = [
        "--hf_model_path", llm_hf_path,
        "--output_onnx_dir", str(output_llm_dir),
        "--opset", common_export_args["opset_version"], # Note: arg is --opset in export_llm_onnx.py
        "--device", common_export_args["device"],
        "--task", "text-generation-with-past" # Recommended for KV cache
    ]
    if args.fp16_llm:
        llm_export_args_list.append("--fp16")
    if args.trust_remote_code_llm:
        llm_export_args_list.append("--trust_remote_code")
    
    run_export_script("export_llm_onnx.py", llm_export_args_list, verbose=args.verbose)

    # --- 7. Cleanup LLM Directory (Optional: keep only essential files) ---
    print(f"\n[INFO] Cleaning up LLM ONNX directory: {output_llm_dir}")
    # This list should match what export_llm_onnx.py (via Optimum) typically produces as essential.
    # Common files include: model.onnx, model.onnx_data (if large), config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json
    # The exact set can vary with Optimum versions and model types. A more robust cleanup might inspect the ORTModelForCausalLM.from_pretrained loading needs.
    # For now, keeping it simple or making it less aggressive.
    # Comment out the files you don't want to keep.
    # Allowed files based on typical Optimum output for LLMs for inference:
    allowed_llm_files = [
        "model.onnx", 
        "model.onnx_data", 
        "tokenizer.json", 
        "tokenizer_config.json", 
        "special_tokens_map.json", 
        "config.json", # The LLM's own config.json
        "merges.txt", 
        "vocab.json", # For some tokenizers like GPT2/BPE
        "added_tokens.json",
        "generation_config.json"
    ]
    removed_count = 0
    kept_count = 0
    if output_llm_dir.exists():
        for item in output_llm_dir.iterdir():
            if item.name not in allowed_llm_files:
                print(f"[INFO] Removing from LLM dir: {item.name}")
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                removed_count +=1
            else:
                kept_count +=1
        print(f"[INFO] LLM directory cleanup complete. Kept {kept_count} essential files, removed {removed_count} other files/dirs.")
    else:
        print(f"[WARNING] LLM output directory {output_llm_dir} not found for cleanup.")


    print(f"\n[INFO] All export processes attempted. Models are in: {output_sparktts_dir}")
    print(f"[INFO] Please check the logs above for success or failure of each step.")

if __name__ == "__main__":
    main() 
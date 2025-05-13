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

import os
import subprocess
import argparse
import shutil
import sys
from pathlib import Path

def run_export_script(script_name: str, script_args: list[str]):
    """Runs a given export script with the provided arguments."""
    command = [sys.executable, script_name] + script_args
    print(f"\nExecuting: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully executed {script_name}.")
        if process.stdout:
            print(f"stdout:\n{process.stdout}")
        if process.stderr:
            print(f"stderr:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout:\n{e.stdout}")
        if e.stderr:
            print(f"stderr:\n{e.stderr}")
        raise # Re-raise the exception to stop the main script

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SparkTTS ONNX model exports.")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="./pretrained_models/Spark-TTS-0.5B",
        help="Path to the base SparkTTS model directory (e.g., ./pretrained_models/Spark-TTS-0.5B)."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./onnx_models_exported", # Changed from onnx_models to avoid conflict if scripts create their own onnx_models
        help="Base directory for the final ONNX model structure (e.g., ./onnx_models_exported)."
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="Global ONNX opset version for export scripts (some scripts may override this)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0"], # Added cuda for more general gpu usage
        help="Device for export (e.g., 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        "--fp16_llm",
        action="store_true",
        help="Export the Main LLM in FP16 precision."
    )
    parser.add_argument(
        "--sample_audio_for_vocoder",
        type=str,
        default="./example/prompt_audio.wav",
        help="Path to a sample WAV file to derive token shapes for BiCodec Vocoder dummy inputs."
    )

    args = parser.parse_args()

    # Define output paths
    final_output_root = Path(args.output_base_dir)
    output_sparktts_dir = final_output_root / "SparkTTS"
    output_llm_dir = output_sparktts_dir / "LLM"

    # Create directories
    output_llm_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory structure exists at: {output_sparktts_dir}")

    # --- 1. Export Wav2Vec2 ---
    wav2vec2_hf_path = str(Path(args.base_model_dir) / "wav2vec2-large-xlsr-53")
    wav2vec2_output_onnx = str(output_sparktts_dir / "wav2vec2_model.onnx")
    run_export_script("export_wav2vec2_onnx.py", [
        "--hf_model_path", wav2vec2_hf_path,
        "--output_onnx_path", wav2vec2_output_onnx,
        "--opset_version", str(args.opset_version)
    ])

    # --- 2. Export Mel Spectrogram ---
    mel_spec_output_onnx = str(output_sparktts_dir / "mel_spectrogram.onnx")
    run_export_script("export_mel_spectrogram_onnx.py", [
        "--model_dir", args.base_model_dir, # This script uses it to load BiCodec config for mel_params
        "--output_onnx_path", mel_spec_output_onnx,
        "--opset_version", str(args.opset_version),
        "--device", args.device
    ])

    # --- 3. Export BiCodec Encoder + Quantizer ---
    # Note: This script has opset 17 hardcoded.
    bicodec_enc_quant_output_onnx = str(output_sparktts_dir / "bicodec_encoder_quantizer.onnx")
    run_export_script("export_bicodec_encoder_quantizer_onnx.py", [
        "--model_dir", args.base_model_dir,
        "--output_name", bicodec_enc_quant_output_onnx
        # Opset is hardcoded in the script
    ])

    # --- 4. Export Speaker Encoder Tokenizer ---
    speaker_enc_tok_output_onnx = str(output_sparktts_dir / "speaker_encoder_tokenizer.onnx")
    run_export_script("export_speaker_encoder_tokenizer_onnx.py", [
        "--model_dir", args.base_model_dir,
        "--output_onnx_path", speaker_enc_tok_output_onnx,
        "--opset_version", str(args.opset_version),
        "--device", args.device
    ])

    # --- 5. Export BiCodec Vocoder ---
    bicodec_voc_output_onnx = str(output_sparktts_dir / "bicodec_vocoder.onnx")
    run_export_script("export_bicodec_vocoder_onnx.py", [
        "--model_dir", args.base_model_dir,
        "--output_onnx_path", bicodec_voc_output_onnx,
        "--sample_audio_for_shapes", args.sample_audio_for_vocoder,
        "--opset_version", str(args.opset_version),
        "--device", args.device
    ])
    
    # --- 6. Export Main LLM ---
    llm_hf_path = str(Path(args.base_model_dir) / "LLM")
    llm_export_args = [
        "--hf_model_path", llm_hf_path,
        "--output_onnx_dir", str(output_llm_dir),
        "--opset", str(args.opset_version), # Note: arg is --opset in export_llm_onnx.py
        "--device", args.device,
        "--task", "text-generation-with-past" # Recommended for KV cache
    ]
    if args.fp16_llm:
        llm_export_args.append("--fp16")
    
    run_export_script("export_llm_onnx.py", llm_export_args)

    # --- 7. Cleanup LLM Directory ---
    print(f"\nCleaning up LLM directory: {output_llm_dir}")
    allowed_llm_files = ["model.onnx", "model.onnx_data", "tokenizer.json"]
    for item in output_llm_dir.iterdir():
        if item.name not in allowed_llm_files:
            print(f"Removing: {item}")
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    print("LLM directory cleanup complete.")

    print(f"\nAll exports completed. Models are in: {output_sparktts_dir}")

if __name__ == "__main__":
    main() 
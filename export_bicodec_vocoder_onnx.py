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

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from pathlib import Path
import onnxruntime

# Assuming BiCodec and its submodules are accessible via sparktts.models
from sparktts.models.bicodec import BiCodec
from sparktts.models.audio_tokenizer import BiCodecTokenizer # For getting dummy token shapes
from sparktts.utils.file import load_config # If needed for BiCodec loading details

class BiCodecVocoderONNXWrapper(nn.Module):
    def __init__(self, bicodec_model: BiCodec):
        super().__init__()
        # Extract necessary components from the pre-trained BiCodec model
        self.quantizer = bicodec_model.quantizer
        self.speaker_encoder = bicodec_model.speaker_encoder
        self.prenet = bicodec_model.prenet
        self.decoder = bicodec_model.decoder # This is the WaveGenerator

    def forward(self, semantic_tokens: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        # Replicate the detokenization logic from BiCodec.detokenize
        # Note: MPS-specific handling is omitted here for ONNX export simplicity;
        #       ONNX models will run on CPU or other providers specified by onnxruntime.

        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens, onnx_export_mode=True)
        
        x_prenet = self.prenet(z_q, d_vector)
        
        # Ensure d_vector is compatible for broadcasting with x_prenet
        # d_vector is likely [B, D], x_prenet is [B, D, T] or [B, T, D]
        # If d_vector is [B, D] and speaker_encoder.detokenize outputs that,
        # and prenet output x_prenet is [B, D, T] (Channels first after prenet)
        # then d_vector.unsqueeze(-1) makes it [B, D, 1] which broadcasts with [B, D, T]
        # If prenet output is [B, T, D], then d_vector.unsqueeze(1) -> [B, 1, D] for broadcasting.        
        # Let's inspect shapes during dummy run to confirm this broadcasting.
        # For now, assuming prenet output is [B, C, T] and d_vector is [B, C]
        if d_vector.ndim == 2 and x_prenet.ndim == 3:
             # Assuming d_vector is [Batch, Channels] and x_prenet is [Batch, Channels, Time]
            if d_vector.shape[0] == x_prenet.shape[0] and d_vector.shape[1] == x_prenet.shape[1]:
                condition_vector = d_vector.unsqueeze(-1)
            else:
                # This case needs careful handling based on actual shapes from submodules
                raise ValueError(f"Shape mismatch for conditioning: d_vector {d_vector.shape}, x_prenet {x_prenet.shape}. Review unsqueeze operation.")
        elif d_vector.ndim == x_prenet.ndim: # If they are already compatible e.g. prenet also incorporates/expects unsqueezed d_vector
            condition_vector = d_vector
        else:
            raise ValueError(f"Unexpected dimensions for conditioning: d_vector {d_vector.ndim}D, x_prenet {x_prenet.ndim}D.")

        x_conditioned = x_prenet + condition_vector
        wav_recon = self.decoder(x_conditioned)
        return wav_recon

def get_dummy_tokens_from_tokenizer(model_base_dir: Path, device: torch.device, sample_audio_path: str):
    """Uses BiCodecTokenizer to generate real tokens from a sample audio file to get their shapes."""
    print(f"Initializing BiCodecTokenizer to get sample token shapes from: {model_base_dir}")
    # The BiCodecTokenizer expects the *parent* directory of BiCodec, wav2vec2, etc.
    # e.g., if BiCodec is in model_base_dir/BiCodec, pass model_base_dir
    tokenizer = BiCodecTokenizer(model_dir=model_base_dir, device=device)
    
    if not os.path.exists(sample_audio_path):
        # Create a dummy wav file if sample doesn't exist, as tokenizer needs a real path
        print(f"Sample audio {sample_audio_path} not found. Creating a dummy 1-sec silence.")
        import soundfile as sf
        dummy_wav_data = np.zeros(16000, dtype=np.float32) # 1 sec at 16kHz
        os.makedirs(os.path.dirname(sample_audio_path), exist_ok=True)
        sf.write(sample_audio_path, dummy_wav_data, 16000)
        print(f"Dummy audio created at {sample_audio_path}")

    print(f"Tokenizing sample audio: {sample_audio_path}")
    # BiCodecTokenizer.tokenize returns: global_tokens, semantic_tokens
    real_global_tokens, real_semantic_tokens = tokenizer.tokenize(sample_audio_path)
    print(f"Shape of real_global_tokens: {real_global_tokens.shape}, dtype: {real_global_tokens.dtype}")
    print(f"Shape of real_semantic_tokens: {real_semantic_tokens.shape}, dtype: {real_semantic_tokens.dtype}")

    # Create random dummy integer tensors with the same shapes and type
    # Assuming token IDs are non-negative. Using a small upper bound for dummy values.
    dummy_global_tokens = torch.randint_like(real_global_tokens, low=0, high=100) # dtype will be inferred from real_global_tokens
    dummy_semantic_tokens = torch.randint_like(real_semantic_tokens, low=0, high=100) # dtype will be inferred
    
    print(f"Shape of dummy_global_tokens: {dummy_global_tokens.shape}, dtype: {dummy_global_tokens.dtype}")
    print(f"Shape of dummy_semantic_tokens: {dummy_semantic_tokens.shape}, dtype: {dummy_semantic_tokens.dtype}")


    return dummy_semantic_tokens, dummy_global_tokens

def main():
    parser = argparse.ArgumentParser(description="Export SparkTTS BiCodec Vocoder to ONNX.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B). Should contain BiCodec subdirectory."
    )
    parser.add_argument(
        "--output_onnx_path", 
        type=str, 
        required=True, 
        help="Full path to save the exported Vocoder ONNX model (e.g., ./onnx_models/bicodec_vocoder.onnx)."
    )
    parser.add_argument("--sample_audio_for_shapes", type=str, default="example/prompt_audio.wav",
                        help="Path to a sample WAV file to derive token shapes for dummy inputs.")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    # Add device argument if necessary, default to CPU for export script
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")

    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load the pre-trained BiCodec model
    bicodec_model_dir = Path(args.model_dir) / "BiCodec"
    print(f"Loading BiCodec model from: {bicodec_model_dir}")
    if not bicodec_model_dir.exists():
        print(f"ERROR: BiCodec model directory not found at {bicodec_model_dir}")
        return
    
    # BiCodec.load_from_checkpoint expects the device to be passed if not CPU.
    # It also handles MPS internally by moving to CPU for certain ops.
    # For ONNX export, we usually want the model on CPU or the target export device.
    bicodec_model = BiCodec.load_from_checkpoint(model_dir=bicodec_model_dir, device=device).to(device)
    bicodec_model.eval()
    print("BiCodec model loaded and set to eval mode.")

    # 2. Instantiate the ONNX wrapper
    onnx_exportable_vocoder = BiCodecVocoderONNXWrapper(bicodec_model).to(device)
    onnx_exportable_vocoder.eval()

    # 3. Prepare dummy inputs (semantic_tokens, global_tokens)
    # Use the actual model_dir for BiCodecTokenizer, which is args.model_dir (parent of BiCodec/)
    dummy_semantic_tokens, dummy_global_tokens = get_dummy_tokens_from_tokenizer(
        model_base_dir=Path(args.model_dir), 
        device=device, 
        sample_audio_path=args.sample_audio_for_shapes
    )
    dummy_semantic_tokens = dummy_semantic_tokens.to(device)
    dummy_global_tokens = dummy_global_tokens.to(device)

    print(f"Using dummy_semantic_tokens shape: {dummy_semantic_tokens.shape}")
    print(f"Using dummy_global_tokens shape: {dummy_global_tokens.shape}")

    # Before export, run a forward pass with the wrapper to check for internal shape issues
    print("Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            # Unsqueeze batch dim if tokenizer provided single instance
            # The shapes printed from get_dummy_tokens_from_tokenizer are already batch_first for single audio
            # e.g., global_tokens: torch.Size([1, 1, 32]), semantic_tokens: torch.Size([1, 497])
            # The wrapper expects semantic_tokens [B, Nq, T_token] and global_tokens [B, D_global]
            # However, BiCodec.detokenize expects global_tokens: [B, Nq_global, T_global_token_ids]
            # and semantic_tokens: [B, Nq_semantic, T_semantic_token_ids]
            # The quantizer.detokenize and speaker_encoder.detokenize handle these.

            # Based on output: Shape of real_global_tokens: torch.Size([1, 1, 32])
            # This is likely [Batch, NumQuantizerLevelsForGlobal, NumTokensPerLevel]
            # Based on output: Shape of real_semantic_tokens: torch.Size([1, 497])
            # This is likely [Batch, NumTokens] (flattened over quantizer levels if multiple, or single level)
            # Or it could be [Batch, NumQuantizers, SeqLen]
            
            # For the BiCodecVocoderONNXWrapper, the inputs `semantic_tokens` and `global_tokens` are directly
            # passed to the respective `detokenize` methods of the quantizer and speaker_encoder.
            # These methods expect the raw token ID shapes.
            # So, the dummy tokens should match these raw shapes. `unsqueeze` is not needed here if tokenizer gives batch_size 1.
            dummy_semantic_tokens_batch = dummy_semantic_tokens
            dummy_global_tokens_batch = dummy_global_tokens

            # Ensure batch sizes match for the wrapper's inputs if they are not already the same
            # (Though typically tokenizer should provide matching batch sizes)
            if dummy_semantic_tokens_batch.shape[0] != dummy_global_tokens_batch.shape[0]:
                print("CRITICAL WARNING: Batch size mismatch between semantic and global tokens before wrapper call.")
                # This should ideally not happen if get_dummy_tokens_from_tokenizer is correct.
                # Handle cautiously, e.g. by taking min batch size or erroring, but for now let it proceed to see error loc.
            
            pytorch_output_waveform = onnx_exportable_vocoder(dummy_semantic_tokens_batch, dummy_global_tokens_batch)
        print(f"PyTorch ONNX wrapper test forward pass successful. Output waveform shape: {pytorch_output_waveform.shape}")
    except Exception as e:
        print(f"Error during PyTorch ONNX wrapper test forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Export to ONNX
    input_names = ["semantic_tokens", "global_tokens"]
    output_names = ["output_waveform"]
    
    dynamic_axes = {}
    # Based on dummy shapes: 
    # real_global_tokens: torch.Size([1, 1, 32]) -> [B, N_quant_global, T_global]
    # real_semantic_tokens: torch.Size([1, 497]) -> [B, T_semantic_flat]
    # OR semantic_tokens: [B, N_quant_semantic, T_semantic]
    # Let's assume the shapes from get_dummy_tokens are already batched [1, ...]

    # For global_tokens: [B, N_quant_global, T_global]
    if dummy_global_tokens_batch.ndim == 3:
        dynamic_axes[input_names[1]] = {0: 'batch_size', 2: 'global_token_seq_len'}
    elif dummy_global_tokens_batch.ndim == 2: # E.g. [B, D_global_flat] - less likely if it has quantizer levels
        dynamic_axes[input_names[1]] = {0: 'batch_size', 1: 'global_feature_len'}
    else:
        print(f"Warning: Unexpected ndim for dummy_global_tokens_batch: {dummy_global_tokens_batch.ndim}")

    # For semantic_tokens: [B, T_semantic_flat] or [B, N_quant_semantic, T_semantic]
    if dummy_semantic_tokens_batch.ndim == 2: # e.g. [B, T_semantic_flat]
        dynamic_axes[input_names[0]] = {0: 'batch_size', 1: 'semantic_token_flat_seq_len'}
    elif dummy_semantic_tokens_batch.ndim == 3: # e.g. [B, N_quant_semantic, T_semantic]
        dynamic_axes[input_names[0]] = {0: 'batch_size', 2: 'semantic_token_seq_len'}
    else:
        print(f"Warning: Unexpected ndim for dummy_semantic_tokens_batch: {dummy_semantic_tokens_batch.ndim}")

    # Output waveform: e.g. [B, AudioSeqLen] or [B, 1, AudioSeqLen]
    if pytorch_output_waveform.ndim == 2: # [B, AudioSeqLen]
        dynamic_axes[output_names[0]] = {0: 'batch_size', 1: 'audio_seq_len'}
    elif pytorch_output_waveform.ndim == 3: # [B, 1, AudioSeqLen] or [B, C, AudioSeqLen]
        dynamic_axes[output_names[0]] = {0: 'batch_size', 2: 'audio_seq_len'}
        if pytorch_output_waveform.shape[1] != 1:
             print(f"Warning: Output waveform has {pytorch_output_waveform.shape[1]} channels. Dynamic axis set on length only.")

    print(f"Using dynamic axes: {dynamic_axes}")

    # Ensure output directory exists
    output_dir = Path(args.output_onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting BiCodec Vocoder to ONNX: {args.output_onnx_path} with opset {args.opset_version}")
    try:
        torch.onnx.export(
            onnx_exportable_vocoder,
            (dummy_semantic_tokens_batch, dummy_global_tokens_batch), # Pass inputs as a tuple
            args.output_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False # Set to True for detailed ONNX export logging
        )
        print("ONNX export complete.")
    except Exception as e:
        print(f"Error during torch.onnx.export: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Verify the ONNX Model (Optional but recommended)
    print("\n--- Starting ONNX Model Verification ---")
    try:
        print("Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(args.output_onnx_path, providers=["CPUExecutionProvider"]) # Or other providers
        onnx_input_names = [inp.name for inp in ort_session.get_inputs()]
        print(f"ONNX model loaded. Expected input names: {onnx_input_names}")

        ort_inputs = {
            onnx_input_names[0]: dummy_semantic_tokens_batch.cpu().numpy(),
            onnx_input_names[1]: dummy_global_tokens_batch.cpu().numpy()
        }

        print("Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_waveform_np = ort_outputs[0]
        print(f"ONNX Runtime inference successful. Output waveform shape: {onnx_output_waveform_np.shape}")

        # Compare with PyTorch output
        np.testing.assert_allclose(
            pytorch_output_waveform.cpu().detach().numpy(), 
            onnx_output_waveform_np, 
            rtol=1e-03, 
            atol=1e-05 # Start with tight tolerance, relax if needed
        )
        print("ONNX Runtime outputs match PyTorch outputs numerically (within tolerance).")

    except Exception as e:
        print(f"Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
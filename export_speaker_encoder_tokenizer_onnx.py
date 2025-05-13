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

from sparktts.models.bicodec import BiCodec # To access SpeakerEncoder
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder

class SpeakerEncoderTokenizerONNXWrapper(nn.Module):
    def __init__(self, speaker_encoder_model: SpeakerEncoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder_model

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        print(f"[Wrapper] Mel input shape: {mels.shape}") # Debug print
        # Call SpeakerEncoder.tokenize with onnx_export_mode=True
        # We'll need to ensure SpeakerEncoder.tokenize supports this flag
        # and passes it to its internal self.quantizer.forward() call.
        return self.speaker_encoder.tokenize(mels, onnx_export_mode=True)

def main():
    parser = argparse.ArgumentParser(description="Export SparkTTS Speaker Encoder Tokenizer to ONNX.")
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
        help="Full path to save the exported Speaker Encoder Tokenizer ONNX model (e.g., ./onnx_models/speaker_encoder_tokenizer.onnx)."
    )
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")
    # Dummy input shape parameters
    # ECAPA_TDNN expects (Batch, Time, Features) and transposes internally to (Batch, Features, Time)
    # The 'feat_dim' of ECAPA_TDNN (128 for this model) is the Features dimension.
    parser.add_argument("--dummy_mel_batch", type=int, default=1, help="Batch size for dummy Mel input.")
    parser.add_argument("--dummy_mel_time", type=int, default=200, help="Time steps for dummy Mel input (becomes middle dim).")
    parser.add_argument("--dummy_mel_channels", type=int, default=128, help="Number of Mel features/channels (becomes last dim, this is ECAPA's feat_dim).")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Load the pre-trained BiCodec model to get the SpeakerEncoder
    bicodec_model_dir = Path(args.model_dir) / "BiCodec"
    print(f"Loading BiCodec model from: {bicodec_model_dir} to access SpeakerEncoder")
    if not bicodec_model_dir.exists():
        print(f"ERROR: BiCodec model directory not found at {bicodec_model_dir}")
        return
    
    bicodec_model = BiCodec.load_from_checkpoint(model_dir=bicodec_model_dir, device=device).to(device)
    speaker_encoder_module = bicodec_model.speaker_encoder
    speaker_encoder_module.eval()
    print("SpeakerEncoder module extracted and set to eval mode.")

    # 2. Instantiate the ONNX wrapper
    onnx_exportable_tokenizer = SpeakerEncoderTokenizerONNXWrapper(speaker_encoder_module).to(device)
    onnx_exportable_tokenizer.eval()

    # 3. Prepare dummy Mel input
    # ECAPA_TDNN expects (B, T, F) and transposes it to (B, F, T) internally.
    # So, dummy_mels should be (Batch, Time, Features/Channels)
    dummy_mels = torch.randn(
        args.dummy_mel_batch,    # Batch
        args.dummy_mel_time,     # Time
        args.dummy_mel_channels, # Features/Channels (feat_dim for ECAPA)
        device=device
    ).contiguous()
    print(f"Using dummy_mels input shape (B, T, F): {dummy_mels.shape}")

    # Test forward pass with the PyTorch ONNX wrapper
    print("Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            pytorch_output_tokens = onnx_exportable_tokenizer(dummy_mels)
        print(f"PyTorch ONNX wrapper test forward pass successful. Output tokens shape: {pytorch_output_tokens.shape}")
    except Exception as e:
        print(f"Error during PyTorch ONNX wrapper test forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Export to ONNX
    input_names = ["mel_spectrogram"]
    output_names = ["global_tokens"]
    
    dynamic_axes = {
        # Input is (Batch, Time, Features)
        input_names[0]: {0: 'batch_size', 1: 'mel_time_steps', 2: 'mel_channels'},
        output_names[0]: {0: 'batch_size'} 
    }
    # Example output: indices shape torch.Size([1, 1, 32]) -> [B, Nq, T_tokens]
    # So, dynamic on batch_size and token_sequence_length (dim 2)
    if pytorch_output_tokens.ndim == 3:
        dynamic_axes[output_names[0]][2] = 'token_seq_len'
    elif pytorch_output_tokens.ndim == 2: # If [B, T_tokens_flat]
        dynamic_axes[output_names[0]][1] = 'token_seq_len_flat'


    print(f"Using dynamic axes: {dynamic_axes}")

    output_dir = Path(args.output_onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting Speaker Encoder Tokenizer to ONNX: {args.output_onnx_path} with opset {args.opset_version}")
    try:
        torch.onnx.export(
            onnx_exportable_tokenizer,
            dummy_mels,
            args.output_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
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
        ort_session = onnxruntime.InferenceSession(args.output_onnx_path, providers=['CPUExecutionProvider'])
        onnx_input_name = ort_session.get_inputs()[0].name
        ort_inputs = {onnx_input_name: dummy_mels.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_tokens_np = ort_outputs[0]
        print(f"ONNX Runtime inference successful. Output tokens shape: {onnx_output_tokens_np.shape}")
        np.testing.assert_allclose(
            pytorch_output_tokens.cpu().detach().numpy(), 
            onnx_output_tokens_np, 
            rtol=1e-03, 
            atol=1e-05
        )
        print("ONNX Runtime outputs match PyTorch outputs numerically (within tolerance).")
    except Exception as e:
        print(f"Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
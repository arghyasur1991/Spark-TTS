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
import torchaudio.functional as F # Changed import
import argparse
import os
import numpy as np
from pathlib import Path
import onnxruntime

from sparktts.utils.file import load_config # To load mel_params from BiCodec config

class MelSpectrogramONNXWrapper(nn.Module):
    def __init__(self, mel_params: dict, device: torch.device): # Added device for window tensor
        super().__init__()
        
        self.n_fft = mel_params['n_fft']
        self.hop_length = mel_params.get('hop_length', self.n_fft // 4)
        self.win_length = mel_params.get('win_length', self.n_fft)
        
        # Create window tensor
        # Spectrogram default is hann_window if window is None
        self.window = torch.hann_window(self.win_length, device=device) # Ensure window is on the correct device

        self.center = True # torchaudio.functional.spectrogram default
        self.pad_mode = "reflect" # torchaudio.functional.spectrogram default
        self.normalized = False # torchaudio.functional.spectrogram default for power=None (magnitude)
        self.onesided = True # torchaudio.functional.spectrogram default for real input
        
        # Mel filterbank parameters
        n_stft = self.n_fft // 2 + 1
        f_min = mel_params.get('mel_fmin', 0.0)
        # Handle f_max being None explicitly
        f_max_param = mel_params.get('mel_fmax')
        if f_max_param is None:
            f_max = mel_params['sample_rate'] / 2.0
        else:
            f_max = f_max_param
            
        n_mels = mel_params['num_mels']
        sample_rate = mel_params['sample_rate']
        mel_norm = mel_params.get('norm', 'slaney') # MelSpectrogram default
        mel_scale_type = mel_params.get('mel_scale', 'slaney') # MelSpectrogram default

        # Register mel_fbanks as a buffer so it's part of the model's state_dict and moved to device
        self.register_buffer('mel_fbanks', F.melscale_fbanks(
            n_freqs=n_stft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=mel_norm,
            mel_scale=mel_scale_type
        ))

    def forward(self, wav_with_channel: torch.Tensor) -> torch.Tensor:
        # Input wav_with_channel shape: (B, 1, T_audio)
        # torch.stft expects input shape (B, T_audio) or (T_audio)
        
        # Squeeze channel dimension: (B, 1, T_audio) -> (B, T_audio)
        wav = wav_with_channel.squeeze(1)

        # 1. Compute STFT
        # torch.stft with return_complex=False returns a real tensor of shape (B, N, T, 2)
        # where N is n_fft // 2 + 1
        stft_result = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=False # Explicitly False
        )
        
        # 2. Compute Magnitude
        # stft_result shape: (B, N, T_frames, 2)
        # We want magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.norm(stft_result, p=2, dim=-1)
        # magnitude shape: (B, N, T_frames)
        
        # 3. Apply Mel Filterbank
        # self.mel_fbanks shape: (n_mels, N)
        # magnitude shape: (B, N, T_frames)
        # We need to make self.mel_fbanks compatible for batched matmul: (B, n_mels, N)
        
        batch_size = magnitude.shape[0]
        # Unsqueeze and expand self.mel_fbanks to match batch dimension of magnitude
        # self.mel_fbanks original shape: (n_mels, n_stft)
        # expanded_mel_fbanks shape: (batch_size, n_mels, n_stft)
        expanded_mel_fbanks = self.mel_fbanks.unsqueeze(0).expand(batch_size, -1, -1)
        
        # --- Diagnostic prints ---
        print(f"Before matmul - expanded_mel_fbanks shape: {expanded_mel_fbanks.shape}, dtype: {expanded_mel_fbanks.dtype}")
        print(f"Before matmul - magnitude shape: {magnitude.shape}, dtype: {magnitude.dtype}")
        # --- End diagnostic prints ---

        # Based on print output, expanded_mel_fbanks is (B, n_stft, n_mels).
        # We need (B, n_mels, n_stft) to matmul with magnitude (B, n_stft, T_frames).
        mel_fbanks_for_matmul = expanded_mel_fbanks.transpose(-2, -1) # Shape: (B, n_mels, n_stft)
        
        print(f"After transpose - mel_fbanks_for_matmul shape: {mel_fbanks_for_matmul.shape}, dtype: {mel_fbanks_for_matmul.dtype}") # Additional diagnostic

        # Result shape: (B, n_mels, T_frames)
        mel_output = torch.matmul(mel_fbanks_for_matmul, magnitude)
        
        return mel_output

def main():
    parser = argparse.ArgumentParser(description="Export SparkTTS MelSpectrogram to ONNX.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B). Used to find BiCodec's config.yaml for mel_params."
    )
    parser.add_argument(
        "--output_onnx_path", 
        type=str, 
        default="onnx_models/mel_spectrogram.onnx", 
        help="Full path to save the exported MelSpectrogram ONNX model."
    )
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--dummy_batch_size", type=int, default=1, help="Batch size for dummy waveform input.")
    parser.add_argument("--dummy_sequence_length", type=int, default=16000, 
                        help="Sequence length for dummy waveform input (e.g., 16000 for 1 sec at 16kHz).")

    args = parser.parse_args()
    export_device = torch.device(args.device) # Use 'export_device' to avoid conflict with wrapper's 'device' argument

    bicodec_config_path = Path(args.model_dir) / "BiCodec" / "config.yaml"
    print(f"Loading BiCodec config from: {bicodec_config_path}")
    if not bicodec_config_path.exists():
        print(f"ERROR: BiCodec config.yaml not found at {bicodec_config_path}")
        return
    
    full_config = load_config(bicodec_config_path)
    if 'audio_tokenizer' not in full_config or 'mel_params' not in full_config['audio_tokenizer']:
        print(f"ERROR: 'mel_params' not found in {bicodec_config_path} under 'audio_tokenizer' key.")
        return
    mel_parameters = full_config['audio_tokenizer']['mel_params']
    print(f"Loaded mel_params: {mel_parameters}")

    # Instantiate the ONNX wrapper, passing the export_device for window tensor creation
    onnx_exportable_mel_spectrogram = MelSpectrogramONNXWrapper(mel_parameters, device=export_device).to(export_device)
    onnx_exportable_mel_spectrogram.eval()

    dummy_waveform = torch.randn(
        args.dummy_batch_size,
        1, # Channel dimension
        args.dummy_sequence_length,
        device=export_device
    ).contiguous()
    print(f"Using dummy_waveform input shape (B, 1, T_audio): {dummy_waveform.shape}")

    print("Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            pytorch_output_mels = onnx_exportable_mel_spectrogram(dummy_waveform)
        print(f"PyTorch ONNX wrapper test forward pass successful. Output mels shape: {pytorch_output_mels.shape}")
    except Exception as e:
        print(f"Error during PyTorch ONNX wrapper test forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    input_names = ["raw_waveform_with_channel"]
    output_names = ["mel_spectrogram"]
    
    dynamic_axes = {
        input_names[0]: {0: 'batch_size', 2: 'sequence_length'}, 
        output_names[0]: {0: 'batch_size', 2: 'mel_time_steps'}
    }
    print(f"Using dynamic axes: {dynamic_axes}")

    output_dir = Path(args.output_onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting MelSpectrogram to ONNX: {args.output_onnx_path} with opset {args.opset_version}")
    try:
        torch.onnx.export(
            onnx_exportable_mel_spectrogram,
            dummy_waveform,
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

    print("\n--- Starting ONNX Model Verification ---")
    try:
        print("Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(args.output_onnx_path, providers=['CPUExecutionProvider'])
        onnx_input_name = ort_session.get_inputs()[0].name
        print(f"ONNX model loaded. Expected input name: {onnx_input_name}")

        ort_inputs = {onnx_input_name: dummy_waveform.cpu().numpy()}
        print("Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_mels_np = ort_outputs[0]
        print(f"ONNX Runtime inference successful. Output mels shape: {onnx_output_mels_np.shape}")

        np.testing.assert_allclose(
            pytorch_output_mels.cpu().detach().numpy(), 
            onnx_output_mels_np, 
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
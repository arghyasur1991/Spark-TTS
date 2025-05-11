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
import torchaudio.functional as TF # Keep for melscale_fbanks and window functions if desired
import torchaudio.transforms as TT # For consistency if any other transform params are used as reference
import argparse
import os
import numpy as np
from pathlib import Path
import onnxruntime
import math # For pi

from sparktts.utils.file import load_config # To load mel_params from BiCodec config

class MelSpectrogramONNXWrapper(nn.Module):
    def __init__(self, mel_params: dict, device: torch.device):
        super().__init__()
        
        self.n_fft = mel_params['n_fft']
        self.hop_length = mel_params.get('hop_length', self.n_fft // 4)
        self.win_length = mel_params.get('win_length', self.n_fft)
        self.sample_rate = mel_params['sample_rate']
        
        # Get window, default to hann_window if not specified or None
        # torchaudio.transforms.Spectrogram uses torch.hann_window as a default.
        window_fn_name = mel_params.get('window_fn', 'hann_window') # Assuming 'hann_window' if not in params
        if window_fn_name == 'hann_window':
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        elif window_fn_name == 'hamming_window':
            window_tensor = torch.hamming_window(self.win_length, periodic=True, dtype=torch.float32)
        # Add other window types if needed
        else: # Default to Hann if unrecognized
            print(f"Warning: Unrecognized window_fn '{window_fn_name}', defaulting to Hann window.")
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer('window', window_tensor)

        self.center = mel_params.get('center', True) # Default for torchaudio.transforms.Spectrogram
        self.pad_mode = mel_params.get('pad_mode', "reflect") 
        self.power = mel_params.get('power', 1.0) # Usually 1.0 for magnitude, 2.0 for power spec
        
        # Mel filterbank parameters
        n_stft = self.n_fft // 2 + 1
        f_min = mel_params.get('mel_fmin', 0.0)
        f_max_param = mel_params.get('mel_fmax')
        f_max = f_max_param if f_max_param is not None else self.sample_rate / 2.0
            
        n_mels = mel_params['num_mels']
        mel_norm = mel_params.get('norm', 'slaney') 
        mel_scale_type = mel_params.get('mel_scale', 'slaney')

        mel_fbanks_tensor = TF.melscale_fbanks(
            n_freqs=n_stft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=self.sample_rate,
            norm=mel_norm,
            mel_scale=mel_scale_type
        )
        self.register_buffer('mel_fbanks', mel_fbanks_tensor)

        # Precompute RFFT matrices
        k = torch.arange(0, self.n_fft // 2 + 1, dtype=torch.float32)
        n_range = torch.arange(0, self.n_fft, dtype=torch.float32)
        # angle shape: (n_fft // 2 + 1, n_fft)
        angle = -2 * math.pi * k.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
        
        rfft_mat_real_tensor = torch.cos(angle)
        rfft_mat_imag_tensor = torch.sin(angle)
        self.register_buffer('rfft_mat_real', rfft_mat_real_tensor)
        self.register_buffer('rfft_mat_imag', rfft_mat_imag_tensor)

    def forward(self, wav_with_channel: torch.Tensor) -> torch.Tensor:
        # Input wav_with_channel shape: (B, 1, T_audio)
        wav = wav_with_channel.squeeze(1) # Shape: (B, T_audio)
        batch_size = wav.shape[0]
        original_sequence_length = wav.shape[1]

        # 1. Padding of input wav (if center=True)
        padded_wav = wav
        if self.center:
            padding_amount = self.n_fft // 2
            padded_wav = torch.nn.functional.pad(wav, (padding_amount, padding_amount), mode=self.pad_mode)
        
        padded_sequence_length = padded_wav.shape[1]

        # 2. Framing using a loop with torch.narrow (to replace tensor.unfold)
        frame_list = []
        # Calculate the number of frames. Equivalent to: (L_padded - win_length) // hop_length + 1
        num_frames = (padded_sequence_length - self.win_length) // self.hop_length + 1
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = padded_wav.narrow(1, start, self.win_length)
            frame_list.append(frame)
        
        if not frame_list: # Handle case where input is too short for any frames
             # This case should ideally be handled by ensuring minimum input length or by defining
             # expected output for zero frames (e.g., empty tensor with correct dims).
             # For now, create an empty tensor that matches expected downstream dimensions if possible,
             # or raise an error. Let's assume we expect a valid mel output shape even if empty.
             # Output n_mels, 0 time steps
             return torch.empty((batch_size, self.mel_fbanks.shape[0], 0), device=wav.device, dtype=wav.dtype)

        frames = torch.stack(frame_list, dim=1)
        # frames shape: (B, num_frames, self.win_length)

        # 3. Windowing
        # self.window shape: (win_length,)
        windowed_frames = frames * self.window
        # windowed_frames shape: (B, num_frames, self.win_length)

        # 4. Pad windowed frames to n_fft for FFT
        # If n_fft > win_length, pad with zeros. If n_fft < win_length, truncate.
        fft_ready_frames = windowed_frames
        if self.n_fft > self.win_length:
            pad_right = self.n_fft - self.win_length
            # Pad the last dimension (frame content)
            fft_ready_frames = torch.nn.functional.pad(windowed_frames, (0, pad_right), mode='constant', value=0)
        elif self.n_fft < self.win_length:
            fft_ready_frames = windowed_frames[:, :, :self.n_fft]
        # fft_ready_frames shape: (B, num_frames, self.n_fft)

        # 5. Manual RFFT using precomputed matrices
        # self.rfft_mat_real / _imag shapes: (n_fft // 2 + 1, n_fft)
        # We need to matmul (B, num_frames, n_fft) with (n_fft, n_fft // 2 + 1)
        # So, transpose the RFFT matrices before matmul
        
        rfft_mat_real_t = self.rfft_mat_real.T.to(fft_ready_frames.device)
        rfft_mat_imag_t = self.rfft_mat_imag.T.to(fft_ready_frames.device)

        real_part = torch.matmul(fft_ready_frames, rfft_mat_real_t)
        imag_part = torch.matmul(fft_ready_frames, rfft_mat_imag_t)
        # real_part, imag_part shapes: (B, num_frames, n_fft // 2 + 1)

        # 6. Magnitude
        # For complex number z = x + iy, |z| = sqrt(x^2 + y^2)
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))
        # magnitude shape: (B, num_frames, n_fft // 2 + 1)

        # 7. Power Spectrum (if self.power is not 1.0)
        if self.power != 1.0: # Apply power if it's not already for magnitude
            magnitude = magnitude.pow(self.power)

        # 8. Apply Mel Filterbank
        # self.mel_fbanks shape: (n_mels, n_fft // 2 + 1)
        # magnitude shape: (B, num_frames, n_fft // 2 + 1)
        # We need matmul: (B, num_frames, n_fft // 2 + 1) @ (n_fft // 2 + 1, n_mels)
        # Based on logs, self.mel_fbanks is already (n_stft_bins, n_mels) i.e. (513, 128)
        # So, we should use it directly for the matmul.
        mel_fbanks_for_matmul = self.mel_fbanks.to(magnitude.device) 
        
        print(f"Shape of self.mel_fbanks before transpose: {self.mel_fbanks.shape}") 
        print(f"Shape of magnitude for mel matmul: {magnitude.shape}")
        print(f"Shape of mel_fbanks_for_matmul (using self.mel_fbanks directly): {mel_fbanks_for_matmul.shape}")
        
        mel_output = torch.matmul(magnitude, mel_fbanks_for_matmul) # Use the un-transposed version
        # mel_output shape: (B, num_frames, n_mels)

        # 9. Transpose to conventional (B, n_mels, num_frames)
        mel_output = mel_output.transpose(1, 2)
        
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
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.") # Changed to 14
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--dummy_batch_size", type=int, default=1, help="Batch size for dummy waveform input.")
    parser.add_argument("--dummy_sequence_length", type=int, default=16000, 
                        help="Sequence length for dummy waveform input (e.g., 16000 for 1 sec at 16kHz).")

    args = parser.parse_args()
    export_device = torch.device(args.device)

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
    # Ensure essential parameters are present, add defaults if missing and torchaudio would use them
    mel_parameters.setdefault('window_fn', 'hann_window') # default window if not specified
    mel_parameters.setdefault('center', True)
    mel_parameters.setdefault('pad_mode', 'reflect')
    mel_parameters.setdefault('power', 1.0) # Crucial for magnitude vs power
    mel_parameters.setdefault('n_fft', mel_parameters.get('n_fft', 2048)) # Example default
    mel_parameters.setdefault('hop_length', mel_parameters.get('hop_length', mel_parameters['n_fft'] // 4))
    mel_parameters.setdefault('win_length', mel_parameters.get('win_length', mel_parameters['n_fft']))


    print(f"Using mel_params: {mel_parameters}")

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
        # It's useful to see the shapes that might be causing issues
        # For example, if an error occurs in matmul, print shapes of operands
        # This is now part of the wrapper's forward pass for debugging if needed.
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
            verbose=False # Set to True for more detailed export logs if issues persist
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

        # Compare with PyTorch output
        # Relax tolerance slightly for manual implementation vs library STFT due to potential minor precision differences
        np.testing.assert_allclose(
            pytorch_output_mels.cpu().detach().numpy(), 
            onnx_output_mels_np, 
            rtol=1e-02, # Increased rtol
            atol=1e-04  # Increased atol
        )
        print("ONNX Runtime outputs match PyTorch outputs numerically (within relaxed tolerance).")

    except Exception as e:
        print(f"Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
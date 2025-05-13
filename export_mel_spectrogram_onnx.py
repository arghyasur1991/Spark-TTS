"""
Exports a MelSpectrogram computation module to ONNX.

This script defines a PyTorch module that manually computes Mel spectrograms
from raw audio waveforms. This manual implementation is designed to be ONNX-exportable
and replicates the behavior of standard library functions like torchaudio.transforms.MelSpectrogram,
but with explicit operations suitable for ONNX conversion.

The main components are:
1.  `MelSpectrogramONNXWrapper`: A nn.Module that takes a raw waveform and outputs its Mel spectrogram.
    It includes manual implementations of STFT (framing, windowing, RFFT) and mel scaling.
2.  A `main` function to parse arguments, load mel parameters from a BiCodec configuration, 
    instantiate the wrapper, export it to ONNX, and verify the ONNX model.
"""

import argparse
import math # For pi
from pathlib import Path
import sys # For sys.exit()

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torchaudio.functional as TF # For melscale_fbanks and window functions

# Assuming load_config is in sparktts.utils.file
from sparktts.utils.file import load_config 

class MelSpectrogramONNXWrapper(nn.Module):
    """
    A PyTorch module to compute Mel spectrograms from raw audio, designed for ONNX export.

    This module manually implements Short-Time Fourier Transform (STFT) and mel scaling
    to ensure ONNX compatibility.
    """
    def __init__(self, mel_params: dict, device: torch.device):
        super().__init__()
        
        self.n_fft = mel_params['n_fft']
        self.hop_length = mel_params.get('hop_length', self.n_fft // 4)
        self.win_length = mel_params.get('win_length', self.n_fft)
        self.sample_rate = mel_params['sample_rate']
        
        window_fn_name = mel_params.get('window_fn', 'hann_window')
        if window_fn_name == 'hann_window':
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        elif window_fn_name == 'hamming_window':
            window_tensor = torch.hamming_window(self.win_length, periodic=True, dtype=torch.float32)
        else:
            print(f"[WARNING] Unrecognized window_fn '{window_fn_name}', defaulting to Hann window.")
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer('window', window_tensor.to(device)) # Ensure window is on the correct device

        self.center = mel_params.get('center', True)
        self.pad_mode = mel_params.get('pad_mode', "reflect") 
        self.power = mel_params.get('power', 1.0) 
        
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
        self.register_buffer('mel_fbanks', mel_fbanks_tensor.to(device)) # Ensure on device

        # Precompute RFFT matrices (real and imaginary parts)
        # These matrices are used to perform RFFT via matrix multiplication.
        k_range = torch.arange(0, self.n_fft // 2 + 1, dtype=torch.float32, device=device)
        n_range = torch.arange(0, self.n_fft, dtype=torch.float32, device=device)
        angle = -2 * math.pi * k_range.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
        
        rfft_mat_real_tensor = torch.cos(angle)
        rfft_mat_imag_tensor = torch.sin(angle)
        # Store transposed versions for efficient matmul later: (n_fft, n_fft // 2 + 1)
        self.register_buffer('rfft_mat_real_t', rfft_mat_real_tensor.T)
        self.register_buffer('rfft_mat_imag_t', rfft_mat_imag_tensor.T)

    def forward(self, wav_with_channel: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mel spectrogram from a batch of raw audio waveforms.

        Args:
            wav_with_channel (torch.Tensor): Input waveform tensor with shape (B, 1, T_audio).

        Returns:
            torch.Tensor: Mel spectrogram tensor with shape (B, n_mels, num_frames).
        """
        if wav_with_channel.ndim != 3 or wav_with_channel.shape[1] != 1:
            # This should ideally raise an error that propagates, or be handled before ONNX export if shapes are fixed
            print(f"[ERROR] Expected input shape (B, 1, T_audio), got {wav_with_channel.shape}")
            # For ONNX export, it's better to conform to dummy input shape or ensure model handles variability.
            # If this occurs during export with dummy_input, it's a setup error.
            raise ValueError(f"MelSpectrogramONNXWrapper: Invalid input shape. Expected (B, 1, T_audio), got {wav_with_channel.shape}")
        
        wav = wav_with_channel.squeeze(1) # Shape: (B, T_audio)
        batch_size = wav.shape[0]

        # 1. Padding (if center=True)
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
        windowed_frames = frames * self.window 

        # 4. Pad windowed frames to n_fft for FFT if win_length < n_fft
        fft_ready_frames = windowed_frames
        if self.n_fft > self.win_length:
            pad_right = self.n_fft - self.win_length
            fft_ready_frames = torch.nn.functional.pad(windowed_frames, (0, pad_right), mode='constant', value=0)
        elif self.n_fft < self.win_length: 
            fft_ready_frames = windowed_frames[:, :, :self.n_fft]

        # 5. Manual RFFT using precomputed matrices
        real_part = torch.matmul(fft_ready_frames, self.rfft_mat_real_t)
        imag_part = torch.matmul(fft_ready_frames, self.rfft_mat_imag_t)

        # 6. Magnitude (Complex modulus)
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))

        # 7. Power Spectrum (if self.power is not 1.0, e.g., 2.0 for power)
        if self.power != 1.0:
            magnitude = magnitude.pow(self.power)

        # 8. Apply Mel Filterbank
        mel_output = torch.matmul(magnitude, self.mel_fbanks)

        # 9. Transpose to conventional (B, n_mels, num_frames)
        mel_output = mel_output.transpose(1, 2)
        
        return mel_output

def export_mel_spectrogram_to_onnx(
    model_dir: Path,
    output_path: Path,
    opset_version: int,
    device_str: str,
    dummy_batch_size: int,
    dummy_sequence_length: int,
):
    """
    Exports the MelSpectrogramONNXWrapper to an ONNX model.
    Args are passed from the main orchestrator script or CLI.
    """
    print(f"[INFO] Starting MelSpectrogram ONNX export process...")
    print(f"[INFO]   Base model directory (for config): {model_dir}")
    print(f"[INFO]   Output ONNX path: {output_path}")
    print(f"[INFO]   Opset version: {opset_version}")
    print(f"[INFO]   Device: {device_str}")

    export_device = torch.device(device_str)

    # 1. Load mel_params from BiCodec config
    bicodec_config_path = model_dir / "BiCodec" / "config.yaml"
    print(f"[INFO] Loading BiCodec config from: {bicodec_config_path}")
    if not bicodec_config_path.exists():
        print(f"[ERROR] BiCodec config.yaml not found at {bicodec_config_path}")
        sys.exit(1)
    
    try:
        full_config = load_config(bicodec_config_path)
        if 'audio_tokenizer' not in full_config or 'mel_params' not in full_config['audio_tokenizer']:
            print(f"[ERROR] 'mel_params' not found in {bicodec_config_path} under 'audio_tokenizer' key.")
            sys.exit(1)
        mel_parameters = full_config['audio_tokenizer']['mel_params']
        # Ensure essential parameters have defaults
        mel_parameters.setdefault('n_fft', 2048)
        mel_parameters.setdefault('hop_length', mel_parameters['n_fft'] // 4)
        mel_parameters.setdefault('win_length', mel_parameters['n_fft'])
        mel_parameters.setdefault('window_fn', 'hann_window')
        mel_parameters.setdefault('center', True)
        mel_parameters.setdefault('pad_mode', 'reflect')
        mel_parameters.setdefault('power', 1.0)
        mel_parameters.setdefault('num_mels', 128)
        mel_parameters.setdefault('sample_rate', 24000) # Must match training data for BiCodec
        mel_parameters.setdefault('mel_fmin', 0.0)
        # mel_fmax defaults to sample_rate / 2 in the wrapper if not present or None

    except Exception as e:
        print(f"[ERROR] Failed to load or parse mel_params from config: {e}")
        sys.exit(1)
    print(f"[INFO] Using mel_params: {mel_parameters}")

    # 2. Instantiate the ONNX wrapper
    try:
        onnx_exportable_mel_spectrogram = MelSpectrogramONNXWrapper(mel_parameters, device=export_device).to(export_device)
        onnx_exportable_mel_spectrogram.eval()
    except Exception as e:
        print(f"[ERROR] Failed to instantiate MelSpectrogramONNXWrapper: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Prepare dummy input
    dummy_waveform = torch.randn(
        dummy_batch_size,
        1, # Channel dimension (B, 1, T_audio)
        dummy_sequence_length,
        device=export_device
    ).contiguous()
    print(f"[INFO] Using dummy_waveform input shape (B, 1, T_audio): {dummy_waveform.shape}")

    # Test forward pass with the PyTorch wrapper
    print("[INFO] Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            pytorch_output_mels = onnx_exportable_mel_spectrogram(dummy_waveform)
        print(f"[INFO] PyTorch wrapper test forward pass successful. Output mels shape: {pytorch_output_mels.shape}")
        if pytorch_output_mels.numel() == 0 and dummy_sequence_length >= mel_parameters.get('win_length', mel_parameters['n_fft']):
            print("[WARNING] PyTorch wrapper produced empty mels for a non-empty input sequence. Check STFT logic if input is valid.")
    except Exception as e:
        print(f"[ERROR] PyTorch wrapper test forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Export to ONNX
    input_names = ["raw_waveform_with_channel"]
    output_names = ["mel_spectrogram"]
    
    dynamic_axes = {
        input_names[0]: {0: 'batch_size', 2: 'sequence_length'}, 
        output_names[0]: {0: 'batch_size', 2: 'mel_time_steps'}
    }
    print(f"[INFO] Using dynamic axes: {dynamic_axes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting MelSpectrogram to ONNX: {output_path}")
    try:
        torch.onnx.export(
            onnx_exportable_mel_spectrogram,
            dummy_waveform,
            str(output_path), 
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print("[INFO] ONNX export successful.")
    except Exception as e:
        print(f"[ERROR] torch.onnx.export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify the ONNX Model
    print("[INFO] --- Starting ONNX Model Verification ---")
    try:
        print("[INFO] Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        onnx_input_name = ort_session.get_inputs()[0].name
        print(f"[INFO] ONNX model loaded. Input name: {onnx_input_name}")

        ort_inputs = {onnx_input_name: dummy_waveform.cpu().numpy()}
        print("[INFO] Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_mels_np = ort_outputs[0]
        print(f"[INFO] ONNX Runtime inference successful. Output mels shape: {onnx_output_mels_np.shape}")

        np.testing.assert_allclose(
            pytorch_output_mels.cpu().detach().numpy(), 
            onnx_output_mels_np, 
            rtol=1e-03, 
            atol=1e-05
        )
        print("[INFO] ONNX Runtime outputs match PyTorch outputs numerically (within relaxed tolerance). Verification successful.")

    except Exception as e:
        print(f"[ERROR] Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()
        # Not exiting with sys.exit(1) here, as export was successful, only verification failed.
    print("[INFO] MelSpectrogram ONNX export and verification process complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Export a manually implemented MelSpectrogram module to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", 
        type=Path, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B). Used to find BiCodec's config.yaml for mel_params."
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        default=Path("onnx_models/mel_spectrogram.onnx"), 
        help="Full path to save the exported MelSpectrogram ONNX model."
    )
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run export on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--dummy_batch_size", type=int, default=1, help="Batch size for dummy waveform input.")
    parser.add_argument(
        "--dummy_sequence_length", 
        type=int, 
        default=16000, 
        help="Sequence length for dummy waveform input (e.g., 16000 for 1 sec at 16kHz)."
    )

    args = parser.parse_args()

    export_mel_spectrogram_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output_path,
        opset_version=args.opset_version,
        device_str=args.device,
        dummy_batch_size=args.dummy_batch_size,
        dummy_sequence_length=args.dummy_sequence_length,
    )

if __name__ == "__main__":
    main() 
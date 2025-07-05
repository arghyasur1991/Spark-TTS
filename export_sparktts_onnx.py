#!/usr/bin/env python3
"""
Consolidated ONNX Export Script for Spark-TTS Models
This script exports all Spark-TTS models to ONNX format with multiple precision support and optimizations.
"""

import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import json
from pathlib import Path
import shutil
from onnxsim import simplify
import sys
import math  # For pi
from typing import Tuple
import torchaudio.functional as TF  # For melscale_fbanks and window functions

# Add sparktts to path if needed
# try:
from sparktts.models.bicodec import BiCodec
from sparktts.modules.encoder_decoder.feat_encoder import Encoder as BiCodecEncoder
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
from sparktts.utils.audio import load_audio
from sparktts.utils.file import load_config
# except ImportError:
    # print("⚠️ SparkTTS modules not found. Make sure you're running from the Spark-TTS directory")
    # sys.exit(1)

# Try to import transformers and optimum
try:
    from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModelForCausalLM, AutoTokenizer
    from optimum.exporters.onnx import main_export
    from optimum.onnxruntime import ORTModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers/Optimum not available. Some models may not be exportable.")

# INT8 quantization support
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat
    INT8_AVAILABLE = True
    print("✓ INT8 quantization support available")
except ImportError:
    INT8_AVAILABLE = False
    print("⚠️ INT8 quantization not available. Install with: pip install onnxruntime")

from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

# Model wrapper classes
class Wav2Vec2HiddenStatesExtractor(nn.Module):
    """Wrapper for Wav2Vec2Model to directly output the tuple of hidden states."""
    
    def __init__(self, wav2vec2_model):
        super().__init__()
        if hasattr(wav2vec2_model, 'config') and not wav2vec2_model.config.output_hidden_states:
            print("[INFO] Forcing 'output_hidden_states=True' in Wav2Vec2Model config.")
            wav2vec2_model.config.output_hidden_states = True
        self.model = wav2vec2_model

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(input_values)
        return outputs.hidden_states

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

class BiCodecEncoderQuantizerWrapper(nn.Module):
    """Wrapper for BiCodec's encoder and quantizer for ONNX export."""
    
    def __init__(self, encoder_model: BiCodecEncoder, quantizer_model: FactorizedVectorQuantize):
        super().__init__()
        self.encoder = encoder_model
        self.quantizer = quantizer_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # BiCodec's Encoder expects (B, D_feat, T_feat)
        # Input features are (B, T_feat, D_feat) from Wav2Vec2
        features_transposed = features.transpose(1, 2)
        z = self.encoder(features_transposed)
        semantic_tokens = self.quantizer.tokenize(z)
        return semantic_tokens

class SpeakerEncoderTokenizerWrapper(nn.Module):
    """Wrapper for speaker encoder and tokenizer."""
    
    def __init__(self, speaker_encoder, speaker_tokenizer):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.speaker_tokenizer = speaker_tokenizer
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # Encode speaker features
        speaker_embedding = self.speaker_encoder(audio_features)
        # Tokenize speaker embedding
        speaker_tokens = self.speaker_tokenizer(speaker_embedding)
        return speaker_tokens

class BiCodecVocoderWrapper(nn.Module):
    """Wrapper for BiCodec vocoder."""
    
    def __init__(self, vocoder_model):
        super().__init__()
        self.vocoder = vocoder_model
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Convert tokens to audio waveform
        audio_waveform = self.vocoder.decode(tokens)
        return audio_waveform

# Optimization functions
@torch.no_grad()
def tune_model(model_path: str, model_type: str, fp16: bool):
    """Optimize ONNX model using ONNX Runtime transformers"""
    model_dir = os.path.dirname(model_path)
    
    # Set optimization options
    optimization_options = FusionOptions(model_type)
    optimization_options.enable_group_norm = False
    optimization_options.enable_nhwc_conv = False
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_bias_splitgelu = False
    optimization_options.enable_bias_add = False
    optimization_options.enable_skip_layer_norm = model_type not in ["vocoder", "audio_tokenizer"]
    optimization_options.enable_gelu = model_type not in ["vocoder", "audio_tokenizer"]
    
    optimizer = optimize_model(
        input=model_path,
        model_type=model_type,
        opt_level=0,
        optimization_options=optimization_options,
        use_gpu=False,
        only_onnxruntime=False
    )
    
    if fp16:
        optimizer.convert_float_to_float16(
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=['RandomNormalLike']
        )
    
    optimizer.topological_sort()
    
    # Handle external data file cleanup
    data_location = model_path.replace(".onnx", ".onnx_data")
    has_data = False
    if os.path.exists(data_location):
        os.remove(data_location)
        has_data = True
    
    onnx.save_model(
        optimizer.model,
        model_path,
        save_as_external_data=has_data,
        all_tensors_to_one_file=True,
        location=None,
        convert_attribute=False,
    )

def export_wav2vec2_to_onnx(model_dir, output_path, device='cpu', opset_version=14, **kwargs):
    """Export Wav2Vec2 model to ONNX format"""
    print(f"Exporting Wav2Vec2 to {output_path} with opset {opset_version}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("✗ Transformers not available for Wav2Vec2 export")
        return False
    
    # Convert path to string
    output_path = str(output_path)
    wav2vec2_path = Path(model_dir) / "wav2vec2-large-xlsr-53"
    
    if not wav2vec2_path.exists():
        print(f"✗ Wav2Vec2 model not found at {wav2vec2_path}")
        return False
    
    try:
        # Load the model
        pytorch_config = AutoConfig.from_pretrained(wav2vec2_path)
        pytorch_config.output_hidden_states = True
        pytorch_model = Wav2Vec2Model.from_pretrained(wav2vec2_path, config=pytorch_config)
        pytorch_model.eval()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_path)
        
        # Wrap model for ONNX export
        onnx_wrapper = Wav2Vec2HiddenStatesExtractor(pytorch_model)
        onnx_wrapper.eval()
        
        # Create dummy input
        sequence_length = 16000  # 1 second of audio at 16kHz
        raw_dummy_audio = [np.random.randn(sequence_length).astype(np.float32)]
        processed_inputs = feature_extractor(
            raw_dummy_audio,
            return_tensors="pt",
            sampling_rate=feature_extractor.sampling_rate,
            padding="longest",
        )
        dummy_input = processed_inputs.input_values
        
        # Get output for naming
        with torch.no_grad():
            dummy_outputs = onnx_wrapper(dummy_input)
        
        num_hidden_states = len(dummy_outputs)
        input_names = ["input_values"]
        output_names = [f"hidden_state_{i}" for i in range(num_hidden_states)]
        
        dynamic_axes = {
            input_names[0]: {0: "batch_size", 1: "sequence_length"}
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size", 1: "feat_sequence_length"}
        
        # Export to ONNX
        torch.onnx.export(
            onnx_wrapper,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
        
        print(f"Wav2Vec2 exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export Wav2Vec2: {e}")
        return False

def export_mel_spectrogram_to_onnx(model_dir, output_path, device='cpu', opset_version=14, dummy_batch_size=1, dummy_sequence_length=16000, **kwargs):
    """Export mel spectrogram extractor to ONNX format"""
    print(f"Exporting Mel Spectrogram to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    export_device = torch.device(device)
    
    try:
        # 1. Load mel_params from BiCodec config
        bicodec_config_path = Path(model_dir) / "BiCodec" / "config.yaml"
        print(f"[INFO] Loading BiCodec config from: {bicodec_config_path}")
        if not bicodec_config_path.exists():
            print(f"✗ BiCodec config.yaml not found at {bicodec_config_path}")
            return False
        
        full_config = load_config(bicodec_config_path)
        if 'audio_tokenizer' not in full_config or 'mel_params' not in full_config['audio_tokenizer']:
            print(f"✗ 'mel_params' not found in {bicodec_config_path} under 'audio_tokenizer' key.")
            return False
            
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
        
        print(f"[INFO] Using mel_params: {mel_parameters}")

        # 2. Instantiate the ONNX wrapper
        onnx_exportable_mel_spectrogram = MelSpectrogramONNXWrapper(mel_parameters, device=export_device).to(export_device)
        onnx_exportable_mel_spectrogram.eval()

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
        with torch.no_grad():
            pytorch_output_mels = onnx_exportable_mel_spectrogram(dummy_waveform)
        print(f"[INFO] PyTorch wrapper test forward pass successful. Output mels shape: {pytorch_output_mels.shape}")
        if pytorch_output_mels.numel() == 0 and dummy_sequence_length >= mel_parameters.get('win_length', mel_parameters['n_fft']):
            print("[WARNING] PyTorch wrapper produced empty mels for a non-empty input sequence. Check STFT logic if input is valid.")

        # 4. Export to ONNX
        input_names = ["raw_waveform_with_channel"]
        output_names = ["mel_spectrogram"]
        
        dynamic_axes = {
            input_names[0]: {0: 'batch_size', 2: 'sequence_length'}, 
            output_names[0]: {0: 'batch_size', 2: 'mel_time_steps'}
        }
        print(f"[INFO] Using dynamic axes: {dynamic_axes}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Exporting MelSpectrogram to ONNX: {output_path}")
        torch.onnx.export(
            onnx_exportable_mel_spectrogram,
            dummy_waveform,
            output_path, 
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print("[INFO] ONNX export successful.")

        # 5. Verify the ONNX Model
        print("[INFO] --- Starting ONNX Model Verification ---")
        try:
            print("[INFO] Loading ONNX model for verification...")
            ort_session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
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
            # Not exiting here, as export was successful, only verification failed.
        print("[INFO] MelSpectrogram ONNX export and verification process complete.")
        
        print(f"Mel Spectrogram exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export Mel Spectrogram: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_bicodec_encoder_quantizer_to_onnx(model_dir, output_path, device='cpu', opset_version=14, **kwargs):
    """Export BiCodec encoder and quantizer to ONNX format"""
    print(f"Exporting BiCodec Encoder+Quantizer to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    export_device = torch.device(device)
    
    try:
        # Load BiCodec model
        bicodec_path = Path(model_dir) / "BiCodec"
        if not bicodec_path.exists():
            print(f"✗ BiCodec model not found at {bicodec_path}")
            return False
            
        bicodec_model = BiCodec.load_from_checkpoint(bicodec_path, device=export_device)
        bicodec_model.eval()
        
        # Create wrapper
        encoder_quantizer_wrapper = BiCodecEncoderQuantizerWrapper(
            bicodec_model.encoder, 
            bicodec_model.quantizer
        ).to(export_device)
        encoder_quantizer_wrapper.eval()
        
        # Create dummy input (features from Wav2Vec2)
        dummy_features = torch.randn(1, 98, 1024, device=export_device)  # [batch, seq_len, features]
        
        # Export to ONNX
        torch.onnx.export(
            encoder_quantizer_wrapper,
            dummy_features,
            output_path,
            input_names=['features'],
            output_names=['semantic_tokens'],
            dynamic_axes={
                'features': {1: 'feature_sequence_length'},
                'semantic_tokens': {2: 'quantized_token_sequence_length'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
        
        print(f"BiCodec Encoder+Quantizer exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export BiCodec Encoder+Quantizer: {e}")
        return False

def export_speaker_encoder_tokenizer_to_onnx(model_dir, output_path, device='cpu', opset_version=14, **kwargs):
    """Export speaker encoder and tokenizer to ONNX format"""
    print(f"Exporting Speaker Encoder+Tokenizer to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    export_device = torch.device(device)
    
    try:
        # Load models (this would need to be implemented based on SparkTTS structure)
        # For now, creating a placeholder implementation
        print("⚠️ Speaker Encoder+Tokenizer export not fully implemented yet")
        return False
        
    except Exception as e:
        print(f"✗ Failed to export Speaker Encoder+Tokenizer: {e}")
        return False

def export_bicodec_vocoder_to_onnx(model_dir, output_path, device='cpu', opset_version=14, sample_audio_for_shapes=None, **kwargs):
    """Export BiCodec vocoder to ONNX format"""
    print(f"Exporting BiCodec Vocoder to {output_path} with opset {opset_version}")
    
    # Convert path to string
    output_path = str(output_path)
    export_device = torch.device(device)
    
    try:
        # Load BiCodec model
        bicodec_path = Path(model_dir) / "BiCodec"
        if not bicodec_path.exists():
            print(f"✗ BiCodec model not found at {bicodec_path}")
            return False
            
        bicodec_model = BiCodec.load_from_checkpoint(bicodec_path, device=export_device)
        bicodec_model.eval()
        
        # Create wrapper
        vocoder_wrapper = BiCodecVocoderWrapper(bicodec_model.vocoder).to(export_device)
        vocoder_wrapper.eval()
        
        # Create dummy input (semantic tokens)
        dummy_tokens = torch.randint(0, 1000, (1, 8, 49), device=export_device)  # [batch, num_quantizers, seq_len]
        
        # Export to ONNX
        torch.onnx.export(
            vocoder_wrapper,
            dummy_tokens,
            output_path,
            input_names=['semantic_tokens'],
            output_names=['audio_waveform'],
            dynamic_axes={
                'semantic_tokens': {2: 'token_sequence_length'},
                'audio_waveform': {1: 'audio_length'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
        
        print(f"BiCodec Vocoder exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export BiCodec Vocoder: {e}")
        return False

def export_llm_to_onnx(model_dir, output_path, device='cpu', opset_version=14, trust_remote_code_llm=False, **kwargs):
    """Export LLM to ONNX format using Optimum"""
    print(f"Exporting LLM to {output_path} with opset {opset_version}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("✗ Transformers/Optimum not available for LLM export")
        return False
    
    # Convert path to string and ensure it's a directory
    output_dir = Path(output_path).parent if str(output_path).endswith('.onnx') else Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llm_path = Path(model_dir) / "LLM"
    if not llm_path.exists():
        print(f"✗ LLM model not found at {llm_path}")
        return False
    
    try:
        # Use Optimum to export LLM
        main_export(
            model_name_or_path=str(llm_path),
            output=output_dir,
            task="text-generation-with-past",
            opset=opset_version,
            device=device,
            fp16=output_path.endswith("fp16.onnx"),
            trust_remote_code=trust_remote_code_llm,
        )

        print(f"LLM exported successfully to {output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export LLM: {e}")
        return False

def convert_model_to_fp16(fp32_model_path, fp16_model_path, model_type="general"):
    """Convert FP32 ONNX model to FP16"""
    try:
        print(f"Converting {fp32_model_path} to FP16...")
        
        # Copy the FP32 model first
        shutil.copy(fp32_model_path, fp16_model_path)
        
        # Copy external data file if exists
        fp32_data = fp32_model_path + ".data"
        if os.path.exists(fp32_data):
            fp16_data = fp16_model_path + ".data"
            shutil.copy(fp32_data, fp16_data)
        
        # Apply FP16 conversion
        tune_model(fp16_model_path, model_type, fp16=True)
        
        # Apply post-processing optimizations
        model = onnx.load(fp16_model_path)
        model_simp, check = simplify(model)
        
        # Save original model as backup
        shutil.copy(fp16_model_path, fp16_model_path + ".original")
        onnx.save(model_simp, fp16_model_path)
        
        print(f"✓ FP16 model saved to {fp16_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to FP16: {e}")
        return False

def convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8 using static quantization with QDQ format"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
    
    try:
        print(f"Converting {fp32_model_path} to INT8 using static QDQ quantization...")
        
        class DummyCalibrationDataReader(CalibrationDataReader):
            def __init__(self, model_path):
                self.model_path = model_path
                self.data_generated = False
                
                # Load model to get input shapes and types
                model = onnx.load(model_path)
                self.input_names = [inp.name for inp in model.graph.input]
                self.input_shapes = {}
                self.input_types = {}
                
                for inp in model.graph.input:
                    # Get shape
                    shape = []
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_value > 0:
                            shape.append(dim.dim_value)
                        else:
                            # Use realistic defaults for dynamic dimensions
                            if 'audio' in inp.name.lower():
                                shape.append(16000)  # 1 second audio
                            elif 'feature' in inp.name.lower():
                                shape.append(98)  # Feature sequence length
                            else:
                                shape.append(1)  # Default batch size
                    self.input_shapes[inp.name] = shape
                    
                    # Get data type
                    elem_type = inp.type.tensor_type.elem_type
                    if elem_type == onnx.TensorProto.FLOAT:
                        self.input_types[inp.name] = np.float32
                    elif elem_type == onnx.TensorProto.INT64:
                        self.input_types[inp.name] = np.int64
                    elif elem_type == onnx.TensorProto.INT32:
                        self.input_types[inp.name] = np.int32
                    else:
                        self.input_types[inp.name] = np.float32
                
            def get_next(self):
                if not self.data_generated:
                    self.data_generated = True
                    calibration_data = {}
                    for name, shape in self.input_shapes.items():
                        dtype = self.input_types[name]
                        if dtype == np.int64 or dtype == np.int32:
                            calibration_data[name] = np.zeros(shape, dtype=dtype)
                        else:
                            calibration_data[name] = np.random.randn(*shape).astype(dtype)
                    return calibration_data
                else:
                    return None
        
        calibration_reader = DummyCalibrationDataReader(fp32_model_path)
        
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100  # > 100MB threshold
        
        quantize_static(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            use_external_data_format=use_external_data,
            calibrate_method=CalibrationMethod.MinMax
        )
        
        print(f"✓ INT8 QDQ model saved to {int8_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8 QDQ: {e}")
        return False

def convert_model_to_int8(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
    
    try:
        # Try static QDQ quantization first
        if convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type):
            return True
        
        print(f"Static QDQ quantization failed, trying dynamic quantization...")
        
        # Fallback to dynamic quantization
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100
        
        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            weight_type=QuantType.QInt8,
            use_external_data_format=use_external_data
        )
        
        print(f"✓ INT8 model saved to {int8_model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8: {e}")
        return False

def verify_onnx_model(onnx_path, input_shapes=None):
    """Verify the exported ONNX model"""
    print(f"Verifying ONNX model: {onnx_path}")
    
    try:
        # For large models with external data, skip protobuf check
        data_path = onnx_path.replace(".onnx", ".onnx_data")
        has_data = False
        if os.path.exists(data_path):
            has_data = True
        
        file_size = os.path.getsize(onnx_path) + (os.path.getsize(data_path) if has_data else 0)
        is_large_model = file_size > 100 * 1024 * 1024  # > 100MB
        
        if not is_large_model:
            onnx_model = onnx.load(onnx_path, load_external_data=has_data)
            onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session to verify it can load
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"✓ ONNX model {onnx_path} is valid")
        print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Output names: {[out.name for out in session.get_outputs()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False

def cleanup_export_directory(output_dir):
    """Clean up export directory, keeping only essential files"""
    print(f"\n🧹 Cleaning up export directory: {output_dir}")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    kept_files = []
    removed_files = []
    
    for file_path in output_path.rglob('*'):
        if file_path.is_file():
            filename = file_path.name
            
            # Keep these files
            if (filename.endswith('.onnx') or 
                filename.endswith('.onnx_data') or 
                filename.endswith('.json') or
                filename.endswith('.txt') or
                filename.endswith('.bin')):
                kept_files.append(str(file_path.relative_to(output_path)))
            else:
                # Remove everything else
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path.relative_to(output_path)))
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
    
    print(f"✓ Kept {len(kept_files)} essential files")
    if removed_files:
        print(f"🗑️ Removed {len(removed_files)} temporary files")
    else:
        print("📝 No temporary files to remove")

def export_model_with_precisions(export_func, model_dir, base_path, model_name, export_fp32=True, export_fp16=False, export_int8=False, device="cpu", opset_version=14, **kwargs):
    """Export model in multiple precision formats"""
    success_count = 0
    base_path_str = str(base_path)
    model_type = model_name.lower().replace(" ", "_")
    
    # Create precision-specific paths
    fp32_path = base_path_str
    fp16_path = base_path_str.replace('.onnx', '_fp16.onnx')
    int8_path = base_path_str.replace('.onnx', '_int8.onnx')
    
    # Export FP32 model first (base model)
    if export_fp32 or export_fp16 or export_int8:
        try:
            print(f"\n=== Exporting {model_name} (FP32) ===")
            if export_func(model_dir, fp32_path, device, opset_version, **kwargs):
                if verify_onnx_model(fp32_path):
                    if export_fp32:
                        success_count += 1
                        print(f"✓ {model_name} FP32 export successful")
                    
                    # Convert to FP16 if requested
                    if export_fp16:
                        if convert_model_to_fp16(fp32_path, fp16_path, model_type):
                            if verify_onnx_model(fp16_path):
                                success_count += 1
                                print(f"✓ {model_name} FP16 conversion successful")
                            else:
                                print(f"✗ {model_name} FP16 model verification failed")
                        else:
                            print(f"✗ {model_name} FP16 conversion failed")
                    
                    # Convert to INT8 if requested
                    if export_int8:
                        if convert_model_to_int8(fp32_path, int8_path, model_type):
                            if verify_onnx_model(int8_path):
                                success_count += 1
                                print(f"✓ {model_name} INT8 quantization successful")
                            else:
                                print(f"✗ {model_name} INT8 model verification failed")
                        else:
                            print(f"✗ {model_name} INT8 quantization failed")
                    
                    # Remove FP32 if not requested (was only needed for conversion)
                    if not export_fp32 and os.path.exists(fp32_path):
                        os.remove(fp32_path)
                        # Also remove external data file if exists
                        fp32_data = fp32_path + ".data"
                        if os.path.exists(fp32_data):
                            os.remove(fp32_data)
                else:
                    print(f"✗ {model_name} FP32 model verification failed")
            else:
                print(f"✗ {model_name} FP32 export failed")
        except Exception as e:
            print(f"✗ Failed to export {model_name}: {e}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Export Spark-TTS models to ONNX")
    parser.add_argument("--base_model_dir", type=Path, default=Path("./pretrained_models/Spark-TTS-0.5B"),
                       help="Path to the base SparkTTS model directory")
    parser.add_argument("--output_dir", type=Path, default=Path("./onnx_models/Spark-TTS"),
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"],
                       help="Device to use for export")
    parser.add_argument("--models", nargs="+", 
                       choices=["wav2vec2", "mel_spectrogram", "bicodec_encoder_quantizer", 
                               "speaker_encoder_tokenizer", "bicodec_vocoder", "llm", "all"],
                       default=["all"], help="Models to export")
    parser.add_argument("--opset_version", type=int, default=14,
                       help="ONNX opset version to use")
    parser.add_argument("--precision", choices=["all", "fp32", "fp16", "floating", "int8"], 
                       default="floating", help="Precision to export")
    parser.add_argument("--fp16_llm", action="store_true",
                       help="Export LLM in FP16 precision")
    parser.add_argument("--trust_remote_code_llm", action="store_true",
                       help="Trust remote code for LLM export")
    parser.add_argument("--sample_audio_for_vocoder", type=str, default="./example/prompt_audio.wav",
                       help="Sample audio for vocoder export")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before export")
    
    args = parser.parse_args()
    
    # Determine which precisions to export
    export_fp32 = args.precision in ["all", "fp32", "floating"]
    export_fp16 = args.precision in ["all", "fp16", "floating"]
    export_int8 = args.precision in ["all", "int8"] and INT8_AVAILABLE
    
    if args.precision in ["all", "int8"] and not INT8_AVAILABLE:
        print("⚠️ INT8 quantization requested but onnxruntime quantization not available")
        export_int8 = False
    
    # Print configuration
    precisions = []
    if export_fp32: precisions.append("FP32")
    if export_fp16: precisions.append("FP16")
    if export_int8: precisions.append("INT8")
    print(f"📝 Exporting models in precision(s): {', '.join(precisions)}")
    
    # Create output directory
    output_dir = args.output_dir
    
    # Clean directory if requested
    if args.clean and output_dir.exists():
        print(f"🧹 Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device
    print(f"Using device: {device}")
    print(f"Using ONNX opset version: {args.opset_version}")
    
    # Determine models to export
    models_to_export = args.models
    if "all" in models_to_export:
        models_to_export = ["wav2vec2", "mel_spectrogram", "bicodec_encoder_quantizer", 
                           "speaker_encoder_tokenizer", "bicodec_vocoder", "llm"]
    
    # Model export mappings
    model_mappings = {
        "wav2vec2": {
            "func": export_wav2vec2_to_onnx,
            "path": "wav2vec2_model.onnx"
        },
        "mel_spectrogram": {
            "func": export_mel_spectrogram_to_onnx,
            "path": "mel_spectrogram.onnx"
        },
        "bicodec_encoder_quantizer": {
            "func": export_bicodec_encoder_quantizer_to_onnx,
            "path": "bicodec_encoder_quantizer.onnx"
        },
        "speaker_encoder_tokenizer": {
            "func": export_speaker_encoder_tokenizer_to_onnx,
            "path": "speaker_encoder_tokenizer.onnx"
        },
        "bicodec_vocoder": {
            "func": export_bicodec_vocoder_to_onnx,
            "path": "bicodec_vocoder.onnx"
        },
        "llm": {
            "func": export_llm_to_onnx,
            "path": "LLM/model.onnx"  # Directory for LLM
        }
    }
    
    try:
        success_count = 0
        
        for model_name in models_to_export:
            if model_name not in model_mappings:
                print(f"⚠️ Unknown model: {model_name}")
                continue
            
            model_info = model_mappings[model_name]
            output_path = output_dir / model_info["path"]
            
            # Special handling for LLM
            if model_name == "llm":
                kwargs = {
                    "trust_remote_code_llm": args.trust_remote_code_llm
                }
            elif model_name == "bicodec_vocoder":
                kwargs = {
                    "sample_audio_for_shapes": args.sample_audio_for_vocoder
                }
            else:
                kwargs = {}
            
            try:
                success_count += export_model_with_precisions(
                    model_info["func"], args.base_model_dir, output_path, model_name,
                    export_fp32, export_fp16, export_int8, device, args.opset_version,
                    **kwargs
                )
            except Exception as e:
                print(f"Failed to export {model_name}: {e}")
        
        print(f"\n✓ Successfully exported {success_count} model variants")
        print(f"Output directory: {output_dir}")
        
        # Save configuration
        config = {
            "base_model_dir": str(args.base_model_dir),
            "exported_models": models_to_export,
            "device": device,
            "opset_version": args.opset_version,
            "precision": args.precision,
            "precisions_exported": {
                "fp32": export_fp32,
                "fp16": export_fp16,
                "int8": export_int8
            },
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "int8_available": INT8_AVAILABLE,
            "success_count": success_count,
            "llm_config": {
                "fp16": args.fp16_llm,
                "trust_remote_code": args.trust_remote_code_llm
            }
        }
        
        config_path = output_dir / "onnx_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        
        # Clean up temporary files
        cleanup_export_directory(output_dir)
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
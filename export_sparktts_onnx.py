#!/usr/bin/env python3
"""
Consolidated ONNX Export Script for Spark-TTS Models
This script exports all Spark-TTS models to ONNX format with multiple precision support and optimizations.

INT8 Quantization Improvements:
- For LLM models, uses realistic token IDs within vocabulary range instead of zeros
- Generates multiple calibration samples with diverse token patterns
- Supports custom calibration texts for better quality
- Uses model-specific quantization settings (entropy method for LLMs)
- Applies conservative quantization options for better accuracy

Usage Examples:

Basic export with INT8:
    python export_sparktts_onnx.py --precision int8 --models llm

Export with custom calibration texts:
    python export_sparktts_onnx.py --precision int8 --models llm \
        --calibration_texts "Hello world" "The quick brown fox" "Machine learning is fascinating"

Export with calibration texts from file:
    python export_sparktts_onnx.py --precision int8 --models llm \
        --calibration_texts_file calibration_texts.txt

For best INT8 quality, provide representative text samples that match your use case.
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
    """
    Wrapper for the SpeakerEncoder's tokenize method for ONNX export.
    This module takes Mel spectrograms and outputs global speaker token IDs.
    """
    def __init__(self, speaker_encoder_model):
        super().__init__()
        self.speaker_encoder = speaker_encoder_model

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mels (torch.Tensor): Input Mel spectrogram tensor.
                                 Expected shape: (B, T_mel, D_mel_features),
                                 e.g., (1, 200, 128) for ECAPA-TDNN.

        Returns:
            torch.Tensor: Global speaker token IDs.
                          Shape depends on the speaker encoder's quantizer configuration,
                          e.g., (B, N_quant_levels, T_quant_tokens) or (B, T_quant_tokens_flat).
        """
        # The SpeakerEncoder.tokenize method should internally handle the onnx_export_mode
        # to ensure its quantizer (FSQ) uses export-friendly operations.
        return self.speaker_encoder.tokenize(mels, onnx_export_mode=True)

class BiCodecVocoderWrapper(nn.Module):
    """
    Wrapper for BiCodec's vocoding components (detokenizer, prenet, generator) for ONNX export.
    Takes semantic tokens and global speaker tokens, and outputs a waveform.
    """
    def __init__(self, bicodec_model):
        super().__init__()
        self.quantizer = bicodec_model.quantizer
        self.speaker_encoder = bicodec_model.speaker_encoder
        self.prenet = bicodec_model.prenet
        self.decoder = bicodec_model.decoder  # This is the WaveGenerator (vocoder)

    def forward(self, semantic_tokens: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        """
        Performs vocoding from semantic and global speaker tokens to a waveform.

        Args:
            semantic_tokens (torch.Tensor): Semantic token IDs.
                Expected shape: (B, N_quant_semantic, T_semantic) or (B, T_semantic_flat).
            global_tokens (torch.Tensor): Global speaker token IDs.
                Expected shape: (B, N_quant_global, T_global).

        Returns:
            torch.Tensor: Reconstructed audio waveform.
                          Shape: (B, 1, T_audio) or (B, T_audio).
        """
        # 1. Detokenize semantic tokens to get quantized embeddings `z_q`
        z_q = self.quantizer.detokenize(semantic_tokens)

        # 2. Detokenize global speaker tokens to get speaker embedding `d_vector`
        d_vector = self.speaker_encoder.detokenize(global_tokens, onnx_export_mode=True)
        
        # 3. Pass `z_q` and `d_vector` through the prenet
        x_prenet = self.prenet(z_q, d_vector)
        
        # 4. Condition prenet output with speaker embedding before the decoder
        # `d_vector` [B, D_spk] needs to be broadcastable with `x_prenet` [B, D_prenet, T_prenet].
        if d_vector.ndim == 2 and x_prenet.ndim == 3:
            if d_vector.shape[0] == x_prenet.shape[0] and d_vector.shape[1] == x_prenet.shape[1]:
                condition_vector = d_vector.unsqueeze(-1)  # Makes d_vector [B, D_spk, 1]
            else:
                raise ValueError(
                    f"Shape mismatch for conditioning: d_vector {d_vector.shape}, x_prenet {x_prenet.shape}. "
                    f"Channel dimensions (dim 1) must match."
                )
        elif d_vector.ndim == x_prenet.ndim and d_vector.shape == x_prenet.shape:
            condition_vector = d_vector
        else:
            raise ValueError(
                f"Unexpected dimensions for conditioning: d_vector {d_vector.ndim}D {d_vector.shape}, "
                f"x_prenet {x_prenet.ndim}D {x_prenet.shape}. Cannot broadcast."
            )

        x_conditioned = x_prenet + condition_vector
        
        # 5. Generate waveform using the decoder (WaveGenerator)
        wav_recon = self.decoder(x_conditioned)
        return wav_recon

def get_dummy_tokens_from_sample_audio(
    model_base_dir: Path, 
    device: torch.device, 
    sample_audio_path_str: str,
    target_sample_rate: int = 16000,  # Default SR for dummy audio if created
    duration_sec: int = 1  # Duration for dummy audio if created
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates dummy semantic and global tokens by tokenizing a sample audio file.
    This ensures the dummy tokens have realistic shapes for ONNX export.
    If the sample audio path doesn't exist, a silent dummy WAV is created.

    Args:
        model_base_dir: The base directory of the SparkTTS model 
                        (e.g., ./pretrained_models/Spark-TTS-0.5B), which is the parent of BiCodec, wav2vec2 etc.
        device: The torch device to use for tokenization.
        sample_audio_path_str: Path to the sample WAV file.
        target_sample_rate: Sample rate for the dummy audio if it needs to be created.
        duration_sec: Duration in seconds for the dummy audio if it needs to be created.

    Returns:
        A tuple (dummy_semantic_tokens, dummy_global_tokens).
    """
    print(f"Initializing BiCodecTokenizer from base model dir: {model_base_dir} to get sample token shapes")
    try:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        tokenizer = BiCodecTokenizer(model_dir=model_base_dir, device=device)
    except Exception as e:
        print(f"✗ Failed to initialize BiCodecTokenizer from {model_base_dir}: {e}")
        raise
    
    sample_audio_path = Path(sample_audio_path_str)
    if not sample_audio_path.exists():
        print(f"⚠️ Sample audio {sample_audio_path} not found. Creating a temporary silent dummy WAV file")
        try:
            import soundfile as sf
            dummy_wav_data = np.zeros(target_sample_rate * duration_sec, dtype=np.float32)
            sample_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(sample_audio_path, dummy_wav_data, target_sample_rate)
            print(f"Dummy silent audio created at {sample_audio_path}")
        except ImportError:
            print("✗ `soundfile` library is not installed. Cannot create dummy audio. Please install it or provide a valid sample audio path")
            raise
        except Exception as e:
            print(f"✗ Failed to create dummy audio file at {sample_audio_path}: {e}")
            raise

    print(f"Tokenizing sample audio: {sample_audio_path} to get token shapes")
    try:
        # BiCodecTokenizer.tokenize returns: global_tokens, semantic_tokens
        real_global_tokens, real_semantic_tokens = tokenizer.tokenize(str(sample_audio_path))
    except Exception as e:
        print(f"✗ Failed to tokenize sample audio {sample_audio_path} with BiCodecTokenizer: {e}")
        raise

    print(f"Shape of real_global_tokens from sample: {real_global_tokens.shape}, dtype: {real_global_tokens.dtype}")
    print(f"Shape of real_semantic_tokens from sample: {real_semantic_tokens.shape}, dtype: {real_semantic_tokens.dtype}")

    # Create random integer tensors with the same shapes and type as the real tokens.
    dummy_global_tokens = torch.randint_like(real_global_tokens, low=0, high=100)
    dummy_semantic_tokens = torch.randint_like(real_semantic_tokens, low=0, high=100)
    
    print(f"Shape of created dummy_global_tokens: {dummy_global_tokens.shape}, dtype: {dummy_global_tokens.dtype}")
    print(f"Shape of created dummy_semantic_tokens: {dummy_semantic_tokens.shape}, dtype: {dummy_semantic_tokens.dtype}")

    return dummy_semantic_tokens, dummy_global_tokens

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
        # Load the pre-trained BiCodec model to get the SpeakerEncoder
        bicodec_model_subdir = Path(model_dir) / "BiCodec"
        print(f"Loading BiCodec model from: {bicodec_model_subdir} to access SpeakerEncoder")
        if not bicodec_model_subdir.exists():
            print(f"✗ BiCodec model directory not found at {bicodec_model_subdir}")
            return False
        
        # Load BiCodec model
        bicodec_model = BiCodec.load_from_checkpoint(model_dir=bicodec_model_subdir, device=export_device)
        speaker_encoder_module = bicodec_model.speaker_encoder
        speaker_encoder_module.eval()
        print("SpeakerEncoder module extracted and set to eval mode")
        
        # Create ONNX wrapper
        onnx_exportable_tokenizer = SpeakerEncoderTokenizerWrapper(speaker_encoder_module).to(export_device)
        onnx_exportable_tokenizer.eval()
        
        # Create dummy Mel input
        # ECAPA-TDNN expects (B, T, F) where F is feature dimension (e.g., 128 for ECAPA in BiCodec)
        dummy_mel_batch = 1
        dummy_mel_time = 200
        dummy_mel_channels = 128
        dummy_mels = torch.randn(dummy_mel_batch, dummy_mel_time, dummy_mel_channels, device=export_device).contiguous()
        print(f"Using dummy_mels input shape (B, T_mel, D_mel_feat): {dummy_mels.shape}")
        
        # Test forward pass
        print("Performing test forward pass...")
        with torch.no_grad():
            pytorch_output_tokens = onnx_exportable_tokenizer(dummy_mels)
        print(f"PyTorch test pass successful. Output tokens shape: {pytorch_output_tokens.shape}")
        
        if pytorch_output_tokens.numel() == 0:
            print("✗ PyTorch wrapper produced empty tokens")
            return False
        
        # Export to ONNX
        input_names = ["mel_spectrogram"]
        output_names = ["global_tokens"]
        
        dynamic_axes = {
            input_names[0]: {0: 'batch_size', 1: 'mel_time_steps'},
            output_names[0]: {0: 'batch_size'}
        }
        
        # Add dynamic axis for token sequence length if needed
        if pytorch_output_tokens.ndim == 3:  # (B, N_quant_levels, T_quant_tokens)
            dynamic_axes[output_names[0]][2] = 'token_seq_len'
        elif pytorch_output_tokens.ndim == 2:  # (B, T_quant_tokens_flat)
            dynamic_axes[output_names[0]][1] = 'token_seq_len_flat'
        
        print(f"Using dynamic axes: {dynamic_axes}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.onnx.export(
            onnx_exportable_tokenizer,
            dummy_mels,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        
        print(f"Speaker Encoder+Tokenizer exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export Speaker Encoder+Tokenizer: {e}")
        import traceback
        traceback.print_exc()
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
        print("BiCodec model loaded and set to eval mode")
        
        # Create wrapper
        vocoder_wrapper = BiCodecVocoderWrapper(bicodec_model).to(export_device)
        vocoder_wrapper.eval()
        
        # Get dummy tokens from sample audio
        if sample_audio_for_shapes is None:
            sample_audio_for_shapes = "./example/prompt_audio.wav"
        
        print(f"Preparing dummy input tokens using sample audio: {sample_audio_for_shapes}")
        dummy_semantic_tokens, dummy_global_tokens = get_dummy_tokens_from_sample_audio(
            model_base_dir=Path(model_dir),
            device=export_device,
            sample_audio_path_str=sample_audio_for_shapes
        )
        
        # Ensure tokens are on the correct device
        dummy_semantic_tokens = dummy_semantic_tokens.to(export_device)
        dummy_global_tokens = dummy_global_tokens.to(export_device)
        
        print(f"Using dummy_semantic_tokens shape: {dummy_semantic_tokens.shape}")
        print(f"Using dummy_global_tokens shape: {dummy_global_tokens.shape}")
        
        # Test forward pass
        print("Performing test forward pass...")
        with torch.no_grad():
            pytorch_output_waveform = vocoder_wrapper(dummy_semantic_tokens, dummy_global_tokens)
        print(f"PyTorch test pass successful. Output waveform shape: {pytorch_output_waveform.shape}")
        
        if pytorch_output_waveform.numel() == 0:
            print("✗ PyTorch wrapper produced empty waveform")
            return False
        
        # Export to ONNX
        input_names = ["semantic_tokens", "global_tokens"]
        output_names = ["output_waveform"]
        
        dynamic_axes = {}
        
        # Semantic tokens: (B, T_semantic_flat) or (B, N_quant_semantic, T_semantic)
        if dummy_semantic_tokens.ndim == 2:
            dynamic_axes[input_names[0]] = {0: 'batch_size', 1: 'semantic_token_flat_seq_len'}
        elif dummy_semantic_tokens.ndim == 3:
            dynamic_axes[input_names[0]] = {0: 'batch_size', 2: 'semantic_token_seq_len'}
        
        # Global tokens: (B, N_quant_global, T_global)
        if dummy_global_tokens.ndim == 3:
            dynamic_axes[input_names[1]] = {0: 'batch_size', 2: 'global_token_seq_len'}
        elif dummy_global_tokens.ndim == 2:
            dynamic_axes[input_names[1]] = {0: 'batch_size', 1: 'global_feature_len'}
        
        # Output waveform: (B, 1, AudioSeqLen) or (B, AudioSeqLen)
        if pytorch_output_waveform.ndim == 3:
            dynamic_axes[output_names[0]] = {0: 'batch_size', 2: 'audio_sequence_length'}
        elif pytorch_output_waveform.ndim == 2:
            dynamic_axes[output_names[0]] = {0: 'batch_size', 1: 'audio_sequence_length'}
        
        print(f"Using dynamic axes: {dynamic_axes}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.onnx.export(
            vocoder_wrapper,
            (dummy_semantic_tokens, dummy_global_tokens),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        
        print(f"BiCodec Vocoder exported successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to export BiCodec Vocoder: {e}")
        import traceback
        traceback.print_exc()
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

def generate_llm_calibration_data_from_text(model_dir, sample_texts=None, max_samples=10, calibration_texts_file=None):
    """Generate calibration data from real text samples for better LLM quantization"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Load tokenizer
        from transformers import AutoTokenizer
        llm_path = Path(model_dir) / "LLM"
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        # Load calibration texts from file if provided
        if calibration_texts_file and Path(calibration_texts_file).exists():
            try:
                with open(calibration_texts_file, 'r', encoding='utf-8') as f:
                    sample_texts = [line.strip() for line in f.readlines() if line.strip()]
                print(f"✓ Loaded {len(sample_texts)} calibration texts from {calibration_texts_file}")
            except Exception as e:
                print(f"⚠️ Failed to load calibration texts from file: {e}")
                sample_texts = None
        
        # Create SparkTTS-specific prompts using the actual format
        if sample_texts is None:
            # Default texts for TTS
            base_texts = [
                "Hello world, how are you today?",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming our world.",
                "Welcome to the future of artificial intelligence.",
                "This is a test of text-to-speech synthesis.",
                "Natural language processing enables amazing applications.",
                "Speech technology continues to advance rapidly.",
                "Voice assistants are becoming more sophisticated.",
                "Audio generation models produce realistic speech.",
                "Text synthesis creates human-like voices."
            ]
        else:
            base_texts = sample_texts
        
        # Create SparkTTS-formatted prompts
        sparktts_prompts = []
        
        # Task token for TTS (from added_tokens.json: "<|task_tts|>": 165137)
        task_token = "<|task_tts|>"
        start_content = "<|start_content|>"
        end_content = "<|end_content|>"
        start_global = "<|start_global_token|>"
        end_global = "<|end_global_token|>"
        
        for i, text in enumerate(base_texts[:max_samples]):
            # Create realistic global tokens (BiCodec tokens typically range 0-4095)
            # Generate varied token sequences to simulate real audio tokens
            num_global_tokens = np.random.randint(10, 50)  # Realistic sequence length
            global_token_ids = np.random.randint(0, 4096, num_global_tokens)
            global_tokens = "".join([f"<|bicodec_global_{token_id}|>" for token_id in global_token_ids])
            
            # Create SparkTTS prompt format
            # Format: <|task_tts|><|start_content|>TEXT<|end_content|><|start_global_token|>TOKENS<|end_global_token|>
            prompt = f"{task_token}{start_content}{text}{end_content}{start_global}{global_tokens}{end_global}"
            sparktts_prompts.append(prompt)
        
        print(f"✓ Created {len(sparktts_prompts)} SparkTTS-formatted prompts")
        print(f"Example prompt length: {len(sparktts_prompts[0])} characters")
        
        # Tokenize all prompts
        calibration_samples = []
        for i, prompt in enumerate(sparktts_prompts):
            # Tokenize with appropriate settings for the model
            tokens = tokenizer(
                prompt,
                max_length=512,  # Use larger context length for TTS
                truncation=True,
                padding='max_length',
                return_tensors='np'
            )
            
            # Prepare sample data with all expected inputs
            sample_data = {}
            
            # Main inputs
            sample_data['input_ids'] = tokens['input_ids'].astype(np.int64)
            
            if 'attention_mask' in tokens:
                sample_data['attention_mask'] = tokens['attention_mask'].astype(np.int64)
            
            # Position IDs (sequential positions)
            seq_len = tokens['input_ids'].shape[1]
            sample_data['position_ids'] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            
            # KV cache inputs (initialize as zeros - typical for first inference)
            # Try to get model config for accurate dimensions
            try:
                import json
                config_path = llm_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    num_layers = config.get('num_hidden_layers', 24)
                    num_attention_heads = config.get('num_attention_heads', 32)
                    num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)  # GQA support
                    hidden_size = config.get('hidden_size', 4096)
                    head_dim = hidden_size // num_attention_heads
                    print(f"✓ Detected model config: {num_layers} layers, {num_attention_heads} heads, {num_key_value_heads} kv_heads, {head_dim} head_dim")
                else:
                    # Fallback values
                    num_layers = 24
                    num_attention_heads = 32
                    num_key_value_heads = 32  # Assume MHA if no config
                    head_dim = 128
                    print("⚠️ No config found, using fallback dimensions")
            except Exception as e:
                # Fallback values
                num_layers = 24
                num_attention_heads = 32
                num_key_value_heads = 32  # Assume MHA if no config
                head_dim = 128
                print(f"⚠️ Failed to read config: {e}, using fallback dimensions")
            
            batch_size = 1
            kv_seq_len = 0   # Empty cache for initial inference (proper sequence alignment)
            
            for layer_idx in range(num_layers):
                # Each layer has key and value caches
                key_name = f'past_key_values.{layer_idx}.key'
                value_name = f'past_key_values.{layer_idx}.value'
                
                # Shape: [batch_size, num_key_value_heads, seq_len, head_dim]
                # For GQA models, use num_key_value_heads instead of num_attention_heads
                # Use proper empty tensors for initial inference
                sample_data[key_name] = np.zeros((batch_size, num_key_value_heads, kv_seq_len, head_dim), dtype=np.float32)
                sample_data[value_name] = np.zeros((batch_size, num_key_value_heads, kv_seq_len, head_dim), dtype=np.float32)
            
            calibration_samples.append(sample_data)
            print(f"Generated calibration sample {i+1}/{len(sparktts_prompts)} with {len(sample_data)} inputs")
        
        return calibration_samples
        
    except Exception as e:
        print(f"Failed to generate calibration data from text: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type="general", calibration_texts=None):
    """Convert FP32 ONNX model to INT8 using static quantization with QDQ format"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
    
    try:
        print(f"Converting {fp32_model_path} to INT8 using static QDQ quantization...")
        
        class SmartCalibrationDataReader(CalibrationDataReader):
            def __init__(self, model_path, model_type="general"):
                self.model_path = model_path
                self.model_type = model_type
                self.data_generated = 0
                self.max_samples = 10  # Generate multiple calibration samples
                
                # Check if we have real calibration data from text samples
                self.real_calibration_data = calibration_texts
                if self.real_calibration_data:
                    print(f"✓ Using {len(self.real_calibration_data)} real calibration samples")
                    self.max_samples = len(self.real_calibration_data)
                else:
                    print("⚠️ No real calibration data available, using synthetic data")
                
                # Load model to get input shapes and types (only needed for synthetic data)
                if not self.real_calibration_data:
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
                                elif model_type == "llm":
                                    # For LLM, use a reasonable sequence length
                                    shape.append(512)  # Larger context for TTS
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
                    
                    # Get vocab size for LLM models
                    if model_type == "llm":
                        try:
                            # Try to get vocab size from the model's config
                            model_dir = Path(model_path).parent
                            config_path = model_dir / "config.json"
                            if config_path.exists():
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                    self.vocab_size = config.get('vocab_size', 32000)
                            else:
                                self.vocab_size = 32000  # Default vocab size for most models
                        except:
                            self.vocab_size = 32000
                        
                        print(f"Using vocab_size={self.vocab_size} for LLM calibration")
                
            def get_next(self):
                if self.data_generated >= self.max_samples:
                    return None
                
                # Use real calibration data if available
                if self.real_calibration_data:
                    sample_data = self.real_calibration_data[self.data_generated].copy()  # Make a copy to avoid modifying original
                    self.data_generated += 1
                    print(f"Using real calibration sample {self.data_generated}/{self.max_samples}")
                    
                    # Robust preprocessing for real calibration data to prevent extreme values
                    for name, data in sample_data.items():
                        if data.dtype in [np.float32, np.float16]:
                            # Handle NaN and infinity values
                            data_clean = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
                            # Clip to very conservative range for entropy method
                            sample_data[name] = np.clip(data_clean, -3.0, 3.0).astype(data.dtype)
                        elif data.dtype in [np.int64, np.int32]:
                            if 'token' in name.lower() or 'input_ids' in name.lower():
                                # Keep token IDs within reasonable vocab range
                                sample_data[name] = np.clip(data, 0, 166000).astype(data.dtype)
                            elif 'position' in name.lower():
                                # Keep position IDs reasonable
                                sample_data[name] = np.clip(data, 0, 32768).astype(data.dtype)
                            elif 'attention_mask' in name.lower():
                                # Ensure attention mask is 0 or 1
                                sample_data[name] = np.clip(data, 0, 1).astype(data.dtype)
                            else:
                                # General integer bounds
                                sample_data[name] = np.clip(data, 0, 100000).astype(data.dtype)
                        
                        # Final validation to ensure no extreme values remain
                        if np.any(np.isinf(sample_data[name])) or np.any(np.isnan(sample_data[name])):
                            print(f"⚠️ Warning: Found inf/nan in {name}, replacing with zeros")
                            sample_data[name] = np.nan_to_num(sample_data[name], nan=0.0, posinf=1.0, neginf=0.0)
                    
                    return sample_data
                
                # Fall back to synthetic data generation
                calibration_data = {}
                for name, shape in self.input_shapes.items():
                    dtype = self.input_types[name]
                    
                    if dtype == np.int64 or dtype == np.int32:
                        if self.model_type == "llm":
                            # Generate realistic token IDs within vocab range
                            # Create diverse patterns: common tokens, rare tokens, mixed sequences
                            if self.data_generated == 0:
                                # First sample: common tokens (lower vocab range)
                                token_ids = np.random.randint(1, min(1000, self.vocab_size // 4), 
                                                           size=shape, dtype=dtype)
                            elif self.data_generated == 1:
                                # Second sample: mixed range
                                token_ids = np.random.randint(1, self.vocab_size // 2, 
                                                           size=shape, dtype=dtype)
                            elif self.data_generated == 2:
                                # Third sample: full vocab range
                                token_ids = np.random.randint(1, self.vocab_size, 
                                                           size=shape, dtype=dtype)
                            else:
                                # Remaining samples: varied patterns
                                if self.data_generated % 2 == 0:
                                    # Even samples: focus on common tokens
                                    token_ids = np.random.randint(1, min(5000, self.vocab_size), 
                                                               size=shape, dtype=dtype)
                                else:
                                    # Odd samples: broader range
                                    token_ids = np.random.randint(1, self.vocab_size, 
                                                               size=shape, dtype=dtype)
                            
                            # Ensure no zero tokens except potentially at the end (padding)
                            # Add some realistic patterns
                            if len(shape) >= 2:  # Batch dimension exists
                                for batch_idx in range(shape[0]):
                                    seq_len = shape[1] if len(shape) > 1 else shape[0]
                                    # Sometimes add padding tokens at the end
                                    if np.random.random() < 0.3:  # 30% chance
                                        pad_start = np.random.randint(seq_len // 2, seq_len)
                                        if len(shape) > 1:
                                            token_ids[batch_idx, pad_start:] = 0
                                        else:
                                            token_ids[pad_start:] = 0
                            
                            calibration_data[name] = token_ids
                        else:
                            # For non-LLM models, generate appropriate integer values
                            if 'token' in name.lower():
                                # Token-like inputs: use reasonable range
                                calibration_data[name] = np.random.randint(0, 1000, size=shape, dtype=dtype)
                            else:
                                # Other integer inputs: use small positive values
                                calibration_data[name] = np.random.randint(0, 10, size=shape, dtype=dtype)
                    else:
                        # Float inputs: generate more realistic distributions
                        if self.model_type == "llm":
                            # For LLM embeddings/features: use conservative normal distribution
                            calibration_data[name] = np.random.normal(0, 0.3, size=shape).astype(dtype)
                        elif 'audio' in name.lower() or 'wav' in name.lower():
                            # Audio data: use realistic audio range
                            calibration_data[name] = np.random.normal(0, 0.1, size=shape).astype(dtype)
                        elif 'mel' in name.lower() or 'spectrogram' in name.lower():
                            # Mel spectrogram: use conservative log-scale values
                            calibration_data[name] = np.clip(np.random.exponential(0.5, size=shape), 0, 5.0).astype(dtype)
                        else:
                            # General float inputs: use conservative normal
                            calibration_data[name] = np.random.normal(0, 0.5, size=shape).astype(dtype)
                
                self.data_generated += 1
                print(f"Generated synthetic calibration sample {self.data_generated}/{self.max_samples}")
                
                # Robust clipping to prevent histogram binning issues (same as real data)
                for name, data in calibration_data.items():
                    if data.dtype in [np.float32, np.float16]:
                        # Handle any potential NaN/inf and clip to conservative range
                        data_clean = np.nan_to_num(data, nan=0.0, posinf=3.0, neginf=-3.0)
                        calibration_data[name] = np.clip(data_clean, -3.0, 3.0).astype(data.dtype)
                    elif data.dtype in [np.int64, np.int32]:
                        # Ensure integer data is within reasonable bounds
                        if self.model_type == "llm" and 'token' in name.lower():
                            # Keep token IDs within vocab range
                            vocab_size = getattr(self, 'vocab_size', 32000)
                            calibration_data[name] = np.clip(data, 0, vocab_size - 1).astype(data.dtype)
                        else:
                            # General integer bounds
                            calibration_data[name] = np.clip(data, 0, 100000).astype(data.dtype)
                    
                    # Final validation for synthetic data too
                    if np.any(np.isinf(calibration_data[name])) or np.any(np.isnan(calibration_data[name])):
                        print(f"⚠️ Warning: Found inf/nan in synthetic {name}, replacing with zeros")
                        calibration_data[name] = np.nan_to_num(calibration_data[name], nan=0.0, posinf=1.0, neginf=0.0)
                
                return calibration_data
        
        calibration_reader = SmartCalibrationDataReader(fp32_model_path, model_type)
        
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100  # > 100MB threshold
        
        # Use more appropriate quantization settings for LLMs
        if model_type == "llm":
            # For LLMs, be more conservative with quantization
            try:
                # Try Entropy method first, fall back to MinMax if it fails
                calibration_method = CalibrationMethod.Entropy
                try:
                    quantize_static(
                        model_input=fp32_model_path,
                        model_output=int8_model_path,
                        calibration_data_reader=calibration_reader,
                        quant_format=QuantFormat.QDQ,
                        weight_type=QuantType.QInt8,
                        activation_type=QuantType.QUInt8,  # Use unsigned for activations
                        use_external_data_format=use_external_data,
                        calibrate_method=calibration_method,
                        extra_options={
                            'WeightSymmetric': True,
                            'ActivationSymmetric': False,
                            'EnableSubgraph': False,  # Disable subgraph optimization for LLMs
                            'ForceQuantizeNoInputCheck': False,
                            'MatMulConstBOnly': True,  # Only quantize constant B matrix in MatMul
                        }
                    )
                except Exception as entropy_error:
                    # Check if this is a histogram/calibration issue that MinMax can handle
                    error_str = str(entropy_error).lower()
                    histogram_errors = ["bins", "histogram", "range", "overflow", "invalid value"]
                    is_histogram_error = any(err in error_str for err in histogram_errors)
                    
                    if is_histogram_error:
                        print(f"⚠️ Entropy calibration failed with histogram issue: {entropy_error}")
                        print("   Falling back to MinMax method which is more robust for this data...")
                        # Recreate calibration reader since it may have been consumed
                        calibration_reader = SmartCalibrationDataReader(fp32_model_path, model_type)
                        quantize_static(
                            model_input=fp32_model_path,
                            model_output=int8_model_path,
                            calibration_data_reader=calibration_reader,
                            quant_format=QuantFormat.QDQ,
                            weight_type=QuantType.QInt8,
                            activation_type=QuantType.QUInt8,
                            use_external_data_format=use_external_data,
                            calibrate_method=CalibrationMethod.MinMax,  # More robust fallback
                            extra_options={
                                'WeightSymmetric': True,
                                'ActivationSymmetric': False,
                                'EnableSubgraph': False,
                                'ForceQuantizeNoInputCheck': False,
                                'MatMulConstBOnly': True,
                            }
                        )
                    else:
                        # For other errors (shape mismatches, etc.), let the outer handler deal with it
                        raise entropy_error
            except Exception as e:
                if "broadcast" in str(e).lower() or "shape" in str(e).lower():
                    print(f"⚠️ Shape mismatch in static quantization: {e}")
                    print("This is often due to KV cache dimension mismatches. Will fall back to dynamic quantization.")
                    raise e  # Re-raise to trigger fallback
                else:
                    raise e
        else:
            # For other models, use standard settings
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

def convert_model_to_int8(fp32_model_path, int8_model_path, model_type="general", model_dir=None):
    """Convert FP32 ONNX model to INT8"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False
    
    try:
        # For LLMs, try to generate better calibration data from text
        calibration_texts = None
        if model_type == "llm" and model_dir:
            print("Attempting to generate realistic calibration data from text samples...")
            # Use custom calibration texts if provided, otherwise use defaults
            custom_texts = getattr(convert_model_to_int8, '_custom_calibration_texts', None)
            custom_texts_file = getattr(convert_model_to_int8, '_custom_calibration_texts_file', None)
            calibration_texts = generate_llm_calibration_data_from_text(
                model_dir, 
                sample_texts=custom_texts, 
                calibration_texts_file=custom_texts_file
            )
            if calibration_texts:
                print(f"✓ Generated {len(calibration_texts)} calibration samples from text")
            else:
                print("⚠️ Failed to generate text-based calibration data, using synthetic data")
        
        # Try static QDQ quantization first
        if convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type, calibration_texts):
            return True
        
        print(f"Static QDQ quantization failed, trying dynamic quantization...")
        
        # Fallback to dynamic quantization
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100
        
        # For LLMs, use more conservative dynamic quantization
        if model_type == "llm":
            print("Using conservative dynamic quantization for LLM...")
            quantize_dynamic(
                model_input=fp32_model_path,
                model_output=int8_model_path,
                weight_type=QuantType.QInt8,
                use_external_data_format=use_external_data,
                extra_options={
                    'MatMulConstBOnly': True,  # Only quantize constant B matrix in MatMul
                    'WeightSymmetric': True,
                }
            )
        else:
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
    """Verify the exported ONNX model with multiple providers including CoreML"""
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
        
        # Test multiple providers in order of preference
        providers_to_test = []
        
        # Check for CoreML compatibility before attempting
        coreml_compatible = True
        try:
            # Load model to check for CoreML limitations
            onnx_model = onnx.load(onnx_path, load_external_data=has_data)
            
            # Check for large vocabulary or dimensions that exceed CoreML limits
            for initializer in onnx_model.graph.initializer:
                for dim in initializer.dims:
                    if dim > 16384:  # CoreML dimension limit
                        print(f"⚠️ CoreML incompatible: {initializer.name} dimension {dim} > 16384")
                        coreml_compatible = True  #False
                        break
                if not coreml_compatible:
                    break
            
            # Check for zero-dimension shapes in graph inputs
            for input_info in onnx_model.graph.input:
                if input_info.type.tensor_type.shape:
                    for dim in input_info.type.tensor_type.shape.dim:
                        if hasattr(dim, 'dim_value') and dim.dim_value == 0:
                            print(f"⚠️ CoreML incompatible: {input_info.name} has zero dimension")
                            coreml_compatible = False
                            break
                    if not coreml_compatible:
                        break
                        
            # For large models, assume incompatible due to vocabulary size
            if file_size > 500 * 1024 * 1024:  # > 500MB likely has large vocab
                print("⚠️ CoreML incompatible: Large model likely has vocabulary > 16K dimensions")
                coreml_compatible = False
                
        except Exception as e:
            print(f"⚠️ Could not check CoreML compatibility: {e}")
            coreml_compatible = False
        
        # CoreML provider with optimized settings for macOS
        try:
            import platform
            if platform.system() == "Darwin" and coreml_compatible:  # macOS and compatible
                coreml_options = {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "CPUAndGPU", 
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "1",
                }
                providers_to_test.append(("CoreMLExecutionProvider", coreml_options))
            elif platform.system() == "Darwin" and not coreml_compatible:
                print("⚠️ CoreML provider skipped: Model has large vocabulary (>16K) or zero-dimension shapes")
                print("  Consider using CPU or CUDA providers for large language models")
        except Exception as e:
            print(f"⚠️ CoreML provider not available: {e}")
        
        # CUDA provider if available
        if torch.cuda.is_available():
            providers_to_test.append("CUDAExecutionProvider")
        
        # CPU provider as fallback
        providers_to_test.append("CPUExecutionProvider")
        
        # Test each provider
        successful_providers = []
        for provider_config in providers_to_test:
            try:
                if isinstance(provider_config, tuple):
                    provider_name, provider_options = provider_config
                    print(f"Testing {provider_name} with options: {provider_options}")
                    session = ort.InferenceSession(onnx_path, providers=[(provider_name, provider_options)])
                else:
                    provider_name = provider_config
                    print(f"Testing {provider_name}")
                    session = ort.InferenceSession(onnx_path, providers=[provider_name])
                
                # Get actual provider used
                actual_provider = session.get_providers()[0]
                successful_providers.append(actual_provider)
                print(f"✓ {provider_name} successfully loaded model")
                print(f"  Actual provider: {actual_provider}")
                
                # Only test the first successful provider for basic validation
                if len(successful_providers) == 1:
                    print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
                    print(f"  Output names: {[out.name for out in session.get_outputs()]}")
                
                break  # Use first successful provider
                
            except Exception as e:
                print(f"✗ {provider_name if isinstance(provider_config, str) else provider_config[0]} failed: {e}")
                continue
        
        if successful_providers:
            print(f"✓ ONNX model {onnx_path} is valid")
            print(f"✓ Successfully tested providers: {successful_providers}")
            return True
        else:
            print(f"✗ No providers could load the model")
            return False
        
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
    model_type = "llm" if model_type == "model" else model_type
    
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
                        if convert_model_to_int8(fp32_path, int8_path, model_type, model_dir):
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
    parser.add_argument("--output_dir", type=Path, default=Path("./onnx_models/SparkTTS"),
                       help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"],
                       help="Device to use for export")
    parser.add_argument("--models", nargs="+", 
                       choices=["wav2vec2", "mel_spectrogram", "bicodec_encoder_quantizer", 
                               "speaker_encoder_tokenizer", "bicodec_vocoder", "llm", "all"],
                       default=["all"], help="Models to export")
    parser.add_argument("--opset_version", type=int, default=18,
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
    parser.add_argument("--calibration_texts", type=str, nargs="+",
                       help="Custom text samples for LLM INT8 calibration (improves quality)")
    parser.add_argument("--calibration_texts_file", type=str,
                       help="Path to file containing calibration texts (one per line)")
    
    args = parser.parse_args()
    
    # Determine which precisions to export
    export_fp32 = args.precision in ["all", "fp32", "floating"]
    export_fp16 = args.precision in ["all", "fp16", "floating"]
    export_int8 = args.precision in ["all", "int8"] and INT8_AVAILABLE
    
    if args.precision in ["all", "int8"] and not INT8_AVAILABLE:
        print("⚠️ INT8 quantization requested but onnxruntime quantization not available")
        export_int8 = False
    
    # Set calibration texts for INT8 quantization
    if export_int8:
        # Store calibration texts in the function for access during quantization
        convert_model_to_int8._custom_calibration_texts = args.calibration_texts
        convert_model_to_int8._custom_calibration_texts_file = args.calibration_texts_file
        
        if args.calibration_texts:
            print(f"📝 Using {len(args.calibration_texts)} custom calibration texts for INT8 quantization")
        elif args.calibration_texts_file:
            print(f"📝 Using calibration texts from file: {args.calibration_texts_file}")
        else:
            print("📝 Using default calibration texts for INT8 quantization")
    
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
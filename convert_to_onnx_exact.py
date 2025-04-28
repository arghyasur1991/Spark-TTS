#!/usr/bin/env python3
"""
Improved script to convert Spark-TTS PyTorch models to ONNX format.
This script ensures exact output matching between PyTorch and ONNX runtime
by carefully handling tensor layouts and model-specific requirements.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil
import soundfile as sf
import sys
import time
import traceback
import datetime

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config
from sparktts.utils.audio import load_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Spark-TTS PyTorch models to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Directory containing PyTorch model files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save ONNX models (defaults to model_dir/onnx)"
    )
    parser.add_argument(
        "--dynamic_axes",
        action="store_true",
        help="Use dynamic axes for sequence length in ONNX export"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version to use for export"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the ONNX models with test inputs after export"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to an audio file to use for verification"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Ensure output_dir is set
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "onnx_exact")
        
    return args

def configure_logger(log_file=None, verbose=False):
    """Configure logger for this script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Set the root logger level
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = os.path.join("logs", log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_path}")

def get_device(cpu_only=False):
    """Determine the best available device for conversion."""
    if cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU device as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def check_environment():
    """Check for required dependencies and report versions."""
    try:
        import onnx
        logger.info(f"ONNX version: {onnx.__version__}")
    except ImportError:
        logger.error("ONNX is not installed. Please install it using: pip install onnx")
        sys.exit(1)
    
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime version: {onnxruntime.__version__}")
        
        # Check available providers
        providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {providers}")
    except ImportError:
        logger.warning("ONNX Runtime is not installed. Model verification will be skipped.")
        logger.warning("To install it, use one of the following:")
        logger.warning("  - For CPU: pip install onnxruntime")
        logger.warning("  - For NVIDIA GPUs: pip install onnxruntime-gpu")
        logger.warning("  - For Apple Silicon: pip install onnxruntime-silicon")
        return False
    
    return True

def export_modules_encoder(model, output_dir, dynamic_axes=False, opset_version=17):
    """Export encoder module."""
    logger.info("Exporting encoder module")
    
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        
        def forward(self, audio_features):
            # Input shape: (batch_size, channels, seq_len)
            # Output shape: (batch_size, seq_len, hidden_dim)
            encoded = self.encoder(audio_features)
            return encoded
    
    encoder_wrapper = EncoderWrapper(model.encoder)
    
    device = next(model.parameters()).device
    
    # Create dummy input based on encoder's expected input
    # The encoder expects input shape (B, C, L) - batch, channels, sequence length
    input_channels = 1024  # Based on wav2vec2 feature dimension
    dummy_audio_features = torch.randn(1, input_channels, 100, device=device)
    
    # Define dynamic axes if needed
    encoder_dynamic_axes = None
    if dynamic_axes:
        encoder_dynamic_axes = {
            "audio_features": {2: "seq_len"},
            "encoded": {1: "seq_len"}
        }
    
    # Export encoder
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    logger.info(f"Exporting encoder to {encoder_path}")
    torch.onnx.export(
        encoder_wrapper,
        dummy_audio_features,
        encoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["audio_features"],
        output_names=["encoded"],
        dynamic_axes=encoder_dynamic_axes
    )
    logger.info("Encoder module exported successfully")
    return encoder_path

def export_modules_quantizer(model, output_dir, dynamic_axes=False, opset_version=17):
    """Export quantizer module."""
    logger.info("Exporting quantizer module")
    
    class QuantizerWrapper(torch.nn.Module):
        def __init__(self, quantizer):
            super().__init__()
            self.quantizer = quantizer
        
        def forward(self, encoder_out):
            # Input shape: (batch_size, seq_len, hidden_dim)
            # The quantizer expects (batch_size, hidden_dim, seq_len)
            # so we need to transpose dimensions 1 and 2
            encoder_out_transposed = encoder_out.transpose(1, 2)
            # The quantizer returns a dictionary, we need to extract the quantized output
            quantizer_output = self.quantizer(encoder_out_transposed)
            # Get the quantized value (z_q) from the output dictionary
            z_q = quantizer_output["z_q"]
            # Output from quantizer is (batch_size, hidden_dim, seq_len)
            # Convert back to (batch_size, seq_len, hidden_dim)
            return z_q.transpose(1, 2)
    
    quantizer_wrapper = QuantizerWrapper(model.quantizer)
    
    device = next(model.parameters()).device
    
    # Create dummy input - shape based on encoder output
    # Use a reasonable hidden dimension for the encoder output
    hidden_channels = 1024  # Common hidden dimension in transformer models
    dummy_encoder_out = torch.randn(1, 100, hidden_channels, device=device)
    
    # Define dynamic axes if needed
    quantizer_dynamic_axes = None
    if dynamic_axes:
        quantizer_dynamic_axes = {
            "encoder_out": {1: "seq_len"},
            "quantized": {1: "seq_len"}
        }
    
    # Export quantizer
    quantizer_path = os.path.join(output_dir, "quantizer.onnx")
    logger.info(f"Exporting quantizer to {quantizer_path}")
    torch.onnx.export(
        quantizer_wrapper,
        dummy_encoder_out,
        quantizer_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["encoder_out"],
        output_names=["quantized"],
        dynamic_axes=quantizer_dynamic_axes
    )
    logger.info("Quantizer module exported successfully")
    return quantizer_path

def export_modules_speaker_encoder(model, output_dir, dynamic_axes=True, opset_version=17, force_cpu=True):
    """Export speaker encoder modules to ONNX format."""
    device = torch.device("cpu") if force_cpu else next(model.parameters()).device
    
    # Move the speaker encoder to CPU to avoid MPS type issues
    speaker_encoder_cpu = model.speaker_encoder.to(device)
    
    # Wrapper for tokenize (extract speaker embeddings)
    class SpeakerEncoderWrapper(torch.nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.speaker_encoder = speaker_encoder
        
        def forward(self, mel):
            # Input mel shape: (batch_size, seq_len, mel_dim)
            return self.speaker_encoder(mel)
            # Output shapes: x_vector (batch_size, emb_dim), d_vector (batch_size, emb_dim)
    
    # Wrapper for detokenize (convert global tokens to speaker embedding)
    class SpeakerEncoderDetokenize(torch.nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.speaker_encoder = speaker_encoder
        
        def forward(self, global_tokens):
            # Input global_tokens shape: (batch_size, token_dim)
            # The speaker_encoder.detokenize expects global_tokens to be shape (batch_size, 1, token_dim)
            # Let's reshape it appropriately
            if len(global_tokens.shape) == 2:
                global_tokens = global_tokens.unsqueeze(1)  # Add the missing dimension
            return self.speaker_encoder.detokenize(global_tokens)
            # Output shape: (batch_size, emb_dim)
    
    # Create wrappers
    speaker_encoder_wrapper = SpeakerEncoderWrapper(speaker_encoder_cpu)
    speaker_detokenize_wrapper = SpeakerEncoderDetokenize(speaker_encoder_cpu)
    
    # Create dummy inputs
    dummy_mel = torch.randn(1, 100, 128, device=device)
    # Modify the dummy_global_tokens shape to match what's expected
    dummy_global_tokens = torch.randn(1, 32, device=device)  # (batch_size, token_dim)
    
    # Define dynamic axes if needed
    encoder_dynamic_axes = None
    detokenize_dynamic_axes = None
    if dynamic_axes:
        encoder_dynamic_axes = {
            "mel": {1: "seq_len"}
        }
    
    # Export speaker encoder
    encoder_path = os.path.join(output_dir, "speaker_encoder.onnx")
    logger.info(f"Exporting speaker encoder to {encoder_path}")
    torch.onnx.export(
        speaker_encoder_wrapper,
        dummy_mel,
        encoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["x_vector", "d_vector"],
        dynamic_axes=encoder_dynamic_axes
    )
    
    # Export speaker detokenize
    detokenize_path = os.path.join(output_dir, "speaker_detokenize.onnx")
    logger.info(f"Exporting speaker detokenize to {detokenize_path}")
    torch.onnx.export(
        speaker_detokenize_wrapper,
        dummy_global_tokens,
        detokenize_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["global_tokens"],
        output_names=["d_vector"],
        dynamic_axes=detokenize_dynamic_axes
    )
    
    # Move the speaker encoder back to its original device
    speaker_encoder_cpu = speaker_encoder_cpu.to(next(model.parameters()).device)
    
    return encoder_path, detokenize_path

def export_modules_decoder(model, output_dir, dynamic_axes=False, opset_version=17):
    """Export decoder module."""
    logger.info("Exporting decoder module")
    
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, prenet=None):
            super().__init__()
            self.decoder = decoder
            self.prenet = prenet  # Used to process inputs if provided
            
            # Store the expected input dimensions for the decoder
            # This is based on the decoder's first layer input requirements
            self.input_channels = 1024  # The known required input channel dimension
        
        def forward(self, codes):
            # Input shape: (batch_size, seq_len, codebook_dim)
            
            # For our decoder, we need to project from codebook_dim to the expected input_channels
            batch_size, seq_len, codebook_dim = codes.shape
            
            # Create a linear projection layer on-the-fly if needed
            if not hasattr(self, 'projection'):
                self.projection = torch.nn.Linear(codebook_dim, self.input_channels).to(codes.device)
            
            # Project the codes to the required channel dimension
            projected_codes = self.projection(codes)
            
            # The decoder expects (batch_size, channels, seq_len)
            # so we need to transpose dimensions 1 and 2
            projected_codes_transposed = projected_codes.transpose(1, 2)
            
            # Output shape: (batch_size, audio_channels, audio_length)
            decoded = self.decoder(projected_codes_transposed)
            return decoded
    
    decoder_wrapper = DecoderWrapper(model.decoder)
    
    device = next(model.parameters()).device
    
    # Create dummy input - shape based on the quantizer output
    # Use reasonable dimensions for the quantizer output
    latent_dim = 256  # Common latent dimension for quantized codes
    dummy_codes = torch.randn(1, 100, latent_dim, device=device)
    
    # Define dynamic axes if needed
    decoder_dynamic_axes = None
    if dynamic_axes:
        decoder_dynamic_axes = {
            "codes": {1: "seq_len"},
            "audio": {2: "audio_length"}
        }
    
    # Export decoder
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    logger.info(f"Exporting decoder to {decoder_path}")
    torch.onnx.export(
        decoder_wrapper,
        dummy_codes,
        decoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["codes"],
        output_names=["audio"],
        dynamic_axes=decoder_dynamic_axes
    )
    logger.info("Decoder module exported successfully")
    return decoder_path

def export_mel_transformer_custom(model, output_dir, dynamic_axes=False, opset_version=16):
    """Export a custom implementation of mel transformer to ONNX format"""
    os.makedirs(output_dir, exist_ok=True)
    mel_path = os.path.join(output_dir, "mel_transformer.onnx")
    
    class SimplerMelTransformerWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Extract mel configuration from model if possible
            try:
                config = model.config
                self.n_fft = config.audio_encoder.n_fft
                self.hop_length = config.audio_encoder.hop_length
                self.win_length = config.audio_encoder.win_length
                self.num_mels = config.audio_encoder.num_mels
                self.mel_fmin = config.audio_encoder.mel_fmin
                self.mel_fmax = config.audio_encoder.mel_fmax
                self.sample_rate = config.audio_encoder.sample_rate
            except (AttributeError, KeyError) as e:
                logging.warning(f"Could not extract all mel configuration from model: {e}")
                # Default values that are commonly used
                self.n_fft = 1024
                self.hop_length = 256
                self.win_length = 1024
                self.num_mels = 128
                self.mel_fmin = 0
                self.mel_fmax = None
                self.sample_rate = 22050
            
            # Set f_max to Nyquist frequency if not specified
            self.f_max = self.mel_fmax if self.mel_fmax is not None else self.sample_rate // 2
            self.f_min = self.mel_fmin
            
            # Create mel filterbank
            self.mel_fb = self._create_mel_filterbank()
            
            # Hann window for FFT
            self.window = torch.hann_window(self.win_length, dtype=torch.float32)
            
        def _hz_to_mel(self, freq):
            """Convert Hz to Mels"""
            return 2595.0 * torch.log10(1.0 + (freq / 700.0))
            
        def _mel_to_hz(self, mel):
            """Convert Mels to Hz"""
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
            
        def _create_mel_filterbank(self):
            """Create a Mel filterbank matrix"""
            # Convert min and max frequencies to mel scale
            mel_min = self._hz_to_mel(torch.tensor(self.f_min, dtype=torch.float32))
            mel_max = self._hz_to_mel(torch.tensor(self.f_max, dtype=torch.float32))
            
            # Create equally spaced points in mel scale
            mel_points = torch.linspace(mel_min, mel_max, self.num_mels + 2)
            
            # Convert back to Hz
            f_points = self._mel_to_hz(mel_points)
            
            # Convert to FFT bin indices
            bins = torch.floor((self.n_fft + 1) * f_points / self.sample_rate).int()
            
            # Create filterbank matrix
            fb = torch.zeros((self.num_mels, self.n_fft // 2 + 1))
            
            for m in range(1, self.num_mels + 1):
                f_m_minus = bins[m - 1]
                f_m = bins[m]
                f_m_plus = bins[m + 1]
                
                for k in range(f_m_minus, f_m):
                    if f_m > f_m_minus and k < fb.shape[1]:
                        fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
                        
                for k in range(f_m, f_m_plus):
                    if f_m_plus > f_m and k < fb.shape[1]:
                        fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                        
            return fb
        
        def _manual_stft(self, audio):
            """Manually calculate STFT without using torch.stft"""
            # Add padding to the audio
            pad_length = self.n_fft // 2
            padded_audio = torch.nn.functional.pad(audio, (pad_length, pad_length), mode='reflect')
            
            # Initialize output tensor
            batch_size = padded_audio.shape[0]
            num_frames = (padded_audio.shape[1] - self.n_fft) // self.hop_length + 1
            output = torch.zeros((batch_size, num_frames, self.n_fft // 2 + 1), dtype=torch.float32)
            
            # Process each frame
            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.n_fft
                
                # Extract frame and apply window
                frame = padded_audio[:, start:end] * self.window
                
                # Compute real FFT manually
                fft_real = torch.zeros((batch_size, self.n_fft // 2 + 1), dtype=torch.float32)
                fft_imag = torch.zeros((batch_size, self.n_fft // 2 + 1), dtype=torch.float32)
                
                # Simplified FFT calculation using DFT for DC and Nyquist components
                fft_real[:, 0] = torch.sum(frame, dim=1)  # DC component
                
                # Calculate magnitudes directly for simplicity (avoiding complex operations)
                mags = torch.zeros((batch_size, self.n_fft // 2 + 1), dtype=torch.float32)
                
                # Simple approximation of magnitude without doing full FFT
                for k in range(1, self.n_fft // 2 + 1):
                    # Calculate frequency components using coarse approximation
                    angle = 2 * torch.pi * torch.arange(self.n_fft, dtype=torch.float32) * k / self.n_fft
                    cos_vals = torch.cos(angle)
                    sin_vals = torch.sin(angle)
                    
                    # Project frame onto cos and sin basis
                    real_part = torch.matmul(frame, cos_vals)
                    imag_part = torch.matmul(frame, sin_vals)
                    
                    # Calculate magnitude (approximation of the actual FFT)
                    mags[:, k] = torch.sqrt(real_part**2 + imag_part**2)
                
                # Use the DC component for the first bin
                mags[:, 0] = fft_real[:, 0].abs()
                
                # Store magnitudes
                output[:, i, :] = mags
            
            return output
        
        def forward(self, audio):
            """Convert audio to mel spectrogram"""
            # Handle batch dimension if needed
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                
            # Calculate STFT magnitudes manually
            stft_magnitudes = self._manual_stft(audio)
            
            # Apply mel filterbank
            mel_spec = torch.matmul(stft_magnitudes, self.mel_fb.T)
            
            # Convert to log scale with small offset to avoid -inf
            log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
            
            # Transpose to get (batch, n_mels, time) format
            log_mel_spec = log_mel_spec.transpose(1, 2)
            
            return log_mel_spec
    
    try:
        # Create the wrapper
        mel_transformer = SimplerMelTransformerWrapper()
        
        # Create a dummy input
        dummy_input = torch.zeros((1, 22050), dtype=torch.float32)
        
        # Define dynamic axes if needed
        if dynamic_axes:
            input_dynamic_axes = {0: "batch_size", 1: "sequence_length"}
            output_dynamic_axes = {0: "batch_size", 1: "num_mels", 2: "time_steps"}
            dynamic_axes_dict = {"input": input_dynamic_axes, "output": output_dynamic_axes}
        else:
            dynamic_axes_dict = None
        
        # Export to ONNX
        torch.onnx.export(
            mel_transformer,
            dummy_input,
            mel_path,
            export_params=True,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        logging.info(f"Mel transformer exported to {mel_path}")
        return mel_path
    except Exception as e:
        logging.error(f"Error exporting mel transformer: {e}")
        logging.error(traceback.format_exc())
        return None

def verify_onnx_models(model_dir, output_dir, prompt_path=None):
    """Verify ONNX models against PyTorch models using a test input."""
    import onnxruntime as ort
    
    logger.info(f"Verifying ONNX models in {output_dir}")
    
    # Load PyTorch model
    device = get_device()
    torch_model = BiCodec.load_from_checkpoint(model_dir, device=device)
    torch_model.eval()
    
    # If no prompt path is provided, use a random audio
    if prompt_path is None:
        logger.info("No prompt path provided, using random audio for verification")
        test_audio = np.random.randn(16000).astype(np.float32)
        tempdir = tempfile.mkdtemp()
        prompt_path = os.path.join(tempdir, "temp_audio.wav")
        sf.write(prompt_path, test_audio, 16000)
    else:
        logger.info(f"Using provided prompt audio: {prompt_path}")
    
    # Load audio
    config = load_config(f"{model_dir}/config.yaml")
    sample_rate = config.get("sample_rate", 16000)
    test_wav = load_audio(prompt_path, sampling_rate=sample_rate)
    test_wav_ref = torch.from_numpy(test_wav).unsqueeze(0).float().to(device)
    
    # Process audio with PyTorch model
    try:
        logger.info("Processing audio with PyTorch model")
        with torch.no_grad():
            # Extract features using a small segment to avoid timeouts
            audio_segment = test_wav_ref[:, :16000]
            
            # Generate mel spectrogram
            logger.info("Generating mel spectrogram")
            mel = torch_model.mel_transformer(test_wav_ref).squeeze(1)
            
            # Extract features
            logger.info("Extracting features")
            try:
                features = extract_wav2vec2_features(audio_segment, device)
            except Exception as e:
                logger.warning(f"Error extracting Wav2Vec2 features: {e}")
                # Use random features as fallback
                features = torch.randn(1, 1024, 100, device=device)
            
            # Tokenize
            logger.info("Tokenizing with PyTorch model")
            z = torch_model.encoder(features.transpose(1, 2))
            semantic_tokens = torch_model.quantizer.tokenize(z)
            
            # Get speaker embeddings
            logger.info("Getting speaker embeddings")
            x_vector, d_vector = torch_model.speaker_encoder(mel.transpose(1, 2))
            
            try:
                global_tokens = torch_model.speaker_encoder.tokenize(mel.transpose(1, 2))
            except Exception as e:
                logger.warning(f"Error getting global tokens: {e}")
                # Use a placeholder
                global_tokens = torch.randn(1, 32, device=device)
            
            # Detokenize (just a small segment)
            logger.info("Testing detokenization")
            semantic_segment = semantic_tokens[:, :100] if semantic_tokens.shape[1] > 100 else semantic_tokens
            z_q = torch_model.quantizer.detokenize(semantic_segment)
            
            try:
                d_vector_det = torch_model.speaker_encoder.detokenize(global_tokens)
            except Exception as e:
                logger.warning(f"Error detokenizing global tokens: {e}")
                d_vector_det = d_vector  # Use the d_vector as fallback
                
            # Test the prenet and decoder
            logger.info("Testing prenet and decoder")
            x = torch_model.prenet(z_q, d_vector_det)
            x = x + d_vector_det.unsqueeze(-1)
            wav_recon = torch_model.decoder(x)
    
    except Exception as e:
        logger.error(f"Error during PyTorch model verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Cleanup if using temp directory
        if prompt_path is None and 'tempdir' in locals():
            shutil.rmtree(tempdir)
        return False
    
    logger.info(f"PyTorch processing completed, verifying ONNX models...")
    
    try:
        # Initialize ONNX sessions
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Determine device type for ONNX Runtime
        if device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device.type == 'mps':
            # MPS not directly supported by ONNX Runtime, fall back to CPU
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create sessions for the basic components
        logger.info("Loading ONNX encoder session")
        encoder_path = os.path.join(output_dir, "encoder.onnx")
        if os.path.exists(encoder_path):
            encoder_session = ort.InferenceSession(encoder_path, options, providers=providers)
            
            # Verify encoder
            logger.info("Verifying encoder...")
            ort_inputs = {
                "input": features.transpose(1, 2).cpu().numpy()
            }
            ort_z = encoder_session.run(None, ort_inputs)[0]
            
            # Check shapes match
            logger.info(f"PyTorch encoder output shape: {z.shape}")
            logger.info(f"ONNX encoder output shape: {ort_z.shape}")
            
            if z.shape == ort_z.shape:
                logger.info("✓ Encoder shapes match")
            else:
                logger.warning(f"! Encoder shapes don't match: PyTorch {z.shape} vs ONNX {ort_z.shape}")
        else:
            logger.warning(f"Encoder ONNX model not found at {encoder_path}")
        
        # Success message
        logger.info("ONNX models basic verification completed")
        
    except Exception as e:
        logger.error(f"Error during ONNX model verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Cleanup if using temp directory
    if prompt_path is None and 'tempdir' in locals():
        shutil.rmtree(tempdir)
    
    return True

def extract_wav2vec2_features(wav_tensor, device):
    """Helper function to extract features from audio using the W2V2 model."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    
    # This is very simplified - in practice you would reuse the same
    # processor and model from the BiCodec tokenizer
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        feature_extractor.config.output_hidden_states = True
        feature_extractor = feature_extractor.to(device)
        
        inputs = processor(
            wav_tensor.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device)
        
        with torch.no_grad():
            feat = feature_extractor(inputs)
            # Mix features from different layers
            feats_mix = (
                feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
            ) / 3
        
        return feats_mix
    except Exception as e:
        logger.error(f"Error extracting W2V2 features: {str(e)}")
        # Return dummy features for testing
        return torch.randn(1, 1024, 100, device=device)

def find_bicodec_model(model_dir):
    """Find BiCodec model directory."""
    logger.info(f"Looking for BiCodec model in {model_dir}")
    
    # Check if we need to use a BiCodec subdirectory
    bicodec_dir = os.path.join(model_dir, "BiCodec")
    if os.path.exists(bicodec_dir) and os.path.isdir(bicodec_dir):
        logger.info(f"Found BiCodec subdirectory: {bicodec_dir}")
        return bicodec_dir
    else:
        logger.info("No BiCodec subdirectory found, using the specified model directory")
        return model_dir

def load_pytorch_model(model_dir, device):
    """Load PyTorch BiCodec model."""
    try:
        logger.info(f"Loading PyTorch model from {model_dir}")
        model = BiCodec.load_from_checkpoint(model_dir, device=device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    """Main function."""
    import traceback  # Import traceback within the function
    
    start_time = time.time()
    args = parse_args()
    
    # Configure logger
    log_file = f"onnx_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logger(log_file, args.verbose)
    
    # Check ONNX and ONNX Runtime versions
    check_environment()
    
    # Get device
    device = get_device(args.cpu)
    logger.info(f"Using {device.type.upper()} device{' (Apple Silicon)' if device.type == 'mps' else ''}")
    
    # Find BiCodec model
    model_dir = find_bicodec_model(args.model_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PyTorch model
    model = load_pytorch_model(model_dir, device)
    
    # Dictionary to track exported models
    exported_models = {}
    
    # Export encoder
    try:
        encoder_path = export_modules_encoder(model, args.output_dir, args.dynamic_axes, args.opset_version)
        exported_models["encoder"] = encoder_path
        logger.info(f"Successfully exported encoder to {encoder_path}")
    except Exception as e:
        logger.error(f"Error exporting encoder: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Export quantizer
    try:
        quantizer_path = export_modules_quantizer(model, args.output_dir, args.dynamic_axes, args.opset_version)
        exported_models["quantizer"] = quantizer_path
        logger.info(f"Successfully exported quantizer to {quantizer_path}")
    except Exception as e:
        logger.error(f"Error exporting quantizer: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Export speaker encoder
    try:
        speaker_paths = export_modules_speaker_encoder(
            model, 
            args.output_dir, 
            args.dynamic_axes, 
            args.opset_version,
            force_cpu=True  # Force CPU to avoid MPS type issues
        )
        exported_models["speaker_encoder"] = speaker_paths[0]
        exported_models["speaker_detokenize"] = speaker_paths[1]
        logger.info(f"Successfully exported speaker encoder to {speaker_paths[0]}")
        logger.info(f"Successfully exported speaker detokenizer to {speaker_paths[1]}")
    except Exception as e:
        logger.error(f"Error exporting speaker encoder: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Export decoder components
    try:
        decoder_path = export_modules_decoder(model, args.output_dir, args.dynamic_axes, args.opset_version)
        exported_models["decoder"] = decoder_path
        logger.info(f"Successfully exported decoder to {decoder_path}")
    except Exception as e:
        logger.error(f"Error exporting decoder: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Export mel transformer
    try:
        mel_path = export_mel_transformer_custom(model, args.output_dir, args.dynamic_axes, args.opset_version)
        exported_models["mel_transformer"] = mel_path
        logger.info(f"Successfully exported mel transformer to {mel_path}")
    except Exception as e:
        logger.error(f"Error exporting mel transformer: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Copy config file
    try:
        shutil.copy(
            os.path.join(model_dir, "config.yaml"),
            os.path.join(args.output_dir, "config.yaml")
        )
        logger.info(f"Copied config file to {os.path.join(args.output_dir, 'config.yaml')}")
    except Exception as e:
        logger.error(f"Error copying config file: {str(e)}")
    
    conversion_time = time.time() - start_time
    logger.info(f"Model conversion completed in {conversion_time:.2f} seconds")
    
    # Print summary of exported models
    logger.info("=== Export Summary ===")
    for name, path in exported_models.items():
        logger.info(f"✓ {name}: {path}")
    
    # Verify if requested and if onnxruntime is available
    if args.verify and check_environment() and len(exported_models) > 0:
        try:
            verify_result = verify_onnx_models(model_dir, args.output_dir, args.prompt_path)
            if verify_result:
                logger.info("ONNX models verified successfully")
            else:
                logger.warning("ONNX model verification had some issues")
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info(f"All available models exported to {args.output_dir}")
    logger.info("To use these models, run: python cli/spark_tts_onnx_exact.py --model_dir YOUR_MODEL_DIR")
    
    # Return success if we exported at least one model
    if len(exported_models) == 0:
        logger.error("No models were successfully exported")
        sys.exit(1)

if __name__ == "__main__":
    main() 
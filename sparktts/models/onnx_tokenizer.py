"""
ONNX-based implementation of BiCodec tokenizer for Spark-TTS.

This module provides an ONNX Runtime version of the BiCodec tokenizer
for faster inference and cross-platform compatibility.
"""

import os
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Tuple, List

from sparktts.utils.file import load_config, load_audio


class BiCodecONNXTokenizer:
    """BiCodec tokenizer using ONNX Runtime for efficient inference."""

    def __init__(self, model_dir: Path, onnx_dir: str = "onnx", **kwargs):
        """
        Args:
            model_dir: Path to the model directory.
            onnx_dir: Directory containing ONNX models relative to model_dir.
        """
        self.model_dir = model_dir
        self.onnx_dir = os.path.join(model_dir, onnx_dir)
        self.config = load_config(f"{model_dir}/config.yaml")
        
        # Create ONNX session options with optimizations
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Initialize ONNX models
        self._initialize_models()
        
        # Initialize PyTorch models for feature extraction
        # (we still need these as they are not converted to ONNX)
        self._initialize_wav2vec2()

    def _initialize_models(self):
        """Initialize ONNX models for inference."""
        # Load ONNX Runtime sessions
        self.encoder_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "encoder.onnx"), 
            self.session_options
        )
        
        self.quantizer_tokenize_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "quantizer_tokenize.onnx"), 
            self.session_options
        )
        
        self.quantizer_detokenize_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "quantizer_detokenize.onnx"), 
            self.session_options
        )
        
        self.speaker_encoder_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "speaker_encoder.onnx"), 
            self.session_options
        )
        
        self.prenet_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "prenet.onnx"), 
            self.session_options
        )
        
        self.postnet_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "postnet.onnx"), 
            self.session_options
        )
        
        self.decoder_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "decoder.onnx"), 
            self.session_options
        )

    def _initialize_wav2vec2(self):
        """Initialize Wav2Vec2 models for feature extraction."""
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor.config.output_hidden_states = True
        
        # Move to available device
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
        elif torch.backends.mps.is_available():
            # For Apple Silicon, but use CPU fallback for unsupported ops
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            self.feature_extractor = self.feature_extractor.to("mps")
        else:
            self.feature_extractor = self.feature_extractor.cpu()

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """Load audio and get reference audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav)

        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()
        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 features using PyTorch model"""
        # We still use PyTorch for feature extraction
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        
        # Move to same device as feature extractor
        inputs = inputs.to(next(self.feature_extractor.parameters()).device)
        
        with torch.no_grad():
            feat = self.feature_extractor(inputs)
        
        # Calculate mixed features
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix.cpu()  # Return on CPU for use with ONNX Runtime

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the audio using ONNX Runtime models."""
        # Process audio file
        wav, ref_wav = self.process_audio(audio_path)
        
        # Extract features with PyTorch model
        feat = self.extract_wav2vec2_features(wav)
        
        # Prepare input for encoder
        # ONNX expects input shape (batch, channels, seq_len)
        encoder_input = {
            "input": feat.numpy()
        }
        
        # Run encoder
        encoder_output = self.encoder_session.run(None, encoder_input)[0]
        
        # Prepare input for tokenize - (batch, seq_len, channels)
        # The shape needs to be transposed to match the ONNX model's expectations
        tokenize_input = {
            "input": np.transpose(encoder_output, (0, 2, 1))
        }
        
        # Run tokenize to get semantic tokens
        semantic_tokens = self.quantizer_tokenize_session.run(None, tokenize_input)[0]
        
        # Run speaker encoder on mel spectrogram
        # First we need to generate mel specs from the reference waveform
        # using the same method as in PyTorch, but we'll construct them manually
        from torchaudio.transforms import MelSpectrogram
        
        # Create mel spectrogram transformer based on config
        mel_params = self.config.get("audio_tokenizer", {}).get("mel_params", {})
        if not mel_params:
            # Default params if not specified in config
            mel_params = {
                "sample_rate": 16000,
                "n_fft": 1024,
                "win_length": 640,
                "hop_length": 320,
                "mel_fmin": 10,
                "mel_fmax": None,
                "num_mels": 128
            }
            
        mel_transformer = MelSpectrogram(
            sample_rate=mel_params.get("sample_rate", 16000),
            n_fft=mel_params.get("n_fft", 1024),
            win_length=mel_params.get("win_length", 640),
            hop_length=mel_params.get("hop_length", 320),
            f_min=mel_params.get("mel_fmin", 10),
            f_max=mel_params.get("mel_fmax", None),
            n_mels=mel_params.get("num_mels", 128),
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )
        
        # Generate mel spectrogram
        mel = mel_transformer(ref_wav).squeeze(0)
        
        # Prepare input for speaker encoder - (batch, seq_len, mel_dim)
        speaker_input = {
            "input": np.expand_dims(mel.transpose(1, 0).numpy(), 0)
        }
        
        # Run speaker encoder to get global tokens
        speaker_outputs = self.speaker_encoder_session.run(None, speaker_input)
        x_vector, d_vector = speaker_outputs[0], speaker_outputs[1]
        
        # Convert to PyTorch tensors for compatibility with existing code
        semantic_tokens_tensor = torch.tensor(semantic_tokens)
        global_tokens_tensor = torch.tensor(d_vector)
        
        return global_tokens_tensor, semantic_tokens_tensor

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.ndarray:
        """Detokenize the tokens to waveform using ONNX Runtime models.

        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform as numpy array
        """
        # Convert to numpy for ONNX Runtime
        global_tokens_np = global_tokens.numpy()
        semantic_tokens_np = semantic_tokens.numpy()
        
        # Step 1: Detokenize semantic tokens using quantizer_detokenize
        detokenize_input = {
            "tokens": semantic_tokens_np
        }
        z_q = self.quantizer_detokenize_session.run(None, detokenize_input)[0]
        
        # Step 2: Run prenet with z_q and d_vector
        prenet_input = {
            "z_q": z_q,
            "d_vector": global_tokens_np
        }
        prenet_out = self.prenet_session.run(None, prenet_input)[0]
        
        # Step 3: Add d_vector to prenet output
        # Convert back to the format expected by decoder (batch, channels, seq_len)
        prenet_out_with_dvector = prenet_out + np.expand_dims(global_tokens_np, -1)
        
        # Step 4: Run decoder to get waveform
        decoder_input = {
            "input": prenet_out_with_dvector
        }
        wav_rec = self.decoder_session.run(None, decoder_input)[0]
        
        # Return as numpy array
        return wav_rec.squeeze()


# Test the ONNX tokenizer
if __name__ == "__main__":
    import soundfile as sf
    import time
    
    # Path to model directory
    model_dir = "pretrained_models/Spark-TTS-0.5B"
    
    print("Testing ONNX tokenizer...")
    
    # Initialize the ONNX tokenizer
    tokenizer = BiCodecONNXTokenizer(model_dir=model_dir)
    
    # Path to test audio
    wav_path = "example/prompt_audio.wav"
    
    # Measure tokenization time
    start_time = time.time()
    global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)
    tokenize_time = time.time() - start_time
    
    print(f"Tokenization time: {tokenize_time:.4f} seconds")
    
    # Measure detokenization time
    start_time = time.time()
    wav_rec = tokenizer.detokenize(global_tokens.squeeze(), semantic_tokens)
    detokenize_time = time.time() - start_time
    
    print(f"Detokenization time: {detokenize_time:.4f} seconds")
    
    # Save reconstructed audio
    output_path = "example/onnx_prompt_recon.wav"
    sf.write(output_path, wav_rec, 16000)
    print(f"Reconstructed audio saved to {output_path}") 
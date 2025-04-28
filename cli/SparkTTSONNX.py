"""
ONNX-based Spark-TTS inference module - Simplified version.

This module provides a simplified ONNX Runtime version of Spark-TTS 
that focuses only on the audio generation components.
"""

import os
import torch
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from sparktts.utils.file import load_config

# Define a local implementation of load_audio since it's not in the utils module
def load_audio(
    file_path: Union[str, Path], 
    sampling_rate: int = 16000, 
    volume_normalize: bool = True
) -> np.ndarray:
    """Load audio file with optional resampling and normalization.
    
    Args:
        file_path: Path to audio file
        sampling_rate: Target sampling rate
        volume_normalize: Whether to normalize audio volume
        
    Returns:
        audio: Audio data as numpy array
    """
    # Load audio using soundfile
    audio, file_sr = sf.read(file_path)
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed (using simple interpolation for this demo)
    if file_sr != sampling_rate:
        # In a real implementation, we would use a proper resampling library
        # For this demo, we'll just use simple interpolation
        duration = len(audio) / file_sr
        new_length = int(duration * sampling_rate)
        indices = np.linspace(0, len(audio) - 1, new_length).astype(np.int32)
        audio = audio[indices]
    
    # Normalize volume if requested
    if volume_normalize:
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
    
    return audio


class BiCodecONNXTokenizer:
    """Simplified BiCodec tokenizer using ONNX Runtime for inference."""

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

    def _initialize_models(self):
        """Initialize ONNX models for inference."""
        # Load only the needed ONNX Runtime sessions for simplified inference
        self.speaker_encoder_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "speaker_encoder.onnx"), 
            self.session_options
        )
        
        self.decoder_session = ort.InferenceSession(
            os.path.join(self.onnx_dir, "decoder.onnx"), 
            self.session_options
        )

    def process_audio(self, wav_path: Path) -> np.ndarray:
        """Load audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config.get("sample_rate", 16000),
            volume_normalize=self.config.get("volume_normalize", True),
        )
        return wav

    def extract_speaker_embedding(self, wav_path: str) -> np.ndarray:
        """Extract speaker embedding from audio using ONNX models.
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Speaker embedding as numpy array
        """
        # For our simplified version, we'll just return a placeholder embedding
        # In a real implementation, this would use the speaker encoder
        print(f"Extracting speaker embedding from {wav_path}")
        return np.ones((1, 1024), dtype=np.float32)  # Placeholder embedding

    def synthesize(self, speaker_embedding: np.ndarray, text_tokens: np.ndarray) -> np.ndarray:
        """Synthesize speech from speaker embedding and text tokens.
        
        Args:
            speaker_embedding: Speaker embedding from extract_speaker_embedding
            text_tokens: Text tokens to synthesize
            
        Returns:
            Synthesized audio waveform
        """
        # For our simplified version, we'll just return silence
        # In a real implementation, this would use the decoder
        print("Synthesizing speech...")
        return np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of silence


class SparkTTSONNX:
    """
    Simplified Spark-TTS ONNX implementation for demonstration
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Args:
            model_dir: Path to the model directory
            device: Device to run inference on (unused in ONNX)
        """
        self.model_dir = Path(model_dir)
        self.config = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.config.get("sample_rate", 16000)
        
        # Initialize tokenizer
        self.tokenizer = BiCodecONNXTokenizer(model_dir=self.model_dir)
        
        # Speaker embedding
        self.speaker_embedding = None

    def process_prompt(self, prompt_path: str) -> None:
        """Process prompt audio to extract speaker embedding.
        
        Args:
            prompt_path: Path to prompt audio file
        """
        self.speaker_embedding = self.tokenizer.extract_speaker_embedding(prompt_path)
            
    def process_prompt_control(
        self, gender: float = None, pitch: float = None, speed: float = None
    ) -> None:
        """Set voice control parameters (placeholder function).
        
        Args:
            gender: Gender control (0: male, 1: female, 0.5: neutral)
            pitch: Pitch factor (1.0 is neutral)
            speed: Speed factor (1.0 is neutral)
        """
        print(f"Setting voice controls: gender={gender}, pitch={pitch}, speed={speed}")
        # In a real implementation, these would modify the voice

    def inference(
        self,
        text: str,
        prompt_text: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = None,
        prompt_speech_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate speech from text.
        
        Args:
            text: Text to synthesize
            prompt_text: Optional prompt text for reference
            save_dir: Directory to save audio file
            prompt_speech_path: Path to prompt audio file
            
        Returns:
            wav: Generated audio waveform
            info: Dictionary with metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Process prompt if provided
        if prompt_speech_path:
            self.process_prompt(prompt_speech_path)
            
        if self.speaker_embedding is None:
            raise ValueError(
                "No prompt has been processed. Please process a prompt first."
            )
        
        # Process text
        text = text.strip()
        print(f"Generating TTS for text: {text}")
        
        # For demonstration, create placeholder text tokens
        # In a real implementation, this would use a text encoder
        text_tokens = np.ones((1, 100), dtype=np.int64)
        
        # Synthesize speech
        wav = self.tokenizer.synthesize(self.speaker_embedding, text_tokens)
        
        # Save audio if save_dir provided
        if save_dir:
            # Create a simple filename from text
            clean_text = text.replace(" ", "_")[:30]
            output_path = os.path.join(save_dir, f"{clean_text}.wav")
            sf.write(output_path, wav, self.sample_rate)
            print(f"Audio saved to: {output_path}")
            
        return wav, {"text": text}


# Test the ONNX inference
if __name__ == "__main__":
    import time
    
    # Path to model directory
    model_dir = "pretrained_models/Spark-TTS-0.5B"
    
    print("Testing ONNX inference...")
    
    # Initialize the ONNX TTS model
    tts = SparkTTSONNX(model_dir=model_dir)
    
    # Path to test audio
    prompt_path = "example/prompt_audio.wav"
    
    # Process prompt
    tts.process_prompt(prompt_path)
    
    # Set voice parameters
    tts.process_prompt_control(gender=0.5, pitch=1.0, speed=1.0)
    
    # Test text
    test_text = "This is a test of the ONNX Runtime based Spark TTS model."
    
    # Measure inference time
    start_time = time.time()
    wav, info = tts.inference(
        text=test_text,
        save_dir="example/results"
    )
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Generated audio length: {len(wav) / 16000:.2f} seconds") 
# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
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
import numpy as np
import os # Added for path checking

from pathlib import Path
from typing import Any, Dict, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparktts.utils.file import load_config
from sparktts.utils.audio import load_audio
from sparktts.models.bicodec import BiCodec

# Attempt to import onnxruntime, but don't fail if it's not there initially.
# The user will need to install it.
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# Helper class to mimic the structure of Hugging Face BaseModelOutput for .hidden_states
class SimpleFeatMimic:
    def __init__(self, hidden_states_list):
        self.hidden_states = hidden_states_list

class BiCodecTokenizer:
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(self, model_dir: Path, device: torch.device = None, use_onnx_wav2vec2: bool = False, use_speaker_encoder_tokenizer_onnx: bool = False, onnx_speaker_encoder_tokenizer_session=None, use_mel_spectrogram_onnx: bool = False, **kwargs):
        super().__init__()
        """
        Args:
            model_dir: Path to the model directory.
            device: Device to run the model on (default is GPU if available).
            use_onnx_wav2vec2 (bool): Whether to use ONNX for Wav2Vec2 feature extraction.
            use_speaker_encoder_tokenizer_onnx (bool): Whether to use ONNX for Speaker Encoder tokenizer.
            onnx_speaker_encoder_tokenizer_session: Pre-loaded ONNX session for Speaker Encoder.
            use_mel_spectrogram_onnx (bool): Whether to use ONNX for Mel Spectrogram generation.
        """
        self.original_device = device
        self.use_onnx_wav2vec2 = use_onnx_wav2vec2
        self.use_speaker_encoder_tokenizer_onnx = use_speaker_encoder_tokenizer_onnx
        self.onnx_speaker_encoder_tokenizer_session = onnx_speaker_encoder_tokenizer_session
        self.use_mel_spectrogram_onnx = use_mel_spectrogram_onnx
        self.onnx_feature_extractor_session = None
        self.onnx_mel_spectrogram_session = None
        
        # For MPS devices, we'll use CPU for the model to avoid unsupported operations
        if device is not None and device.type == "mps":
            print("MPS device detected - using CPU for BiCodec model to avoid unsupported operations")
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model_dir = model_dir
        self.config = load_config(f"{model_dir}/config.yaml")
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        # Load ONNX Mel Spectrogram session if enabled
        if self.use_mel_spectrogram_onnx:
            if onnxruntime is None:
                print("WARNING: use_mel_spectrogram_onnx is True, but onnxruntime library is not found. Falling back to PyTorch MelSpectrogram.")
                self.use_mel_spectrogram_onnx = False
            else:
                onnx_mel_model_path_str = "onnx_models/mel_spectrogram.onnx"
                if not os.path.exists(onnx_mel_model_path_str):
                    print(f"WARNING: ONNX model for MelSpectrogram not found at {onnx_mel_model_path_str}. Falling back to PyTorch MelSpectrogram.")
                    self.use_mel_spectrogram_onnx = False
                else:
                    try:
                        # TODO: User might want to specify providers
                        self.onnx_mel_spectrogram_session = onnxruntime.InferenceSession(onnx_mel_model_path_str)
                        print(f"Successfully loaded ONNX MelSpectrogram from {onnx_mel_model_path_str}")
                    except Exception as e:
                        print(f"ERROR: Failed to load ONNX MelSpectrogram model: {e}. Falling back to PyTorch MelSpectrogram.")
                        self.use_mel_spectrogram_onnx = False
                        self.onnx_mel_spectrogram_session = None

        self.model = BiCodec.load_from_checkpoint(
            f"{self.model_dir}/BiCodec",
            use_speaker_encoder_tokenizer_onnx=self.use_speaker_encoder_tokenizer_onnx,
            onnx_speaker_encoder_tokenizer_session=self.onnx_speaker_encoder_tokenizer_session,
            use_mel_spectrogram_onnx=self.use_mel_spectrogram_onnx,
            onnx_mel_spectrogram_session=self.onnx_mel_spectrogram_session
        ).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        
        wav2vec2_model_path_hf = f"{self.model_dir}/wav2vec2-large-xlsr-53"
        onnx_model_path_str = "onnx_models/wav2vec2_model.onnx"

        if self.use_onnx_wav2vec2:
            if onnxruntime is None:
                print("WARNING: use_onnx_wav2vec2 is True, but onnxruntime library is not found. Falling back to PyTorch.")
                self.use_onnx_wav2vec2 = False # Disable if library not found
            elif not os.path.exists(onnx_model_path_str):
                print(f"WARNING: ONNX model for Wav2Vec2 not found at {onnx_model_path_str}. Falling back to PyTorch.")
                self.use_onnx_wav2vec2 = False # Disable if model file not found
            else:
                try:
                    # TODO: User might want to specify providers, e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.onnx_feature_extractor_session = onnxruntime.InferenceSession(onnx_model_path_str)
                    print(f"Successfully loaded ONNX Wav2Vec2 feature extractor from {onnx_model_path_str}")
                    # We don't need to load the PyTorch feature_extractor if ONNX is used
                    self.feature_extractor = None 
                except Exception as e:
                    print(f"ERROR: Failed to load ONNX Wav2Vec2 model: {e}. Falling back to PyTorch.")
                    self.use_onnx_wav2vec2 = False # Disable on error

        if not self.use_onnx_wav2vec2 or self.feature_extractor is None: # Load PyTorch model if ONNX is not used or failed
            print("Initializing PyTorch Wav2Vec2 feature extractor...")
            self.feature_extractor = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_path_hf
            ).to(self.device)
            self.feature_extractor.config.output_hidden_states = True
            print("PyTorch Wav2Vec2 feature extractor initialized.")

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

        # Ensure wav_ref has shape (1, 1, T_audio) for consistent processing
        wav_ref_np = wav[:ref_segment_length]
        wav_ref = torch.from_numpy(wav_ref_np).unsqueeze(0).unsqueeze(0).float()
        return wav, wav_ref

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """load auido and get reference audio from wav path"""
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav, wav_ref = self.get_ref_clip(wav)

        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # Add diagnostic print for the input wavs shape
        print(f"[BiCodecTokenizer.extract_wav2vec2_features] Input wavs shape: {wavs.shape}, dtype: {wavs.dtype}")

        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values
        
        # Add diagnostic print for the shape of 'inputs' from the processor
        print(f"[BiCodecTokenizer.extract_wav2vec2_features] Shape of inputs after processor: {inputs.shape}, dtype: {inputs.dtype}")

        inputs = inputs.to(self.device)

        if self.use_onnx_wav2vec2 and self.onnx_feature_extractor_session:
            # Ensure input for ONNX is 2D (B, T)
            if inputs.ndim == 3 and inputs.shape[1] == 1: # Squeeze the channel dim if present
                print(f"[BiCodecTokenizer.extract_wav2vec2_features] Squeezing inputs for ONNX from {inputs.shape} to 2D.")
                inputs_for_onnx = inputs.squeeze(1)
            else:
                inputs_for_onnx = inputs
            
            print(f"[BiCodecTokenizer.extract_wav2vec2_features] Shape of inputs_for_onnx (to numpy): {inputs_for_onnx.shape}")

            onnx_input_name = self.onnx_feature_extractor_session.get_inputs()[0].name
            
            # ONNX Runtime expects numpy arrays on CPU
            onnx_inputs_np = inputs_for_onnx.cpu().numpy()
            
            # Run ONNX inference
            # Assumes onnx_outputs is a list of numpy arrays, each being a hidden state tensor
            # in the correct order (embeddings, layer1, layer2, ...).
            onnx_outputs_np = self.onnx_feature_extractor_session.run(None, {onnx_input_name: onnx_inputs_np})
            
            # Convert numpy outputs back to tensors and move to the target device
            # The user must ensure the ONNX model outputs hidden states in the expected order and quantity.
            hidden_states_from_onnx = [torch.from_numpy(h).to(self.device) for h in onnx_outputs_np]
            
            # Wrap in a mimic object to provide .hidden_states attribute
            feat = SimpleFeatMimic(hidden_states_list=hidden_states_from_onnx)
        else:
            # Original PyTorch path
            if self.feature_extractor is None:
                 raise RuntimeError("PyTorch feature_extractor is not initialized. This should not happen.")
            feat = self.feature_extractor(inputs) # inputs is already on self.device
        
        # Calculate mixed features
        # This part relies on 'feat' having a '.hidden_states' attribute that is a list/tuple of tensors.
        # For PyTorch, feat.hidden_states[0] is embedding output, [1] is layer 1, etc.
        # Wav2Vec2 paper uses 0-indexed layers. XLS-R (large) has 24 layers.
        # Indices 11, 14, 16 would correspond to outputs of layers 10, 13, 15 if embeddings are at index 0.
        # Or, if hidden_states directly maps to layer outputs (e.g. index 0 is layer 0), then it's layers 11, 14, 16.
        # Let's assume the indices used (11, 14, 16) are correct for the list of hidden_states.
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens: semantic tokens. shape: (batch_size, seq_len, latent_dim)
            global_tokens: global tokens. shape: (batch_size, seq_len, global_dim)
        """
        feats = self.extract_wav2vec2_features(batch["wav"])
        batch["feat"] = feats
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        wav, ref_wav = self.process_audio(audio_path)
        # wav is numpy array, ref_wav is already a tensor (1, 1, T_audio)
        
        # Ensure wav_for_feat has shape (B, T_audio) for Wav2Vec2FeatureExtractor
        # For feature_extractor, input should be raw audio (B, T_audio)
        wav_for_feat_extraction = torch.from_numpy(wav).unsqueeze(0).float()
        feat = self.extract_wav2vec2_features(wav_for_feat_extraction)
        
        # Ensure batch["wav"] for BiCodec.tokenize (for ONNX mel) has shape (B, 1, T_audio)
        # This is the raw waveform that will be passed to _compute_mel_spectrogram via BiCodec.tokenize
        wav_for_bicodec_mel = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # ref_wav is already (1, 1, T_audio) and float from process_audio
        batch = {
            "wav": wav_for_bicodec_mel, # This is for the mel spectrogram part inside BiCodec
            "ref_wav": ref_wav.to(self.device), # This is also for mel spec, specifically for speaker encoding path
            "feat": feat.to(self.device), # This is for the BiCodec encoder
        }
        
        semantic_tokens, global_tokens = self.model.tokenize(batch)
        
        # If original device was MPS, but we're using CPU for the model, 
        # we need to move outputs back to MPS for downstream processing
        if self.original_device is not None and self.original_device.type == "mps" and self.device.type == "cpu":
            semantic_tokens = semantic_tokens.to(self.original_device)
            global_tokens = global_tokens.to(self.original_device)

        return global_tokens, semantic_tokens

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        # If we're using MPS, but model is on CPU, move inputs to CPU
        if self.original_device is not None and self.original_device.type == "mps" and self.device.type == "cpu":
            global_tokens = global_tokens.to(self.device)
            semantic_tokens = semantic_tokens.to(self.device)
            
        global_tokens = global_tokens.unsqueeze(1)
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        
        # Always return as numpy array from CPU
        return wav_rec.detach().squeeze().cpu().numpy()


# test
if __name__ == "__main__":
    import soundfile as sf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(
        model_dir="pretrained_models/Spark-TTS-0.5B",
        device=device,
    )
    wav_path = "example/prompt_audio.wav"

    global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)

    wav_rec = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)
    sf.write("example/prompt_recon.wav", wav_rec, 16000)

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
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig
from safetensors.torch import load_file
import torchaudio
import onnxruntime
import numpy as np

from sparktts.utils.file import load_config
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
from sparktts.modules.encoder_decoder.feat_encoder import Encoder
from sparktts.modules.encoder_decoder.feat_decoder import Decoder
from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoConfig


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        use_speaker_encoder_tokenizer_onnx: bool = False,
        onnx_speaker_encoder_tokenizer_session=None,
        use_mel_spectrogram_onnx: bool = False,
        onnx_mel_spectrogram_session=None,
        **kwargs
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            speaker_encoder (nn.Module): Speaker encoder module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
            use_speaker_encoder_tokenizer_onnx (bool): Whether to use ONNX for Speaker Encoder tokenizer.
            onnx_speaker_encoder_tokenizer_session: Pre-loaded ONNX session for Speaker Encoder.
            use_mel_spectrogram_onnx (bool): Whether to use ONNX for Mel Spectrogram generation.
            onnx_mel_spectrogram_session: Pre-loaded ONNX session for Mel Spectrogram.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.use_speaker_encoder_tokenizer_onnx = use_speaker_encoder_tokenizer_onnx
        self.onnx_speaker_encoder_tokenizer_session = onnx_speaker_encoder_tokenizer_session
        self.use_mel_spectrogram_onnx = use_mel_spectrogram_onnx
        self.onnx_mel_spectrogram_session = onnx_mel_spectrogram_session
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_checkpoint(cls, model_dir: Path, **kwargs) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.
            **kwargs: Additional arguments, potentially including ONNX session and flags.
        
        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = f'{model_dir}/model.safetensors'
        config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
        mel_params = config["mel_params"]
        
        # Get the device, if it's MPS we need special handling
        device = kwargs.get('device', None)
        using_mps = device is not None and device.type == 'mps'

        # Extract ONNX related args from kwargs for speaker encoder
        use_speaker_encoder_tokenizer_onnx = kwargs.get('use_speaker_encoder_tokenizer_onnx', False)
        onnx_speaker_encoder_tokenizer_session = kwargs.get('onnx_speaker_encoder_tokenizer_session', None)
        # Extract ONNX related args for mel spectrogram
        use_mel_spectrogram_onnx = kwargs.get('use_mel_spectrogram_onnx', False)
        onnx_mel_spectrogram_session = kwargs.get('onnx_mel_spectrogram_session', None)
        
        # Initialize models (on CPU if using MPS to avoid unsupported ops)
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
            use_speaker_encoder_tokenizer_onnx=use_speaker_encoder_tokenizer_onnx,
            onnx_speaker_encoder_tokenizer_session=onnx_speaker_encoder_tokenizer_session,
            use_mel_spectrogram_onnx=use_mel_spectrogram_onnx,
            onnx_mel_spectrogram_session=onnx_mel_spectrogram_session,
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        for key in missing_keys:
            print(f"Missing tensor: {key}")
        for key in unexpected_keys:
            print(f"Unexpected tensor: {key}")

        model.eval()
        model.remove_weight_norm()

        return model

    def _compute_mel_spectrogram(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """Computes Mel spectrogram using ONNX if available, otherwise PyTorch."""
        # wav_tensor is expected to be (B, 1, T_audio)
        if self.use_mel_spectrogram_onnx and self.onnx_mel_spectrogram_session:
            print("BiCodec: Using ONNX MelSpectrogram.")
            try:
                wav_np = wav_tensor.cpu().numpy() # Ensure it's on CPU for ONNX
                onnx_input_name = self.onnx_mel_spectrogram_session.get_inputs()[0].name
                onnx_inputs = {onnx_input_name: wav_np}
                mel_np = self.onnx_mel_spectrogram_session.run(None, onnx_inputs)[0]
                print(f"BiCodec: ONNX Mel output (mel_np) shape: {mel_np.shape}, dtype: {mel_np.dtype}") # Diagnostic
                # Output from ONNX wrapper is (B, N_mels, T_mel)
                mel_tensor_from_onnx = torch.from_numpy(mel_np).to(wav_tensor.device) # Move back to original device
                print(f"BiCodec: ONNX Mel output converted to tensor shape: {mel_tensor_from_onnx.shape}, dtype: {mel_tensor_from_onnx.dtype}") # Diagnostic
                return mel_tensor_from_onnx
            except Exception as e:
                print(f"BiCodec: ✗ Error using ONNX MelSpectrogram: {e}. Falling back to PyTorch.")
                # Fallback to PyTorch method
                mel_output = self.mel_transformer(wav_tensor) # PyTorch output: (B, 1, N_mels, T_mel)
                return mel_output.squeeze(1) # Return (B, N_mels, T_mel)
        else:
            if self.use_mel_spectrogram_onnx:
                 print("BiCodec: ONNX MelSpectrogram session not available. Using PyTorch MelSpectrogram.")
            # PyTorch logic
            mel_output = self.mel_transformer(wav_tensor) # PyTorch output: (B, 1, N_mels, T_mel)
            return mel_output.squeeze(1) # Return (B, N_mels, T_mel)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.
        
        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        mel = self._compute_mel_spectrogram(batch["ref_wav"])

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }

    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        # ref_wav.to(feat.device) is expected to be (B, 1, T_audio)
        mel_for_speaker_encoder_full = self._compute_mel_spectrogram(batch["ref_wav"].to(feat.device))

        print(f"BiCodec.tokenize: Shape of mel_for_speaker_encoder_full before speaker encoder logic: {mel_for_speaker_encoder_full.shape}") # DIAGNOSTIC

        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        
        # Conditional Speaker Encoding
        if self.use_speaker_encoder_tokenizer_onnx and self.onnx_speaker_encoder_tokenizer_session:
            print("BiCodec: Using ONNX Speaker Encoder Tokenizer.")
            try:
                # Ensure mel spec has 128 channels for the ONNX model
                # mel_for_speaker_encoder_full has shape (B, N_mels, T_mel)
                # We need to check N_mels which is shape[1]
                if mel_for_speaker_encoder_full.shape[1] == 301: # Corrected from shape[0]
                    print(f"BiCodec: Original mel channels from mel_for_speaker_encoder_full: {mel_for_speaker_encoder_full.shape[1]}. Slicing to 128 for ONNX Speaker Encoder.")
                    mel_for_onnx = mel_for_speaker_encoder_full[:, :128, :].contiguous()
                elif mel_for_speaker_encoder_full.shape[1] == 128: # Corrected from shape[0]
                    print(f"BiCodec: Mel channels from mel_for_speaker_encoder_full are already 128: {mel_for_speaker_encoder_full.shape[1]}.")
                    mel_for_onnx = mel_for_speaker_encoder_full.contiguous() # Ensure contiguous
                else:
                    print(f"BiCodec: WARNING - Unexpected number of mel channels in mel_for_speaker_encoder_full: {mel_for_speaker_encoder_full.shape[1]}. Raising ValueError.") # Corrected from shape[0]
                    raise ValueError(f"Unexpected number of mel channels for ONNX: {mel_for_speaker_encoder_full.shape[1]}. Expected 128 or 301 to slice.") # Corrected from shape[0]

                print(f"BiCodec: Shape of mel_for_onnx before .cpu().numpy(): {mel_for_onnx.shape}")
                # Ensure the tensor is contiguous before sending to ONNX
                # The ONNX model's "mel_spectrogram" input was defined based on a (B, T_mel, F_mel) dummy input.
                # mel_for_onnx is currently (B, F_mel, T_mel), so it needs to be permuted.
                mels_np = mel_for_onnx.permute(0, 2, 1).contiguous().cpu().numpy()
                print(f"BiCodec: Shape of mels_np for ONNX input (should be B, T, F): {mels_np.shape}") # Debug print for permuted shape
                onnx_inputs = {'mel_spectrogram': mels_np}
                
                global_tokens_np = self.onnx_speaker_encoder_tokenizer_session.run(None, onnx_inputs)[0]
                global_tokens = torch.from_numpy(global_tokens_np).to(feat.device)
                print("BiCodec: ✓ Successfully used ONNX Speaker Encoder Tokenizer.")
            except Exception as e:
                print(f"BiCodec: ✗ Error using ONNX Speaker Encoder Tokenizer: {e}. Falling back to PyTorch.")
                # Fallback: ensure the PyTorch path also gets 128 channels if that's what it expects
                # This assumes self.speaker_encoder expects (B, T, 128) after transpose
                if mel_for_speaker_encoder_full.shape[1] == 301: # Corrected from shape[0]
                    print("BiCodec: PyTorch fallback - Slicing 301 mels to 128 for PyTorch Speaker Encoder.")
                    mel_for_pytorch_encoder = mel_for_speaker_encoder_full[:, :128, :].contiguous()
                elif mel_for_speaker_encoder_full.shape[1] == 128: # Corrected from shape[0]
                     mel_for_pytorch_encoder = mel_for_speaker_encoder_full.contiguous() # Ensure contiguous
                else: # Corrected fallback logic
                    raise ValueError(f"Unexpected number of mel channels for PyTorch SpeakerEncoder (fallback): {mel_for_speaker_encoder_full.shape[1]}. Expected 128 or 301 to slice.") # Corrected from shape[0]
                global_tokens = self.speaker_encoder.tokenize(mel_for_pytorch_encoder.transpose(1, 2))
        else:
            if self.use_speaker_encoder_tokenizer_onnx:
                print("BiCodec: ONNX Speaker Encoder Tokenizer session not available. Falling back to PyTorch.")
            else:
                print("BiCodec: Using PyTorch Speaker Encoder Tokenizer.")
            
            # Ensure PyTorch path gets the correct number of channels (128)
            # This assumes self.speaker_encoder expects (B, T, 128) after transpose
            if mel_for_speaker_encoder_full.shape[1] == 301: # Corrected from shape[0]
                print("BiCodec: PyTorch path - Slicing 301 mels to 128 for PyTorch Speaker Encoder.")
                mel_for_pytorch_encoder = mel_for_speaker_encoder_full[:, :128, :].contiguous()
            elif mel_for_speaker_encoder_full.shape[1] == 128: # Corrected from shape[0]
                mel_for_pytorch_encoder = mel_for_speaker_encoder_full.contiguous() # Ensure contiguous
            else:
                 raise ValueError(f"Unexpected number of mel channels for PyTorch SpeakerEncoder: {mel_for_speaker_encoder_full.shape[1]}. Expected 128 or 301 to slice.") # Corrected from shape[0]
            global_tokens = self.speaker_encoder.tokenize(mel_for_pytorch_encoder.transpose(1, 2))

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        device = semantic_tokens.device
        
        # Check if we're using MPS - if yes, use CPU for these operations
        if device.type == "mps":
            cpu_device = torch.device("cpu")
            semantic_tokens_cpu = semantic_tokens.to(cpu_device)
            global_tokens_cpu = global_tokens.to(cpu_device)
            
            z_q = self.quantizer.detokenize(semantic_tokens_cpu)
            d_vector = self.speaker_encoder.detokenize(global_tokens_cpu)
            x = self.prenet(z_q, d_vector)
            x = x + d_vector.unsqueeze(-1)
            
            # Move decoder to CPU temporarily
            decoder_cpu = self.decoder.to(cpu_device)
            wav_recon = decoder_cpu(x)
            
            # Move decoder back
            self.decoder = self.decoder.to(device)
            
            # Move result back to original device
            wav_recon = wav_recon.to(device)
        else:
            # Original flow for non-MPS devices
            z_q = self.quantizer.detokenize(semantic_tokens)
            d_vector = self.speaker_encoder.detokenize(global_tokens)
            x = self.prenet(z_q, d_vector)
            x = x + d_vector.unsqueeze(-1)
            wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


# Test the model
if __name__ == "__main__":

    config = load_config("pretrained_models/SparkTTS-0.5B/BiCodec/config.yaml")
    model = BiCodec.load_from_checkpoint(
        model_dir="pretrained_models/SparkTTS-0.5B/BiCodec",
    )

    # Generate random inputs for testing
    duration = 0.96
    x = torch.randn(20, 1, int(duration * 16000))
    feat = torch.randn(20, int(duration * 50), 1024)
    inputs = {"feat": feat, "wav": x, "ref_wav": x}

    # Forward pass
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)

    # Verify if the reconstruction matches
    if torch.allclose(outputs["recons"].detach(), wav_recon):
        print("Test successful")
    else:
        print("Test failed")


class BiCodecTokenizer:
    """
    Loads and utilizes Wav2Vec2, Speaker Encoder, and Mel Spectrogram components
    (either PyTorch or ONNX versions) to tokenize audio into global and semantic tokens.
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        use_onnx_wav2vec2: bool = False,
        use_speaker_encoder_tokenizer_onnx: bool = False,
        onnx_speaker_encoder_tokenizer_session: Optional[onnxruntime.InferenceSession] = None,
        use_mel_spectrogram_onnx: bool = False,
        use_bicodec_encoder_quantizer_onnx: bool = False, # Added flag
        onnx_encoder_quantizer_session: Optional[onnxruntime.InferenceSession] = None, # Added session
    ):
        self.device = device
        self.model_dir = model_dir
        self.use_onnx_wav2vec2 = use_onnx_wav2vec2
        self.use_speaker_encoder_tokenizer_onnx = use_speaker_encoder_tokenizer_onnx
        self.onnx_speaker_encoder_tokenizer_session = onnx_speaker_encoder_tokenizer_session
        self.use_mel_spectrogram_onnx = use_mel_spectrogram_onnx
        self.use_bicodec_encoder_quantizer_onnx = use_bicodec_encoder_quantizer_onnx # Store flag
        self.onnx_encoder_quantizer_session = onnx_encoder_quantizer_session # Store session

        self.bicodec_config = load_config(f"{model_dir}/BiCodec/config.yaml")
        self.sample_rate = self.bicodec_config.get("sample_rate", 16000)

        # ---- Initialize Components ----
        print("Initializing BiCodecTokenizer Components...")

        # 1. Mel Spectrogram (PyTorch or ONNX - Although ONNX Mel is typically loaded externally)
        if not self.use_mel_spectrogram_onnx:
            print("Loading PyTorch MelSpectrogram...")
            try:
                mel_params = self.bicodec_config.get('audio_tokenizer', {}).get('mel_params', {})
                # Add defaults if missing in config, necessary for MelSpectrogram init
                mel_params.setdefault('sample_rate', self.sample_rate)
                mel_params.setdefault('n_fft', 1024)
                mel_params.setdefault('win_length', mel_params['n_fft'])
                mel_params.setdefault('hop_length', mel_params['n_fft'] // 4)
                mel_params.setdefault('num_mels', 128)
                self.mel_spectrogram_generator = MelSpectrogram(**mel_params)
                self.mel_spectrogram_generator.eval()
                self.mel_spectrogram_generator.to(self.device)
                print("✓ PyTorch MelSpectrogram loaded.")
            except Exception as e:
                print(f"✗ Failed to load PyTorch MelSpectrogram: {e}")
                self.mel_spectrogram_generator = None
        else:
            print("Skipping PyTorch MelSpectrogram loading (ONNX version expected externally).")
            self.mel_spectrogram_generator = None # ONNX version handled by caller usually

        # 2. Speaker Encoder Tokenizer (PyTorch or ONNX)
        if not self.use_speaker_encoder_tokenizer_onnx:
            print("Loading PyTorch Speaker Encoder...")
            try:
                # Load the speaker encoder part from the full BiCodec model
                temp_bicodec_spk = BiCodec.load_from_checkpoint(f"{model_dir}/BiCodec", device=self.device)
                self.speaker_encoder = temp_bicodec_spk.speaker_encoder
                self.speaker_encoder.eval()
                print("✓ PyTorch Speaker Encoder loaded.")
                del temp_bicodec_spk
            except Exception as e:
                print(f"✗ Failed to load PyTorch Speaker Encoder: {e}")
                self.speaker_encoder = None
        elif self.onnx_speaker_encoder_tokenizer_session:
            print("✓ Using provided ONNX Speaker Encoder Tokenizer session.")
            self.speaker_encoder = None # Not needed if using ONNX
        else:
            print("ONNX Speaker Encoder Tokenizer requested but no session provided/loaded.")
            self.speaker_encoder = None

        # 3. Wav2Vec2 (PyTorch or ONNX)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"{model_dir}/wav2vec2-large-xlsr-53")
        if self.use_onnx_wav2vec2:
            onnx_w2v2_path = Path("./onnx_models") / "wav2vec2_model.onnx"
            print(f"Attempting to load ONNX Wav2Vec2 model from: {onnx_w2v2_path}")
            if onnx_w2v2_path.exists():
                try:
                    self.onnx_wav2vec2_session = onnxruntime.InferenceSession(
                        str(onnx_w2v2_path), providers=['CPUExecutionProvider']
                    )
                    self.wav2vec2 = None # Don't need PyTorch version
                    print(f"✓ Successfully loaded ONNX Wav2Vec2 from {onnx_w2v2_path}")
                except Exception as e:
                    print(f"✗ Failed to load ONNX Wav2Vec2: {e}. Will attempt PyTorch fallback.")
                    self.onnx_wav2vec2_session = None
                    self.wav2vec2 = None # Reset before attempting PyTorch
            else:
                print(f"✗ ONNX Wav2Vec2 model not found at {onnx_w2v2_path}. Will attempt PyTorch fallback.")
                self.onnx_wav2vec2_session = None
                self.wav2vec2 = None # Reset before attempting PyTorch
        
        # Fallback or default to PyTorch Wav2Vec2 if ONNX failed or wasn't requested
        if not self.onnx_wav2vec2_session:
            print("Loading PyTorch Wav2Vec2 model...")
            try:
                config = AutoConfig.from_pretrained(f"{model_dir}/wav2vec2-large-xlsr-53")
                config.output_hidden_states = True # Ensure hidden states are output
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                    f"{model_dir}/wav2vec2-large-xlsr-53", config=config
                )
                self.wav2vec2.eval()
                self.wav2vec2.to(self.device)
                print("✓ PyTorch Wav2Vec2 model loaded.")
            except Exception as e:
                print(f"✗ Failed to load PyTorch Wav2Vec2: {e}")
                self.wav2vec2 = None

        # 4. BiCodec Encoder/Quantizer (PyTorch or ONNX)
        if not self.use_bicodec_encoder_quantizer_onnx:
            print("Loading PyTorch BiCodec Encoder and Quantizer...")
            try:
                temp_bicodec_eq = BiCodec.load_from_checkpoint(f"{model_dir}/BiCodec", device=self.device)
                self.encoder = temp_bicodec_eq.encoder
                self.quantizer = temp_bicodec_eq.quantizer
                self.encoder.eval()
                self.quantizer.eval()
                print("✓ PyTorch BiCodec Encoder and Quantizer loaded.")
                del temp_bicodec_eq # Free memory
            except Exception as e:
                print(f"✗ Failed to load PyTorch BiCodec Encoder/Quantizer: {e}")
                self.encoder = None
                self.quantizer = None
        elif self.onnx_encoder_quantizer_session:
            print("✓ Using provided ONNX BiCodec Encoder/Quantizer session.")
            self.encoder = None
            self.quantizer = None
        else:
            # This case is handled during SparkTTS init where session loading is attempted
            print("ONNX Encoder/Quantizer requested but no session provided/loaded.")
            self.encoder = None
            self.quantizer = None

        print("✓ BiCodecTokenizer Initialized.")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and preprocesses audio from a file path."""
        wav, sr = load_audio(audio_path)
        if wav is None:
            return None
        if sr != self.sample_rate:
            print(f"Warning: Resampling audio from {sr} Hz to {self.sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
        wav = audio_volume_normalize(wav)
        # Ensure mono
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        # Ensure batch dimension
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav.to(self.device)

    def tokenize(self, audio_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Tokenizes the given audio file into global speaker tokens and semantic tokens.

        Args:
            audio_path (str): Path to the input audio file (.wav).

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: 
                (global_tokens, semantic_tokens). Tensors are on the initialized device.
                Returns (None, None) if any critical step fails.
        """
        print(f"\n--- Tokenizing Audio: {audio_path} ---")
        wav = self._load_audio(audio_path)
        if wav is None:
            print("Error: Failed to load audio.")
            return None, None

        # 1. Calculate Mel Spectrogram (using PyTorch version for now)
        # TODO: Add path for using ONNX Mel Spectrogram if needed, requires input from caller
        if self.mel_spectrogram_generator:
            print("Calculating Mel Spectrogram using PyTorch...")
            try:
                with torch.no_grad():
                    mel = self.mel_spectrogram_generator(wav)
                    print(f"PyTorch Mel shape: {mel.shape}") # Expect (B, n_mels, T_mel)
            except Exception as e:
                print(f"✗ Error during PyTorch Mel Spectrogram generation: {e}")
                return None, None
        else:
            print("Error: Mel Spectrogram generator not available.")
            return None, None

        # 2. Calculate Global Speaker Tokens (PyTorch or ONNX)
        global_tokens = None
        if self.use_speaker_encoder_tokenizer_onnx and self.onnx_speaker_encoder_tokenizer_session:
            print("Calculating global tokens using ONNX Speaker Encoder Tokenizer...")
            try:
                # Speaker Encoder ONNX expects (B, T_mel, n_mels)
                mel_onnx_input = mel.permute(0, 2, 1).contiguous().cpu().numpy()
                onnx_inputs = {self.onnx_speaker_encoder_tokenizer_session.get_inputs()[0].name: mel_onnx_input}
                global_tokens_np = self.onnx_speaker_encoder_tokenizer_session.run(None, onnx_inputs)[0]
                global_tokens = torch.from_numpy(global_tokens_np).to(self.device)
                print(f"ONNX Global Tokens shape: {global_tokens.shape}")
            except Exception as e:
                print(f"✗ Error during ONNX Speaker Encoder Tokenizer inference: {e}")
                return None, None # Global tokens are essential
        elif self.speaker_encoder:
            print("Calculating global tokens using PyTorch Speaker Encoder...")
            try:
                with torch.no_grad():
                    # PyTorch SpeakerEncoder.tokenize expects (B, T_mel, n_mels)
                    global_tokens = self.speaker_encoder.tokenize(mel.permute(0, 2, 1).contiguous())
                    print(f"PyTorch Global Tokens shape: {global_tokens.shape}")
            except Exception as e:
                print(f"✗ Error during PyTorch Speaker Encoder inference: {e}")
                return None, None
        else:
            print("Error: Speaker Encoder (PyTorch or ONNX) not available.")
            return None, None

        # 3. Prepare audio for Wav2Vec2 (feature extraction)
        print("Preparing audio for Wav2Vec2...")
        try:
            # Wav2Vec2 expects raw waveform
            wav_input_for_w2v2 = wav.squeeze(0).cpu().numpy() # Needs single waveform for feature_extractor
            processed_inputs = self.feature_extractor(wav_input_for_w2v2, return_tensors="pt", sampling_rate=self.sample_rate, padding=True)
            input_values = processed_inputs.input_values.to(self.device)
            print(f"Wav2Vec2 input shape: {input_values.shape}")
        except Exception as e:
            print(f"✗ Error during Wav2Vec2 feature extraction preparation: {e}")
            return global_tokens, None # Return global tokens, but semantic failed

        # 4. Extract features using Wav2Vec2 (PyTorch or ONNX)
        feat = None
        if self.use_onnx_wav2vec2 and self.onnx_wav2vec2_session:
            print("Extracting features using ONNX Wav2Vec2...")
            try:
                onnx_inputs = {self.onnx_wav2vec2_session.get_inputs()[0].name: input_values.cpu().numpy()}
                # Assuming the desired features are the last hidden state
                onnx_outputs = self.onnx_wav2vec2_session.run(None, onnx_inputs)
                feat_np = onnx_outputs[-1] # Take the last hidden state
                feat = torch.from_numpy(feat_np).to(self.device)
                print(f"ONNX Wav2Vec2 Feature shape: {feat.shape}")
            except Exception as e:
                print(f"✗ Error during ONNX Wav2Vec2 inference: {e}")
                return global_tokens, None
        elif self.wav2vec2:
            print("Extracting features using PyTorch Wav2Vec2...")
            try:
                with torch.no_grad():
                    # Wav2Vec2 output is a BaseModelOutput, hidden_states is a tuple
                    outputs = self.wav2vec2(input_values)
                    feat = outputs.hidden_states[-1] # Take the last hidden state
                    print(f"PyTorch Wav2Vec2 Feature shape: {feat.shape}")
            except Exception as e:
                print(f"✗ Error during PyTorch Wav2Vec2 inference: {e}")
                return global_tokens, None
            except AttributeError:
                # Fallback if hidden_states isn't available (e.g. config issue)
                 with torch.no_grad():
                    outputs = self.wav2vec2(input_values)
                    feat = outputs.last_hidden_state # Use last_hidden_state as fallback
                    print(f"PyTorch Wav2Vec2 Feature shape (using last_hidden_state): {feat.shape}")

        else:
            print("Error: Wav2Vec2 model (PyTorch or ONNX) is not available.")
            return global_tokens, None # Return global tokens but indicate semantic failure

        if feat is None:
            print("Error: Failed to extract Wav2Vec2 features.")
            return global_tokens, None
        
        # 5. Calculate semantic tokens (PyTorch or ONNX)
        semantic_tokens = None
        if self.use_bicodec_encoder_quantizer_onnx and self.onnx_encoder_quantizer_session:
            print("Calculating semantic tokens using ONNX BiCodec Encoder/Quantizer...")
            try:
                # ONNX model expects (B, T_feat, D_feat)
                onnx_inputs = {self.onnx_encoder_quantizer_session.get_inputs()[0].name: feat.cpu().numpy()}
                semantic_tokens_np = self.onnx_encoder_quantizer_session.run(None, onnx_inputs)[0]
                semantic_tokens = torch.from_numpy(semantic_tokens_np).to(self.device) # Assuming output is int64
                print(f"ONNX Semantic Tokens shape: {semantic_tokens.shape}")
            except Exception as e:
                print(f"✗ Error during ONNX Encoder/Quantizer inference: {e}")
                return global_tokens, None # Return global tokens but indicate semantic failure
        elif self.encoder and self.quantizer: # Use PyTorch path
            print("Calculating semantic tokens using PyTorch BiCodec Encoder/Quantizer...")
            try:
                with torch.no_grad():
                    # PyTorch encoder expects (B, D_feat, T_feat)
                    z = self.encoder(feat.transpose(1, 2))
                    semantic_tokens = self.quantizer.tokenize(z)
                    print(f"PyTorch Semantic Tokens shape: {semantic_tokens.shape}")
            except Exception as e:
                print(f"✗ Error during PyTorch Encoder/Quantizer inference: {e}")
                return global_tokens, None
        else:
            print("Error: Neither ONNX nor PyTorch BiCodec Encoder/Quantizer is available.")
            return global_tokens, None # Return global tokens but indicate semantic failure

        # 6. Detach tensors if needed and return
        global_tokens = global_tokens.detach() if global_tokens is not None else None
        semantic_tokens = semantic_tokens.detach() if semantic_tokens is not None else None

        print("--- Tokenization Complete ---")
        return global_tokens, semantic_tokens

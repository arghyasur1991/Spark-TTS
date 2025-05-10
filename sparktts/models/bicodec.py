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
from typing import Dict, Any
from omegaconf import DictConfig
from safetensors.torch import load_file

from sparktts.utils.file import load_config
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
from sparktts.modules.encoder_decoder.feat_encoder import Encoder
from sparktts.modules.encoder_decoder.feat_decoder import Decoder
from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize


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

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.
        
        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

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
        mel_for_speaker_encoder_full = self.mel_transformer(batch["ref_wav"].to(feat.device)).squeeze(1) # This might be (B, 301, T)

        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        
        # Conditional Speaker Encoding
        if self.use_speaker_encoder_tokenizer_onnx and self.onnx_speaker_encoder_tokenizer_session:
            print("BiCodec: Using ONNX Speaker Encoder Tokenizer.")
            try:
                # Ensure mel spec has 128 channels for the ONNX model
                # Original ONNX model expects (B, 128, T)
                if mel_for_speaker_encoder_full.shape[1] == 301:
                    print(f"BiCodec: Original mel channels from mel_for_speaker_encoder_full: {mel_for_speaker_encoder_full.shape[1]}. Slicing to 128 for ONNX Speaker Encoder.")
                    mel_for_onnx = mel_for_speaker_encoder_full[:, :128, :].contiguous()
                elif mel_for_speaker_encoder_full.shape[1] == 128:
                    print(f"BiCodec: Mel channels from mel_for_speaker_encoder_full are already 128: {mel_for_speaker_encoder_full.shape[1]}.")
                    mel_for_onnx = mel_for_speaker_encoder_full # No slice needed, but ensure contiguous
                else:
                    # This case should ideally not be hit if the ONNX error consistently reports C:301 from a 301-channel input
                    print(f"BiCodec: WARNING - Unexpected number of mel channels in mel_for_speaker_encoder_full: {mel_for_speaker_encoder_full.shape[1]}. Raising ValueError.")
                    raise ValueError(f"Unexpected number of mel channels for ONNX: {mel_for_speaker_encoder_full.shape[1]}. Expected 128 or 301 to slice.")

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
                if mel_for_speaker_encoder_full.shape[1] == 301:
                    print("BiCodec: PyTorch fallback - Slicing 301 mels to 128 for PyTorch Speaker Encoder.")
                    mel_for_pytorch_encoder = mel_for_speaker_encoder_full[:, :128, :].contiguous()
                else: # Assuming it's already 128 or compatible
                    mel_for_pytorch_encoder = mel_for_speaker_encoder_full
                global_tokens = self.speaker_encoder.tokenize(mel_for_pytorch_encoder.transpose(1, 2))
        else:
            if self.use_speaker_encoder_tokenizer_onnx:
                print("BiCodec: ONNX Speaker Encoder Tokenizer session not available. Falling back to PyTorch.")
            else:
                print("BiCodec: Using PyTorch Speaker Encoder Tokenizer.")
            
            # Ensure PyTorch path gets the correct number of channels (128)
            # This assumes self.speaker_encoder expects (B, T, 128) after transpose
            if mel_for_speaker_encoder_full.shape[1] == 301:
                print("BiCodec: PyTorch path - Slicing 301 mels to 128 for PyTorch Speaker Encoder.")
                mel_for_pytorch_encoder = mel_for_speaker_encoder_full[:, :128, :].contiguous()
            elif mel_for_speaker_encoder_full.shape[1] == 128:
                mel_for_pytorch_encoder = mel_for_speaker_encoder_full
            else:
                 raise ValueError(f"Unexpected number of mel channels for PyTorch SpeakerEncoder: {mel_for_speaker_encoder_full.shape[1]}. Expected 128 or 301 to slice.")
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

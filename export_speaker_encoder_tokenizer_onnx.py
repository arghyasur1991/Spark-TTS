"""
Exports the Speaker Encoder Tokenizer component of SparkTTS to ONNX.

This script loads a pre-trained BiCodec model to access its SpeakerEncoder.
It then wraps the `tokenize` method of the SpeakerEncoder (which includes quantization)
into an ONNX-exportable module. The resulting ONNX model takes a Mel spectrogram
as input and outputs global speaker token IDs.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from sparktts.models.bicodec import BiCodec # To access SpeakerEncoder from BiCodec
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder

class SpeakerEncoderTokenizerONNXWrapper(nn.Module):
    """
    Wrapper for the SpeakerEncoder's tokenize method for ONNX export.
    This module takes Mel spectrograms and outputs global speaker token IDs.
    It relies on the `onnx_export_mode=True` flag in the underlying
    `SpeakerEncoder.tokenize` and `FactorizedVectorQuantize.forward` methods.
    """
    def __init__(self, speaker_encoder_model: SpeakerEncoder):
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
                          Typically (B, 1, N_tokens) for a single-level FSQ.
        """
        # The SpeakerEncoder.tokenize method should internally handle the onnx_export_mode
        # to ensure its quantizer (FSQ) uses export-friendly operations.
        return self.speaker_encoder.tokenize(mels, onnx_export_mode=True)

def export_speaker_encoder_tokenizer_to_onnx(
    model_dir: Path,
    output_path: Path,
    opset_version: int,
    device_str: str,
    dummy_mel_batch: int,
    dummy_mel_time: int,
    dummy_mel_channels: int,
):
    """
    Exports the SparkTTS Speaker Encoder Tokenizer to an ONNX model.

    Args:
        model_dir: Path to the base SparkTTS model directory (containing BiCodec subdir).
        output_path: Path to save the exported ONNX model.
        opset_version: ONNX opset version.
        device_str: Device to run export on (e.g., 'cpu', 'cuda:0').
        dummy_mel_batch: Batch size for the dummy Mel input.
        dummy_mel_time: Time steps for the dummy Mel input.
        dummy_mel_channels: Number of Mel features/channels for the dummy Mel input.
    """
    print(f"[INFO] Starting Speaker Encoder Tokenizer ONNX export process...")
    print(f"[INFO]   Base model directory (for BiCodec): {model_dir}")
    print(f"[INFO]   Output ONNX path: {output_path}")
    print(f"[INFO]   Opset version: {opset_version}")
    print(f"[INFO]   Device: {device_str}")

    export_device = torch.device(device_str)

    # 1. Load the pre-trained BiCodec model to get the SpeakerEncoder
    bicodec_model_subdir = model_dir / "BiCodec"
    print(f"[INFO] Loading BiCodec model from: {bicodec_model_subdir} to access SpeakerEncoder")
    if not bicodec_model_subdir.exists():
        print(f"[ERROR] BiCodec model directory not found at {bicodec_model_subdir}. Please check the path.")
        return
    
    try:
        # Load BiCodec model onto the specified export_device
        bicodec_model = BiCodec.load_from_checkpoint(model_dir=bicodec_model_subdir, device=export_device)
        speaker_encoder_module = bicodec_model.speaker_encoder
        speaker_encoder_module.eval() # Ensure speaker encoder is in eval mode
        print("[INFO] SpeakerEncoder module extracted and set to eval mode.")
    except FileNotFoundError:
        print(f"[ERROR] BiCodec model files (e.g., config or checkpoint) not found in {bicodec_model_subdir}.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load BiCodec model or extract SpeakerEncoder: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Instantiate the ONNX wrapper
    try:
        onnx_exportable_tokenizer = SpeakerEncoderTokenizerONNXWrapper(speaker_encoder_module).to(export_device)
        onnx_exportable_tokenizer.eval()
    except Exception as e:
        print(f"[ERROR] Failed to instantiate SpeakerEncoderTokenizerONNXWrapper: {e}")
        return

    # 3. Prepare dummy Mel input
    # ECAPA-TDNN (common for speaker encoders) expects (B, T, F)
    # where F is the feature dimension (e.g., 128 for ECAPA in BiCodec).
    dummy_mels = torch.randn(
        dummy_mel_batch,    
        dummy_mel_time,     
        dummy_mel_channels, 
        device=export_device
    ).contiguous() # Ensure contiguous tensor for some ONNX ops if needed
    print(f"[INFO] Using dummy_mels input shape (B, T_mel, D_mel_feat): {dummy_mels.shape}")

    # Test forward pass with the PyTorch ONNX wrapper
    print("[INFO] Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            pytorch_output_tokens = onnx_exportable_tokenizer(dummy_mels)
        print(f"[INFO] PyTorch ONNX wrapper test forward pass successful. Output tokens shape: {pytorch_output_tokens.shape}")
        if pytorch_output_tokens.numel() == 0:
            print("[ERROR] PyTorch wrapper produced empty tokens. Check dummy input or model logic.")
            return
    except Exception as e:
        print(f"[ERROR] PyTorch ONNX wrapper test forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Export to ONNX
    input_names = ["mel_spectrogram"]
    output_names = ["global_tokens"]
    
    dynamic_axes = {
        input_names[0]: {0: 'batch_size', 1: 'mel_time_steps'}, # D_mel_channels (dim 2) is usually fixed
        output_names[0]: {0: 'batch_size'} 
    }
    # The token sequence length dimension(s) can also be dynamic if the quantizer output length varies.
    # For a typical FSQ output like (B, 1, N_tokens) or (B, N_tokens), N_tokens is usually fixed per speaker utterance.
    # If output is (B, N_quant_levels, T_quant_tokens), T_quant_tokens might be dynamic.
    if pytorch_output_tokens.ndim == 3: # e.g., (B, N_quant_levels, T_quant_tokens)
        dynamic_axes[output_names[0]][2] = 'token_seq_len'
    elif pytorch_output_tokens.ndim == 2: # e.g. (B, T_quant_tokens_flat)
        dynamic_axes[output_names[0]][1] = 'token_seq_len_flat'
    print(f"[INFO] Using dynamic axes: {dynamic_axes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting Speaker Encoder Tokenizer to ONNX: {output_path}")
    try:
        torch.onnx.export(
            onnx_exportable_tokenizer,
            dummy_mels,
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
        return

    # 5. Verify the ONNX Model
    print("[INFO] --- Starting ONNX Model Verification ---")
    try:
        print("[INFO] Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        onnx_input_name = ort_session.get_inputs()[0].name
        print(f"[INFO] ONNX model loaded. Input name: {onnx_input_name}")

        ort_inputs = {onnx_input_name: dummy_mels.cpu().numpy()}
        print("[INFO] Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_tokens_np = ort_outputs[0]
        print(f"[INFO] ONNX Runtime inference successful. Output tokens shape: {onnx_output_tokens_np.shape}")
        
        np.testing.assert_allclose(
            pytorch_output_tokens.cpu().detach().numpy(), 
            onnx_output_tokens_np, 
            rtol=1e-03, 
            atol=1e-05
        )
        print("[INFO] ONNX Runtime outputs match PyTorch outputs numerically (within tolerance). Verification successful.")
    except Exception as e:
        print(f"[ERROR] Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()
    print("[INFO] Speaker Encoder Tokenizer ONNX export and verification process complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Export SparkTTS Speaker Encoder Tokenizer to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", 
        type=Path, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B), which should contain the BiCodec subdirectory."
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        required=True, # Making output path required for clarity
        help="Full path to save the exported Speaker Encoder Tokenizer ONNX model (e.g., ./onnx_models/speaker_encoder_tokenizer.onnx)."
    )
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--dummy_mel_batch", type=int, default=1, help="Batch size for dummy Mel input.")
    parser.add_argument("--dummy_mel_time", type=int, default=200, help="Time steps for dummy Mel input (e.g., corresponding to a few seconds of audio).")
    parser.add_argument("--dummy_mel_channels", type=int, default=128, help="Number of Mel features/channels (e.g., 128 for ECAPA-TDNN based speaker encoders).")

    args = parser.parse_args()

    export_speaker_encoder_tokenizer_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output_path,
        opset_version=args.opset_version,
        device_str=args.device,
        dummy_mel_batch=args.dummy_mel_batch,
        dummy_mel_time=args.dummy_mel_time,
        dummy_mel_channels=args.dummy_mel_channels,
    )

if __name__ == "__main__":
    main() 
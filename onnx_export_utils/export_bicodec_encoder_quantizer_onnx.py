"""
Exports the Encoder and Quantizer parts of the BiCodec model to ONNX.

This script loads a pre-trained BiCodec model, extracts its encoder and quantizer
components, wraps them in an ONNX-exportable module, and then exports this
wrapper to an ONNX file. The resulting ONNX model takes features (e.g., from Wav2Vec2)
and outputs semantic token IDs.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from sparktts.models.bicodec import BiCodec
from sparktts.modules.encoder_decoder.feat_encoder import Encoder as BiCodecEncoder # Renamed for clarity
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
# from sparktts.utils.file import load_config # Not strictly needed if BiCodec.load_from_checkpoint handles config loading

class BiCodecEncoderQuantizerONNXWrapper(nn.Module):
    """
    Wrapper for BiCodec's encoder and quantizer for ONNX export.
    Takes features (e.g., from Wav2Vec2) and outputs semantic token IDs.
    """
    def __init__(self, encoder_model: BiCodecEncoder, quantizer_model: FactorizedVectorQuantize):
        super().__init__()
        self.encoder = encoder_model
        self.quantizer = quantizer_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Input features, e.g., from Wav2Vec2.
                                     Expected shape: (B, T_feat, D_feat), e.g., (1, 98, 1024).
                                     The BiCodec encoder expects (B, D_feat, T_feat), so a transpose is done internally.

        Returns:
            torch.Tensor: Semantic token IDs. 
                          Shape: (B, num_quantizers, T_quantized), e.g., (1, 8, 49).
        """
        # BiCodec's Encoder expects (B, D_feat, T_feat)
        # Input `features` are (B, T_feat, D_feat) from Wav2Vec2
        features_transposed = features.transpose(1, 2)
        
        # Encoder output `z` has shape (B, D_encoded, T_quantized)
        z = self.encoder(features_transposed)
        
        # Quantizer.tokenize expects (B, D_encoded, T_quantized) and returns (B, num_quantizers, T_quantized)
        semantic_tokens = self.quantizer.tokenize(z)
        return semantic_tokens

def export_encoder_quantizer_to_onnx(
    model_dir: Path,
    output_path: Path,
    opset_version: int,
    dummy_feat_seq_len: int,
    dummy_feat_dim: int,
    device_str: str = "cpu",
):
    """
    Exports the BiCodec encoder and quantizer to an ONNX model.

    Args:
        model_dir: Path to the base SparkTTS model directory containing BiCodec subdir.
        output_path: Path to save the exported ONNX model.
        opset_version: ONNX opset version.
        dummy_feat_seq_len: Sequence length for dummy input features.
        dummy_feat_dim: Feature dimension for dummy input features.
        device_str: Device to run export on (e.g., 'cpu', 'cuda:0'). Defaults to "cpu".
    """
    print(f"[INFO] Starting BiCodec Encoder+Quantizer ONNX export process...")
    print(f"[INFO]   Base model directory: {model_dir}")
    print(f"[INFO]   Output ONNX path: {output_path}")
    print(f"[INFO]   Opset version: {opset_version}")
    print(f"[INFO]   Device: {device_str}")

    export_device = torch.device(device_str)

    # 1. Load PyTorch BiCodec model and extract encoder/quantizer
    bicodec_subdir = model_dir / "BiCodec"
    print(f"[INFO] Loading BiCodec model from: {bicodec_subdir}")
    try:
        # BiCodec.load_from_checkpoint typically handles moving model to the specified device.
        bicodec_model = BiCodec.load_from_checkpoint(bicodec_subdir, device=export_device)
        bicodec_model.eval()
        encoder = bicodec_model.encoder
        quantizer = bicodec_model.quantizer
        print("[INFO] BiCodec model loaded and encoder/quantizer extracted successfully.")
    except FileNotFoundError:
        print(f"[ERROR] BiCodec model directory or necessary files not found in {bicodec_subdir}. Please check the path.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading BiCodec model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Create ONNX Wrapper
    try:
        onnx_wrapper = BiCodecEncoderQuantizerONNXWrapper(encoder, quantizer).to(export_device)
        onnx_wrapper.eval()
    except Exception as e:
        print(f"[ERROR] Failed to instantiate BiCodecEncoderQuantizerONNXWrapper: {e}")
        return

    # 3. Create Dummy Input
    # Expected shape for Wav2Vec2-like features: (Batch, SeqLen_Features, FeatureDim)
    dummy_input_features = torch.randn(1, dummy_feat_seq_len, dummy_feat_dim, device=export_device)
    print(f"[INFO] Dummy input features shape: {dummy_input_features.shape}")

    # Run wrapper for a dummy output (and to get output shape for dynamic_axes)
    try:
        with torch.no_grad():
            dummy_output_tokens = onnx_wrapper(dummy_input_features)
        print(f"[INFO] Wrapper dummy output tokens shape: {dummy_output_tokens.shape}") 
        print(f"[INFO] Wrapper dummy output tokens dtype: {dummy_output_tokens.dtype}")
        if dummy_output_tokens.ndim != 3:
            print(f"[WARNING] Expected 3D output (B, N_quant, T_quant), got {dummy_output_tokens.ndim}D. Check model logic.")

    except Exception as e:
        print(f"[ERROR] Failed during wrapper dummy forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Export to ONNX
    input_names = ["features"]
    output_names = ["semantic_tokens"]
    
    # Dynamic axes: Batch fixed (1 for export), Sequence dynamic, Dim fixed for features.
    # For tokens: Batch fixed, NumQuantizers fixed, QuantizedSequence dynamic.
    dynamic_axes = {
        input_names[0]: {1: "feature_sequence_length"}, 
        output_names[0]: {2: "quantized_token_sequence_length"} 
    }
    # If batch is also dynamic (e.g. for batch_size > 1 during inference):
    # dynamic_axes[input_names[0]][0] = "batch_size"
    # dynamic_axes[output_names[0]][0] = "batch_size"
    print(f"[INFO] Using dynamic axes: {dynamic_axes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting model to {output_path} with opset {opset_version}...")
    try:
        torch.onnx.export(
            onnx_wrapper,
            dummy_input_features,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True, # Ensure parameters are embedded
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print("[INFO] ONNX export successful.")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Verify ONNX Model
    print("[INFO] --- Starting ONNX Model Verification ---")
    try:
        print("[INFO] Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        onnx_input_name = ort_session.get_inputs()[0].name
        print(f"[INFO] ONNX model loaded. Input name: {onnx_input_name}")

        ort_inputs = {onnx_input_name: dummy_input_features.cpu().numpy()}
        print("[INFO] Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_tokens = ort_outputs[0]
        print(f"[INFO] ONNX Runtime inference successful. Output tokens shape: {onnx_output_tokens.shape}")

        np.testing.assert_allclose(
            dummy_output_tokens.cpu().numpy(), onnx_output_tokens, rtol=1e-3, atol=1e-5
        )
        print("[INFO] ONNX model verification successful (outputs match PyTorch within tolerance).")

    except Exception as e:
        print(f"[ERROR] ONNX model verification failed: {e}")
        import traceback
        traceback.print_exc()
    print("[INFO] BiCodec Encoder+Quantizer ONNX export and verification process complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Export BiCodec Encoder and Quantizer to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("pretrained_models/Spark-TTS-0.5B"),
        help="Path to the base SparkTTS model directory containing the BiCodec subdirectory.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("onnx_models/bicodec_encoder_quantizer.onnx"),
        help="Output path for the exported ONNX file.",
    )
    parser.add_argument(
        "--opset_version", 
        type=int, 
        default=17, # Defaulting to 17 as previously hardcoded, but now configurable
        help="ONNX opset version for export."
    )
    parser.add_argument(
        "--dummy_feat_seq_len", 
        type=int, 
        default=98, 
        help="Sequence length for dummy input features (e.g., typical output length from Wav2Vec2 for a few seconds of audio)."
    )
    parser.add_argument(
        "--dummy_feat_dim", 
        type=int, 
        default=1024, 
        help="Feature dimension for dummy input features (e.g., 1024 for Wav2Vec2-large)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda"], # Add more if needed, e.g. "cuda:0"
        help="Device to run the export on. Note: some operations might fallback to CPU during ONNX export regardless."
    )
    args = parser.parse_args()

    export_encoder_quantizer_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output_path,
        opset_version=args.opset_version,
        dummy_feat_seq_len=args.dummy_feat_seq_len,
        dummy_feat_dim=args.dummy_feat_dim,
        device_str=args.device,
    )

if __name__ == "__main__":
    main()
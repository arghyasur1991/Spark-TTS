# Spark-TTS/export_bicodec_encoder_quantizer_onnx.py
import torch
import torch.nn as nn
import numpy as np
import onnxruntime
from pathlib import Path
import argparse
import os

# Assuming SparkTTS modules are accessible from this script's location
from sparktts.models.bicodec import BiCodec # To load the full model first
from sparktts.modules.encoder_decoder.feat_encoder import Encoder # For type hint
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize # For type hint
from sparktts.utils.file import load_config

# Ensure the output directory exists
os.makedirs("onnx_models", exist_ok=True)

class BiCodecEncoderQuantizerONNXWrapper(nn.Module):
    """
    Wrapper for BiCodec's encoder and quantizer for ONNX export.
    Takes Wav2Vec2-like features and outputs semantic token IDs.
    """
    def __init__(self, encoder_model: Encoder, quantizer_model: FactorizedVectorQuantize):
        super().__init__()
        self.encoder = encoder_model
        self.quantizer = quantizer_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Input features, e.g., from Wav2Vec2.
                                     Expected shape: (B, T_feat, D_feat) like (1, T, 1024)
        Returns:
            torch.Tensor: Semantic token IDs. Shape: (B, num_quantizers, T_quantized) like (1, 8, T_q)
        """
        # Encoder expects (B, D_feat, T_feat)
        # Quantizer.tokenize expects (B, D_encoded, T_quantized) from encoder
        z = self.encoder(features.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        return semantic_tokens

def main():
    parser = argparse.ArgumentParser(description="Export BiCodec Encoder+Quantizer to ONNX")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B", # Adjust path relative to Spark-TTS folder
        help="Path to the base SparkTTS model directory containing BiCodec subdir",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="onnx_models/bicodec_encoder_quantizer.onnx",
        help="Output ONNX file name",
    )
    parser.add_argument(
        "--dummy_feat_seq_len", type=int, default=98, # Example length from wav2vec2 output
        help="Sequence length for dummy input features"
    )
    parser.add_argument(
        "--dummy_feat_dim", type=int, default=1024, # Wav2Vec2 Large feature dim
        help="Feature dimension for dummy input features"
    )
    args = parser.parse_args()

    # --- 1. Load PyTorch BiCodec model ---
    # Use CPU for export consistency
    device = torch.device("cpu")
    bicodec_model_dir = Path(args.model_dir) / "BiCodec"

    print(f"Loading BiCodec model from: {bicodec_model_dir}")
    try:
        bicodec_model = BiCodec.load_from_checkpoint(bicodec_model_dir, device=device)
        bicodec_model.eval()
        encoder = bicodec_model.encoder
        quantizer = bicodec_model.quantizer
        print("BiCodec model loaded successfully.")
    except Exception as e:
        print(f"Error loading BiCodec model: {e}")
        return

    # --- 2. Create Wrapper ---
    onnx_wrapper = BiCodecEncoderQuantizerONNXWrapper(encoder, quantizer)
    onnx_wrapper.eval()

    # --- 3. Create Dummy Input ---
    # Shape: (Batch, SeqLen_Features, FeatureDim) e.g., (1, 98, 1024)
    dummy_input_features = torch.randn(1, args.dummy_feat_seq_len, args.dummy_feat_dim, device=device)
    print(f"Dummy input features shape: {dummy_input_features.shape}")

    # --- 4. Run Wrapper for Dummy Output ---
    with torch.no_grad():
        dummy_output_tokens = onnx_wrapper(dummy_input_features)
    print(f"Dummy output tokens shape: {dummy_output_tokens.shape}") # Should be (B, num_quantizers, T_quantized) e.g., (1, 8, 49)
    print(f"Dummy output tokens dtype: {dummy_output_tokens.dtype}") # Should be torch.int64

    # --- 5. Export to ONNX ---
    input_names = ["features"]
    output_names = ["semantic_tokens"]
    dynamic_axes = {
        "features": {1: "sequence_length"},  # Batch fixed, Sequence dynamic, Dim fixed
        "semantic_tokens": {2: "quantized_sequence_length"} # Batch fixed, NumQuantizers fixed, Sequence dynamic
    }

    print(f"Exporting model to {args.output_name}...")
    try:
        torch.onnx.export(
            onnx_wrapper,
            dummy_input_features,
            args.output_name,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17, # Use a recent opset
            do_constant_folding=True,
            export_params=True,
        )
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return

    # --- 6. Verify ONNX Model and Save Dummy Data ---
    print("Verifying ONNX model...")
    try:
        ort_session = onnxruntime.InferenceSession(args.output_name, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_features.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_tokens = ort_outputs[0]

        # Compare outputs
        np.testing.assert_allclose(
            dummy_output_tokens.cpu().numpy(), onnx_output_tokens, rtol=1e-3, atol=1e-5
        )
        print("ONNX model verification successful (outputs match PyTorch).")

        # Save dummy data for C# testing
        features_npy_path = "onnx_models/dummy_encoder_quantizer_input_features.npy"
        tokens_npy_path = "onnx_models/dummy_encoder_quantizer_output_tokens.npy"
        np.save(features_npy_path, dummy_input_features.cpu().numpy())
        np.save(tokens_npy_path, dummy_output_tokens.cpu().numpy()) # Save pytorch output as golden
        print(f"Saved dummy input features to {features_npy_path}")
        print(f"Saved dummy output tokens to {tokens_npy_path}")

    except Exception as e:
        print(f"ONNX model verification or dummy data saving failed: {e}")

if __name__ == "__main__":
    main()
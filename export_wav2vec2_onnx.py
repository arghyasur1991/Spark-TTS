import argparse
import shutil
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2HiddenStatesExtractor(nn.Module):
    """
    Wrapper for Wav2Vec2Model to directly output the tuple of hidden states.
    This is useful for ONNX export when only hidden states are required.
    """

    def __init__(self, wav2vec2_model: Wav2Vec2Model):
        super().__init__()
        # Ensure the underlying model is configured to output hidden states
        if not wav2vec2_model.config.output_hidden_states:
            print("[INFO] Forcing 'output_hidden_states=True' in Wav2Vec2Model config.")
            wav2vec2_model.config.output_hidden_states = True
        self.model = wav2vec2_model

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass that returns only the hidden states.
        """
        outputs = self.model(input_values)
        return outputs.hidden_states


def export_wav2vec2_to_onnx(
    hf_model_path: Path,
    output_path: Path,
    opset_version: int,
    batch_size: int,
    sequence_length: int,
):
    """
    Exports a Hugging Face Wav2Vec2Model to ONNX format, specifically for extracting hidden states.

    Args:
        hf_model_path: Path to the Hugging Face Wav2Vec2Model directory.
        output_path: Path to save the exported ONNX model.
        opset_version: ONNX opset version.
        batch_size: Batch size for the dummy input.
        sequence_length: Sequence length for the dummy audio input.
    """
    print(f"[INFO] Starting Wav2Vec2 ONNX export process...")
    print(f"[INFO]   Hugging Face model path: {hf_model_path}")
    print(f"[INFO]   Output ONNX path: {output_path}")
    print(f"[INFO]   Opset version: {opset_version}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean the specific ONNX file if it exists
    if output_path.exists():
        print(f"[INFO] Removing existing ONNX file: {output_path}")
        output_path.unlink()

    print(f"[INFO] Loading Wav2Vec2Model from: {hf_model_path}")
    try:
        # Load the config and ensure output_hidden_states is True for the PyTorch model
        pytorch_config = AutoConfig.from_pretrained(hf_model_path)
        pytorch_config.output_hidden_states = True  # Explicitly set for clarity
        pytorch_model = Wav2Vec2Model.from_pretrained(hf_model_path, config=pytorch_config)
        pytorch_model.eval()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load Wav2Vec2Model or FeatureExtractor: {e}")
        return

    # Wrap the PyTorch model for ONNX export
    onnx_exportable_model = Wav2Vec2HiddenStatesExtractor(pytorch_model)
    onnx_exportable_model.eval() # Ensure wrapper is also in eval mode

    print("[INFO] Creating dummy input using Wav2Vec2FeatureExtractor...")
    raw_dummy_audio = [np.random.randn(sequence_length).astype(np.float32) for _ in range(batch_size)]
    try:
        processed_inputs = feature_extractor(
            raw_dummy_audio,
            return_tensors="pt",
            sampling_rate=feature_extractor.sampling_rate,
            padding="longest", # Use consistent padding strategy
        )
    except Exception as e:
        print(f"[ERROR] Failed during feature extraction for dummy input: {e}")
        return
        
    dummy_input_values_pt = processed_inputs.input_values
    print(f"[INFO] Processed dummy input shape: {dummy_input_values_pt.shape}")

    # Get PyTorch model output for verification
    try:
        with torch.no_grad():
            dummy_pytorch_outputs_tuple = onnx_exportable_model(dummy_input_values_pt)
        num_hidden_states = len(dummy_pytorch_outputs_tuple)
        print(f"[INFO] PyTorch model will output {num_hidden_states} hidden states.")
        if num_hidden_states == 0:
            print("[ERROR] PyTorch model produced zero hidden states. Check model configuration.")
            return
    except Exception as e:
        print(f"[ERROR] Failed during PyTorch model inference with dummy input: {e}")
        return

    input_names = ["input_values"]
    output_names = [f"hidden_state_{i}" for i in range(num_hidden_states)]

    dynamic_axes = {input_names[0]: {0: "batch_size", 1: "sequence_length"}}
    for name in output_names:
        # Assuming hidden states might have different sequence length than input audio due to convolutions
        dynamic_axes[name] = {0: "batch_size", 1: "feat_sequence_length"}

    print(f"[INFO] Exporting model to ONNX: {output_path}")
    try:
        torch.onnx.export(
            onnx_exportable_model,
            dummy_input_values_pt,
            str(output_path), # torch.onnx.export expects string path
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
        print("[INFO] ONNX export successful.")
    except Exception as e:
        print(f"[ERROR] torch.onnx.export failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("[INFO] --- Starting ONNX Model Verification ---")
    try:
        print("[INFO] Loading ONNX model for verification...")
        # It's good practice to specify providers, can help catch compatibility issues early.
        ort_session = onnxruntime.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        onnx_model_input_name = ort_session.get_inputs()[0].name
        print(f"[INFO] ONNX model loaded. Input name: {onnx_model_input_name}")

        ort_inputs_np = {onnx_model_input_name: dummy_input_values_pt.cpu().numpy()}
        print("[INFO] Running ONNX Runtime inference...")
        ort_outputs_np_list = ort_session.run(None, ort_inputs_np)
        print(f"[INFO] ONNX Runtime inference successful. Number of outputs: {len(ort_outputs_np_list)}")

        if num_hidden_states != len(ort_outputs_np_list):
            print(f"[ERROR] Mismatch in number of outputs: PyTorch={num_hidden_states}, ONNX={len(ort_outputs_np_list)}")
            return

        print(f"[INFO] Comparing {num_hidden_states} hidden states from PyTorch and ONNX...")
        all_match = True
        for i in range(num_hidden_states):
            try:
                np.testing.assert_allclose(
                    dummy_pytorch_outputs_tuple[i].cpu().numpy(),
                    ort_outputs_np_list[i],
                    rtol=1e-3,  # Relaxed tolerance a bit
                    atol=1e-5,
                )
            except AssertionError as error_msg:
                print(f"[ERROR] Mismatch in hidden state {i}:\n{error_msg}")
                all_match = False
        
        if all_match:
            print("[INFO] ONNX Runtime outputs match PyTorch outputs numerically (within tolerance). Verification successful.")
        else:
            print("[ERROR] ONNX Runtime outputs DO NOT match PyTorch outputs. Verification failed.")

    except Exception as e:
        print(f"[ERROR] Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()
    print("[INFO] Wav2Vec2 ONNX export and verification process complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Export Hugging Face Wav2Vec2Model to ONNX for hidden state extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf_model_path",
        type=Path,
        required=True,
        help="Path to the Hugging Face Wav2Vec2Model directory (e.g., ./pretrained_models/Spark-TTS-0.5B/wav2vec2-large-xlsr-53).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("onnx_models/wav2vec2_model.onnx"),
        help="Full path to save the exported ONNX model.",
    )
    parser.add_argument(
        "--opset_version", type=int, default=14, help="ONNX opset version."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for dummy input."
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=16000,
        help="Sequence length for dummy audio input (e.g., 16000 for 1 sec at 16kHz).",
    )

    args = parser.parse_args()

    export_wav2vec2_to_onnx(
        hf_model_path=args.hf_model_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )

if __name__ == "__main__":
    main() 
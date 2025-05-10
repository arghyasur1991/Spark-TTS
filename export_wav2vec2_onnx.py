import torch
import torch.nn as nn
import argparse
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoConfig
import os
import numpy as np 
from typing import Tuple
import shutil # For cleaning output directory if needed
import onnx # For checking the ONNX model
import onnxruntime # For running and verifying the ONNX model

class Wav2Vec2HiddenStatesExtractor(nn.Module):
    """
    Wrapper for Wav2Vec2Model to directly output the tuple of hidden states.
    """
    def __init__(self, wav2vec2_model: Wav2Vec2Model):
        super().__init__()
        # Ensure the underlying model is configured to output hidden states
        if not wav2vec2_model.config.output_hidden_states:
            print("Warning: Underlying Wav2Vec2Model config does not have output_hidden_states=True. Forcing it.")
            wav2vec2_model.config.output_hidden_states = True
        self.model = wav2vec2_model

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(input_values)
        return outputs.hidden_states

def main():
    parser = argparse.ArgumentParser(description="Export Hugging Face Wav2Vec2Model to ONNX for hidden state extraction using torch.onnx.export.")
    parser.add_argument("--hf_model_path", type=str, required=True,
                        help="Path to the Hugging Face Wav2Vec2Model directory (e.g., ./pretrained_models/Spark-TTS-0.5B/wav2vec2-large-xlsr-53).")
    parser.add_argument("--output_onnx_path", type=str, default=None, # Default to placing it inside hf_model_path
                        help="Full path to save the exported ONNX model (e.g., ./hf_model_path/model.onnx).")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy input.")
    parser.add_argument("--sequence_length", type=int, default=16000, 
                        help="Sequence length for dummy audio input (e.g., 16000 for 1 sec at 16kHz).")

    args = parser.parse_args()

    if args.output_onnx_path is None:
        args.output_onnx_path = os.path.join(args.hf_model_path, "model.onnx")
        print(f"Output ONNX path not specified, defaulting to: {args.output_onnx_path}")

    # Clean the specific ONNX file if it exists
    if os.path.exists(args.output_onnx_path):
        print(f"Removing existing ONNX file: {args.output_onnx_path}")
        os.remove(args.output_onnx_path)
    
    output_dir = os.path.dirname(args.output_onnx_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Wav2Vec2Model from: {args.hf_model_path}")
    # Load the config and ensure output_hidden_states is True for the PyTorch model
    pytorch_config = AutoConfig.from_pretrained(args.hf_model_path)
    pytorch_config.output_hidden_states = True 
    pytorch_model = Wav2Vec2Model.from_pretrained(args.hf_model_path, config=pytorch_config)
    pytorch_model.eval() 

    onnx_exportable_model = Wav2Vec2HiddenStatesExtractor(pytorch_model)

    print("Creating processed dummy input using Wav2Vec2FeatureExtractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.hf_model_path)
    raw_dummy_audio = [np.random.randn(args.sequence_length).astype(np.float32) for _ in range(args.batch_size)]
    processed_inputs = feature_extractor(raw_dummy_audio, return_tensors="pt", sampling_rate=feature_extractor.sampling_rate, padding=True)
    dummy_input_values_pt = processed_inputs.input_values
    print(f"Processed dummy input shape: {dummy_input_values_pt.shape}")

    with torch.no_grad():
        dummy_pytorch_outputs_tuple = onnx_exportable_model(dummy_input_values_pt)
        num_hidden_states = len(dummy_pytorch_outputs_tuple)
        print(f"PyTorch model will output {num_hidden_states} hidden states.")

    input_names = ["input_values"]
    output_names = [f"hidden_state_{i}" for i in range(num_hidden_states)]

    dynamic_axes = { input_names[0]: {0: 'batch_size', 1: 'sequence_length'} }
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size', 1: 'feat_sequence_length'}

    print(f"Exporting model to ONNX: {args.output_onnx_path} with opset {args.opset_version}")
    try:
        torch.onnx.export(
            onnx_exportable_model,
            dummy_input_values_pt,
            args.output_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print("ONNX export complete.")
    except Exception as e:
        print(f"Error during torch.onnx.export: {e}")
        import traceback
        traceback.print_exc()
        return # Stop if export fails

    print("\\n--- Starting ONNX Model Verification ---")
    try:
        print("Loading ONNX model for verification...")
        ort_session = onnxruntime.InferenceSession(args.output_onnx_path, providers=['CPUExecutionProvider'])
        onnx_model_input_name = ort_session.get_inputs()[0].name
        print(f"ONNX model loaded. Expected input name: {onnx_model_input_name}")

        ort_inputs_np = {onnx_model_input_name: dummy_input_values_pt.cpu().numpy()}
        print("Running ONNX Runtime inference...")
        ort_outputs_np_list = ort_session.run(None, ort_inputs_np)
        print(f"ONNX Runtime inference successful. Number of outputs: {len(ort_outputs_np_list)}")

        all_match = True
        if num_hidden_states != len(ort_outputs_np_list):
            print(f"Mismatch in number of outputs: PyTorch expected {num_hidden_states}, ONNX provided {len(ort_outputs_np_list)}")
            all_match = False
        else:
            print(f"Comparing {num_hidden_states} hidden states...")
            for i in range(num_hidden_states):
                try:
                    np.testing.assert_allclose(dummy_pytorch_outputs_tuple[i].cpu().numpy(), ort_outputs_np_list[i], rtol=1.5, atol=0.005)
                except AssertionError as error_msg:
                    print(f"Mismatch in hidden state {i}:\\n{error_msg}")
                    all_match = False
            
            if all_match:
                print("ONNX Runtime outputs match PyTorch outputs numerically (within tolerance).")
            else:
                print("ONNX Runtime outputs DO NOT match PyTorch outputs numerically.")

    except Exception as e:
        print(f"Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
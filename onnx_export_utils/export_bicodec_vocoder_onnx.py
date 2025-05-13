"""
Exports the BiCodec Vocoder (detokenizer and waveform generator) to ONNX.

This script loads a pre-trained BiCodec model and its associated BiCodecTokenizer.
It defines an ONNX-exportable wrapper (`BiCodecVocoderONNXWrapper`) that replicates
the vocoding process: taking semantic and global speaker tokens as input and
outputting a reconstructed audio waveform.

The shape of the dummy input tokens is derived by tokenizing a sample audio file
using the actual BiCodecTokenizer to ensure realistic dimensions.
"""

import argparse
from pathlib import Path
import os # For checking sample audio path, creating dummy if needed
from typing import Tuple # Import Tuple for type hinting

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

# SparkTTS specific imports
from sparktts.models.bicodec import BiCodec
from sparktts.models.audio_tokenizer import BiCodecTokenizer
# from sparktts.utils.file import load_config # Not directly used here, BiCodec.load handles config

class BiCodecVocoderONNXWrapper(nn.Module):
    """
    Wrapper for BiCodec's vocoding components (detokenizer, prenet, generator) for ONNX export.
    Takes semantic tokens and global speaker tokens, and outputs a waveform.
    """
    def __init__(self, bicodec_model: BiCodec):
        super().__init__()
        self.quantizer = bicodec_model.quantizer
        self.speaker_encoder = bicodec_model.speaker_encoder
        self.prenet = bicodec_model.prenet
        self.decoder = bicodec_model.decoder # This is the WaveGenerator (vocoder)

    def forward(self, semantic_tokens: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        """
        Performs vocoding from semantic and global speaker tokens to a waveform.

        Args:
            semantic_tokens (torch.Tensor): Semantic token IDs.
                Expected shape: (B, N_quant_semantic, T_semantic) or (B, T_semantic_flat).
            global_tokens (torch.Tensor): Global speaker token IDs.
                Expected shape: (B, N_quant_global, T_global).

        Returns:
            torch.Tensor: Reconstructed audio waveform.
                          Shape: (B, 1, T_audio) or (B, T_audio).
        """
        # 1. Detokenize semantic tokens to get quantized embeddings `z_q`
        # Expected `semantic_tokens` shape by quantizer.detokenize depends on its internal FSQ structure.
        # Typically (B, N_quantizers, T_quantized) or (B, T_quantized_flat).
        # Output `z_q` shape: (B, D_model, T_quantized_after_detok)
        z_q = self.quantizer.detokenize(semantic_tokens)

        # 2. Detokenize global speaker tokens to get speaker embedding `d_vector`
        # Expected `global_tokens` by speaker_encoder.detokenize: (B, N_quant_spk, T_spk_tokens)
        # Output `d_vector` shape: (B, D_speaker_embedding)
        d_vector = self.speaker_encoder.detokenize(global_tokens, onnx_export_mode=True)
        
        # 3. Pass `z_q` and `d_vector` through the prenet
        # `prenet` might expect `d_vector` to be unsqueezed for broadcasting.
        # Output `x_prenet` shape: (B, D_prenet_out, T_prenet_out) (usually T_prenet_out == T_quantized_after_detok)
        x_prenet = self.prenet(z_q, d_vector)
        
        # 4. Condition prenet output with speaker embedding before the decoder
        # This step matches the original BiCodec.detokenize logic.
        # `d_vector` [B, D_spk] needs to be broadcastable with `x_prenet` [B, D_prenet, T_prenet].
        # If D_spk == D_prenet, unsqueeze d_vector to [B, D_spk, 1].
        if d_vector.ndim == 2 and x_prenet.ndim == 3:
            if d_vector.shape[0] == x_prenet.shape[0] and d_vector.shape[1] == x_prenet.shape[1]:
                condition_vector = d_vector.unsqueeze(-1) # Makes d_vector [B, D_spk, 1]
            else:
                raise ValueError(
                    f"Shape mismatch for conditioning: d_vector {d_vector.shape}, x_prenet {x_prenet.shape}. "
                    f"Channel dimensions (dim 1) must match."
                )
        elif d_vector.ndim == x_prenet.ndim and d_vector.shape == x_prenet.shape:
            # This case implies prenet might have already handled broadcasting or d_vector is already correctly shaped.
            condition_vector = d_vector
        else:
            raise ValueError(
                f"Unexpected dimensions for conditioning: d_vector {d_vector.ndim}D {d_vector.shape}, "
                f"x_prenet {x_prenet.ndim}D {x_prenet.shape}. Cannot broadcast."
            )

        x_conditioned = x_prenet + condition_vector
        
        # 5. Generate waveform using the decoder (WaveGenerator)
        # Input `x_conditioned` shape: (B, D_prenet_out, T_prenet_out)
        # Output `wav_recon` shape: (B, 1, T_audio) or (B, T_audio)
        wav_recon = self.decoder(x_conditioned)
        return wav_recon

def get_dummy_tokens_from_sample_audio(
    model_base_dir: Path, 
    device: torch.device, 
    sample_audio_path_str: str,
    target_sample_rate: int = 16000, # Default SR for dummy audio if created
    duration_sec: int = 1 # Duration for dummy audio if created
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates dummy semantic and global tokens by tokenizing a sample audio file.
    This ensures the dummy tokens have realistic shapes for ONNX export.
    If the sample audio path doesn't exist, a silent dummy WAV is created.

    Args:
        model_base_dir: The base directory of the SparkTTS model 
                        (e.g., ./pretrained_models/Spark-TTS-0.5B), which is the parent of BiCodec, wav2vec2 etc.
        device: The torch device to use for tokenization.
        sample_audio_path_str: Path to the sample WAV file.
        target_sample_rate: Sample rate for the dummy audio if it needs to be created.
        duration_sec: Duration in seconds for the dummy audio if it needs to be created.

    Returns:
        A tuple (dummy_semantic_tokens, dummy_global_tokens).
    """
    print(f"[INFO] Initializing BiCodecTokenizer from base model dir: {model_base_dir} to get sample token shapes.")
    try:
        tokenizer = BiCodecTokenizer(model_dir=model_base_dir, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to initialize BiCodecTokenizer from {model_base_dir}: {e}")
        raise
    
    sample_audio_path = Path(sample_audio_path_str)
    if not sample_audio_path.exists():
        print(f"[WARNING] Sample audio {sample_audio_path} not found. Creating a temporary silent dummy WAV file.")
        try:
            import soundfile as sf
            dummy_wav_data = np.zeros(target_sample_rate * duration_sec, dtype=np.float32)
            sample_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(sample_audio_path, dummy_wav_data, target_sample_rate)
            print(f"[INFO] Dummy silent audio created at {sample_audio_path}")
        except ImportError:
            print("[ERROR] `soundfile` library is not installed. Cannot create dummy audio. Please install it (`pip install soundfile`) or provide a valid sample audio path.")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to create dummy audio file at {sample_audio_path}: {e}")
            raise

    print(f"[INFO] Tokenizing sample audio: {sample_audio_path} to get token shapes.")
    try:
        # BiCodecTokenizer.tokenize returns: global_tokens, semantic_tokens
        real_global_tokens, real_semantic_tokens = tokenizer.tokenize(str(sample_audio_path))
    except Exception as e:
        print(f"[ERROR] Failed to tokenize sample audio {sample_audio_path} with BiCodecTokenizer: {e}")
        raise

    print(f"[INFO]   Shape of real_global_tokens from sample: {real_global_tokens.shape}, dtype: {real_global_tokens.dtype}")
    print(f"[INFO]   Shape of real_semantic_tokens from sample: {real_semantic_tokens.shape}, dtype: {real_semantic_tokens.dtype}")

    # Create random integer tensors with the same shapes and type as the real tokens.
    # Using torch.randint_like ensures dtype and device match the real tokens.
    # Token IDs are non-negative; using a small upper bound for dummy values (e.g., 100, actual vocab size can be larger).
    dummy_global_tokens = torch.randint_like(real_global_tokens, low=0, high=100)
    dummy_semantic_tokens = torch.randint_like(real_semantic_tokens, low=0, high=100)
    
    print(f"[INFO]   Shape of created dummy_global_tokens: {dummy_global_tokens.shape}, dtype: {dummy_global_tokens.dtype}")
    print(f"[INFO]   Shape of created dummy_semantic_tokens: {dummy_semantic_tokens.shape}, dtype: {dummy_semantic_tokens.dtype}")

    return dummy_semantic_tokens, dummy_global_tokens

def export_bicodec_vocoder_to_onnx(
    model_dir: Path,
    output_path: Path,
    sample_audio_for_shapes: str, # Keep as str for argparse, will be Path in function
    opset_version: int,
    device_str: str,
):
    """
    Exports the SparkTTS BiCodec Vocoder to an ONNX model.

    Args:
        model_dir: Path to the base SparkTTS model directory (containing BiCodec subdir).
        output_path: Path to save the exported Vocoder ONNX model.
        sample_audio_for_shapes: Path to a sample WAV file used to derive realistic token shapes for dummy inputs.
        opset_version: ONNX opset version.
        device_str: Device to run export on (e.g., 'cpu', 'cuda:0').
    """
    print(f"[INFO] Starting BiCodec Vocoder ONNX export process...")
    print(f"[INFO]   Base model directory: {model_dir}")
    print(f"[INFO]   Output ONNX path: {output_path}")
    print(f"[INFO]   Sample audio for token shapes: {sample_audio_for_shapes}")
    print(f"[INFO]   Opset version: {opset_version}")
    print(f"[INFO]   Device: {device_str}")

    export_device = torch.device(device_str)

    # 1. Load the pre-trained BiCodec model
    bicodec_model_subdir = model_dir / "BiCodec"
    print(f"[INFO] Loading BiCodec model from: {bicodec_model_subdir}")
    if not bicodec_model_subdir.exists():
        print(f"[ERROR] BiCodec model directory not found at {bicodec_model_subdir}. Please check the path.")
        return
    
    try:
        bicodec_model = BiCodec.load_from_checkpoint(model_dir=bicodec_model_subdir, device=export_device)
        bicodec_model.eval()
        print("[INFO] BiCodec model loaded and set to eval mode.")
    except FileNotFoundError:
        print(f"[ERROR] BiCodec model files (config/checkpoint) not found in {bicodec_model_subdir}.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load BiCodec model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Instantiate the ONNX wrapper
    try:
        onnx_exportable_vocoder = BiCodecVocoderONNXWrapper(bicodec_model).to(export_device)
        onnx_exportable_vocoder.eval()
    except Exception as e:
        print(f"[ERROR] Failed to instantiate BiCodecVocoderONNXWrapper: {e}")
        return

    # 3. Prepare dummy inputs (semantic_tokens, global_tokens)
    # Use the actual model_dir (e.g. pretrained_models/Spark-TTS-0.5B) for BiCodecTokenizer path logic.
    print("[INFO] Preparing dummy input tokens using a sample audio file...")
    try:
        dummy_semantic_tokens, dummy_global_tokens = get_dummy_tokens_from_sample_audio(
            model_base_dir=model_dir, 
            device=export_device, 
            sample_audio_path_str=sample_audio_for_shapes
        )
    except Exception as e:
        print(f"[ERROR] Failed to get dummy tokens: {e}. Aborting export.")
        return
    # Ensure tokens are on the export device (though get_dummy_tokens should handle it if tokenizer is on export_device)
    dummy_semantic_tokens = dummy_semantic_tokens.to(export_device)
    dummy_global_tokens = dummy_global_tokens.to(export_device)

    print(f"[INFO] Using dummy_semantic_tokens shape: {dummy_semantic_tokens.shape}, dtype: {dummy_semantic_tokens.dtype}")
    print(f"[INFO] Using dummy_global_tokens shape: {dummy_global_tokens.shape}, dtype: {dummy_global_tokens.dtype}")

    # Test forward pass with the PyTorch ONNX wrapper
    print("[INFO] Performing a test forward pass with the PyTorch ONNX wrapper...")
    try:
        with torch.no_grad():
            # The shapes from get_dummy_tokens are already batched (e.g., [1, ...])
            pytorch_output_waveform = onnx_exportable_vocoder(dummy_semantic_tokens, dummy_global_tokens)
        print(f"[INFO] PyTorch ONNX wrapper test forward pass successful. Output waveform shape: {pytorch_output_waveform.shape}")
        if pytorch_output_waveform.numel() == 0:
            print("[ERROR] PyTorch wrapper produced empty waveform. Check dummy inputs or model logic.")
            return
    except Exception as e:
        print(f"[ERROR] PyTorch ONNX wrapper test forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Export to ONNX
    input_names = ["semantic_tokens", "global_tokens"]
    output_names = ["output_waveform"]
    
    dynamic_axes = {}
    # Semantic tokens: e.g., (B, T_semantic_flat) or (B, N_quant_semantic, T_semantic)
    if dummy_semantic_tokens.ndim == 2: 
        dynamic_axes[input_names[0]] = {0: 'batch_size', 1: 'semantic_token_flat_seq_len'}
    elif dummy_semantic_tokens.ndim == 3:
        dynamic_axes[input_names[0]] = {0: 'batch_size', 2: 'semantic_token_seq_len'} 
    else:
        print(f"[WARNING] Unexpected ndim ({dummy_semantic_tokens.ndim}) for dummy_semantic_tokens. Dynamic axes may be incomplete.")

    # Global tokens: e.g., (B, N_quant_global, T_global)
    if dummy_global_tokens.ndim == 3:
        dynamic_axes[input_names[1]] = {0: 'batch_size', 2: 'global_token_seq_len'}
    elif dummy_global_tokens.ndim == 2: # Less common for multi-level FSQ but possible if flattened
        dynamic_axes[input_names[1]] = {0: 'batch_size', 1: 'global_feature_len'}
    else:
        print(f"[WARNING] Unexpected ndim ({dummy_global_tokens.ndim}) for dummy_global_tokens. Dynamic axes may be incomplete.")

    # Output waveform: (B, 1, AudioSeqLen) or (B, AudioSeqLen)
    if pytorch_output_waveform.ndim == 3:
        dynamic_axes[output_names[0]] = {0: 'batch_size', 2: 'audio_sequence_length'}
    elif pytorch_output_waveform.ndim == 2:
        dynamic_axes[output_names[0]] = {0: 'batch_size', 1: 'audio_sequence_length'}
    print(f"[INFO] Using dynamic axes: {dynamic_axes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting BiCodec Vocoder to ONNX: {output_path}")
    try:
        torch.onnx.export(
            onnx_exportable_vocoder,
            (dummy_semantic_tokens, dummy_global_tokens), # Pass inputs as a tuple
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
        ort_session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_input_names_map = {inp.name: i for i, inp in enumerate(ort_session.get_inputs())}
        print(f"[INFO] ONNX model loaded. Input names: {list(onnx_input_names_map.keys())}")

        ort_inputs = {
            input_names[0]: dummy_semantic_tokens.cpu().numpy(), # Ensure correct mapping by name
            input_names[1]: dummy_global_tokens.cpu().numpy()
        }
        # Verify all expected input names are present in ort_inputs
        if not all(name in ort_inputs for name in onnx_input_names_map.keys()):
            print(f"[ERROR] Mismatch between expected ONNX input names {list(onnx_input_names_map.keys())} and provided ort_inputs {list(ort_inputs.keys())}")
            return

        print("[INFO] Running ONNX Runtime inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output_waveform_np = ort_outputs[0]
        print(f"[INFO] ONNX Runtime inference successful. Output waveform shape: {onnx_output_waveform_np.shape}")

        np.testing.assert_allclose(
            pytorch_output_waveform.cpu().detach().numpy(), 
            onnx_output_waveform_np, 
            rtol=1e-03, 
            atol=1e-05
        )
        print("[INFO] ONNX Runtime outputs match PyTorch outputs numerically (within tolerance). Verification successful.")

    except Exception as e:
        print(f"[ERROR] Error during ONNX verification: {e}")
        import traceback
        traceback.print_exc()
    print("[INFO] BiCodec Vocoder ONNX export and verification process complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Export SparkTTS BiCodec Vocoder (detokenizer + generator) to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir", 
        type=Path, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B), which should contain BiCodec, wav2vec2, etc., subdirectories for the tokenizer."
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        required=True, 
        help="Full path to save the exported Vocoder ONNX model (e.g., ./onnx_models/bicodec_vocoder.onnx)."
    )
    parser.add_argument(
        "--sample_audio_for_shapes", 
        type=str, # Keep as str for argparse, will be Path in function
        default="example/prompt_audio.wav",
        help="Path to a sample WAV file used to derive realistic token shapes for dummy inputs. If it doesn't exist, a silent dummy will be created (requires `soundfile` lib)."
    )
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (e.g., 'cpu', 'cuda:0').")

    args = parser.parse_args()

    export_bicodec_vocoder_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output_path,
        sample_audio_for_shapes=args.sample_audio_for_shapes,
        opset_version=args.opset_version,
        device_str=args.device,
    )

if __name__ == "__main__":
    main() 
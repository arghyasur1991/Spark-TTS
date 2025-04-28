#!/usr/bin/env python3
"""
Script to convert Spark-TTS PyTorch models to ONNX format.

This script handles the conversion of the BiCodec model and its submodels to ONNX format
for faster inference and cross-platform compatibility.
"""

import os
import argparse
import torch
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import sys

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Spark-TTS PyTorch models to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Directory containing PyTorch model files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B/onnx",
        help="Directory to save ONNX models"
    )
    parser.add_argument(
        "--dynamic_axes",
        action="store_true",
        help="Use dynamic axes for sequence length in ONNX export"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage even if CUDA is available"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version to use for export"
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip the verification of ONNX models (useful if onnxruntime is not installed)"
    )
    return parser.parse_args()

def get_device(cpu_only: bool = False):
    """Determine the best available device for conversion."""
    if cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU device as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def check_environment():
    """Check the environment for required dependencies."""
    try:
        import onnx
        logger.info(f"ONNX version: {onnx.__version__}")
    except ImportError:
        logger.error("ONNX is not installed. Please install it using: pip install onnx")
        sys.exit(1)
    
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        logger.warning("ONNX Runtime is not installed. Model verification will be skipped.")
        logger.warning("To install it, use one of the following:")
        logger.warning("  - For CPU: pip install onnxruntime")
        logger.warning("  - For NVIDIA GPUs: pip install onnxruntime-gpu")
        logger.warning("  - For Apple Silicon: pip install onnxruntime-silicon")
        return False
    
    return True

def convert_bicodec_to_onnx(
    model: BiCodec,
    output_dir: str,
    dynamic_axes: bool = True,
    opset_version: int = 17,
    skip_verification: bool = False
):
    """
    Convert the BiCodec model and its submodels to ONNX format.
    
    Args:
        model: The BiCodec model to convert
        output_dir: Directory to save the ONNX models
        dynamic_axes: Whether to use dynamic axes for sequence dimensions
        opset_version: ONNX opset version to use
        skip_verification: Whether to skip verification with ONNX Runtime
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    # Export encoder
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    logger.info(f"Converting encoder to ONNX: {encoder_path}")
    
    # Prepare dummy input for encoder - the encoder expects (batch, channels, length)
    dummy_feat = torch.randn(1, 1024, 100, device=device)  # (batch, channels, seq_len)
    
    # Define dynamic axes if needed
    encoder_dynamic_axes = None
    if dynamic_axes:
        encoder_dynamic_axes = {
            "input": {2: "seq_len"},  # sequence length can vary
            "output": {2: "encoded_seq_len"}  # encoded sequence length can vary
        }
    
    # Export encoder model - note that we don't need to transpose here since the encoder expects (batch, channels, length)
    torch.onnx.export(
        model.encoder,
        dummy_feat,  # No transpose needed - encoder expects (batch, channels, length)
        encoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=encoder_dynamic_axes
    )
    
    # Export quantizer
    quantizer_path = os.path.join(output_dir, "quantizer.onnx")
    logger.info(f"Converting quantizer to ONNX: {quantizer_path}")
    
    # The output of encoder has shape (batch, seq_len, hidden_dim)
    # Using 1024 as the hidden dimension based on the model config
    dummy_z = torch.randn(1, 100, 1024, device=device)  # (batch, seq_len, hidden_dim)
    
    # Define dynamic axes if needed
    quantizer_dynamic_axes = None
    if dynamic_axes:
        quantizer_dynamic_axes = {
            "input": {1: "seq_len"},  # sequence length can vary
            "output": {1: "seq_len"}  # output sequence length can vary
        }
    
    # Create a class to represent the tokenize method of quantizer
    class QuantizerTokenize(torch.nn.Module):
        def __init__(self, quantizer):
            super().__init__()
            self.quantizer = quantizer
            
        def forward(self, z):
            # The issue is that the quantizer expects the tensor in the format (batch, channels, seq_len)
            # but we're providing it in the format (batch, seq_len, channels)
            # so we need to transpose it before passing to tokenize
            z = z.transpose(1, 2)  # Convert from (batch, seq_len, channels) to (batch, channels, seq_len)
            return self.quantizer.tokenize(z)
    
    quantizer_tokenize = QuantizerTokenize(model.quantizer)
    
    # Export quantizer tokenize
    torch.onnx.export(
        quantizer_tokenize,
        dummy_z,
        os.path.join(output_dir, "quantizer_tokenize.onnx"),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["tokens"],
        dynamic_axes=quantizer_dynamic_axes if dynamic_axes else None
    )
    
    # Create a class to represent the detokenize method of quantizer
    class QuantizerDetokenize(torch.nn.Module):
        def __init__(self, quantizer):
            super().__init__()
            self.quantizer = quantizer
            
        def forward(self, tokens):
            # detokenize returns tensor in format (batch, channels, seq_len)
            # but our model expects (batch, seq_len, channels)
            z_q = self.quantizer.detokenize(tokens)
            # Convert back to (batch, seq_len, channels) for consistency with other parts
            return z_q.transpose(1, 2)
    
    # Create dummy token input - use the codebook size from config
    dummy_tokens = torch.randint(0, 8192, (1, 100), device=device)  # (batch, seq_len)
    
    quantizer_detokenize = QuantizerDetokenize(model.quantizer)
    
    # Export quantizer detokenize
    torch.onnx.export(
        quantizer_detokenize,
        dummy_tokens,
        os.path.join(output_dir, "quantizer_detokenize.onnx"),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["tokens"],
        output_names=["z_q"],
        dynamic_axes={"tokens": {1: "seq_len"}, "z_q": {1: "seq_len"}} if dynamic_axes else None
    )
    
    # Export speaker encoder
    speaker_encoder_path = os.path.join(output_dir, "speaker_encoder.onnx")
    logger.info(f"Converting speaker encoder to ONNX: {speaker_encoder_path}")
    
    # The SpeakerEncoder expects mels in format (batch, seq_len, mel_dim)
    dummy_mel = torch.randn(1, 100, 128, device=device)  # (batch, seq_len, mel_dim)
    
    # Define dynamic axes if needed
    speaker_encoder_dynamic_axes = None
    if dynamic_axes:
        speaker_encoder_dynamic_axes = {
            "input": {1: "seq_len"}  # mel sequence length can vary
        }
    
    # Export speaker encoder model - input is already in the correct format
    torch.onnx.export(
        model.speaker_encoder,
        dummy_mel,
        speaker_encoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["x_vector", "d_vector"],
        dynamic_axes=speaker_encoder_dynamic_axes
    )
    
    # Export decoder components (prenet, postnet, and wave generator)
    # Prenet
    prenet_path = os.path.join(output_dir, "prenet.onnx")
    logger.info(f"Converting prenet to ONNX: {prenet_path}")
    
    # The decoder expects z_q with shape (batch, channels, seq_len)
    dummy_z_q = torch.randn(1, 1024, 100, device=device)  # (batch, channels, seq_len)
    # The d_vector from speaker encoder has shape (batch, latent_dim)
    dummy_d_vector = torch.randn(1, 1024, device=device)  # (batch, latent_dim)
    
    # Define a wrapper class for prenet that takes both inputs
    class PrenetWrapper(torch.nn.Module):
        def __init__(self, prenet):
            super().__init__()
            self.prenet = prenet
            
        def forward(self, z_q, d_vector):
            # prenet expects z_q to be in format (batch, channels, seq_len)
            # and d_vector to be (batch, latent_dim)
            return self.prenet(z_q, d_vector)
    
    prenet_wrapper = PrenetWrapper(model.prenet)
    
    # Define dynamic axes if needed
    prenet_dynamic_axes = None
    if dynamic_axes:
        prenet_dynamic_axes = {
            "z_q": {1: "seq_len"},  # sequence length can vary
            "output": {2: "seq_len"}  # output sequence length can vary
        }
    
    # Export prenet model
    torch.onnx.export(
        prenet_wrapper,
        (dummy_z_q, dummy_d_vector),
        prenet_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["z_q", "d_vector"],
        output_names=["output"],
        dynamic_axes=prenet_dynamic_axes
    )
    
    # Postnet
    postnet_path = os.path.join(output_dir, "postnet.onnx")
    logger.info(f"Converting postnet to ONNX: {postnet_path}")
    
    # The output of prenet has shape (batch, hidden_dim, seq_len)
    dummy_prenet_output = torch.randn(1, 1024, 100, device=device)  # (batch, hidden_dim, seq_len)
    
    # Define dynamic axes if needed
    postnet_dynamic_axes = None
    if dynamic_axes:
        postnet_dynamic_axes = {
            "input": {2: "seq_len"},  # sequence length can vary
            "output": {2: "seq_len"}  # output sequence length can vary
        }
    
    # Export postnet model
    torch.onnx.export(
        model.postnet,
        dummy_prenet_output,
        postnet_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=postnet_dynamic_axes
    )
    
    # Wave generator (decoder)
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    logger.info(f"Converting wave generator (decoder) to ONNX: {decoder_path}")
    
    # The input to decoder has shape (batch, hidden_dim, seq_len)
    dummy_x = torch.randn(1, 1024, 100, device=device)  # (batch, hidden_dim, seq_len)
    
    # Define dynamic axes if needed
    decoder_dynamic_axes = None
    if dynamic_axes:
        decoder_dynamic_axes = {
            "input": {2: "seq_len"},  # sequence length can vary
            "output": {2: "seq_len"}  # output sequence length can vary
        }
    
    # Export decoder model
    torch.onnx.export(
        model.decoder,
        dummy_x,
        decoder_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=decoder_dynamic_axes
    )
    
    # Create a config file to track the exported models
    with open(os.path.join(output_dir, "onnx_models.txt"), "w") as f:
        f.write("ONNX Models:\n")
        f.write(f"- encoder.onnx: BiCodec Encoder\n")
        f.write(f"- quantizer_tokenize.onnx: BiCodec Quantizer (tokenize)\n")
        f.write(f"- quantizer_detokenize.onnx: BiCodec Quantizer (detokenize)\n")
        f.write(f"- speaker_encoder.onnx: Speaker Encoder\n")
        f.write(f"- prenet.onnx: BiCodec Prenet\n")
        f.write(f"- postnet.onnx: BiCodec Postnet\n")
        f.write(f"- decoder.onnx: BiCodec Wave Generator\n")
    
    logger.info(f"All models exported to ONNX format in: {output_dir}")
    
    # Verify the exported ONNX models if onnxruntime is available and verification is not skipped
    if skip_verification:
        logger.info("Skipping model verification as requested")
        return
    
    try:
        import onnxruntime as ort
        logger.info("Verifying ONNX models with ONNX Runtime...")
        
        for model_file in ["encoder.onnx", "quantizer_tokenize.onnx", "quantizer_detokenize.onnx", 
                           "speaker_encoder.onnx", "prenet.onnx", "postnet.onnx", "decoder.onnx"]:
            model_path = os.path.join(output_dir, model_file)
            logger.info(f"Verifying {model_file}...")
            
            # Create an ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(model_path, sess_options)
            
            logger.info(f"✓ Verified {model_file}")
            
        logger.info("All ONNX models verified successfully!")
    except ImportError:
        logger.warning("ONNX Runtime not installed. Skipping model verification.")
        logger.warning("To enable verification, install ONNX Runtime:")
        logger.warning("  pip install onnxruntime")

def convert_all_models_to_onnx(args):
    """
    Load and convert all models in the specified directory to ONNX format.
    
    Args:
        args: Command-line arguments
    """
    # Check environment for dependencies
    has_onnxruntime = check_environment()
    if not has_onnxruntime and not args.skip_verification:
        logger.warning("ONNX Runtime not found but verification not explicitly skipped.")
        logger.warning("Continuing with --skip_verification flag automatically set.")
        args.skip_verification = True
    
    device = get_device(args.cpu_only)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    # Check if model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)
    
    # Look for BiCodec directory which contains the actual model files
    bicodec_dir = model_dir / "BiCodec"
    if bicodec_dir.exists() and (bicodec_dir / "config.yaml").exists() and (bicodec_dir / "model.safetensors").exists():
        logger.info(f"Found BiCodec model in: {bicodec_dir}")
        model_dir = bicodec_dir
    else:
        logger.warning(f"BiCodec directory not found in {model_dir}, trying to use the directory as is.")
        # Check if we at least have the required files
        if not (model_dir / "config.yaml").exists():
            logger.error(f"config.yaml not found in {model_dir}")
            sys.exit(1)
        if not (model_dir / "model.safetensors").exists():
            logger.error(f"model.safetensors not found in {model_dir}")
            sys.exit(1)
    
    logger.info(f"Loading model from: {model_dir}")
    
    try:
        # Load the BiCodec model
        model = BiCodec.load_from_checkpoint(model_dir, device=device)
        model.eval()
        
        # Move model to device
        model.to(device)
        
        logger.info("Model loaded successfully")
        
        # Convert BiCodec models to ONNX
        convert_bicodec_to_onnx(
            model=model,
            output_dir=str(output_dir),
            dynamic_axes=args.dynamic_axes,
            opset_version=args.opset_version,
            skip_verification=args.skip_verification
        )
        
        logger.info(f"All models converted to ONNX format in: {output_dir}")
    except Exception as e:
        logger.error(f"Error converting models to ONNX: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    convert_all_models_to_onnx(args) 
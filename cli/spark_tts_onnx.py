#!/usr/bin/env python3
"""
Command-line interface for Spark-TTS ONNX inference.
This is a simplified version that demonstrates how to use ONNX models
with Spark-TTS.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Import the simplified SparkTTSONNX class
from cli.SparkTTSONNX import SparkTTSONNX

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Spark-TTS ONNX command-line interface")
    
    parser.add_argument(
        "--model-dir", 
        type=str, 
        required=True,
        help="Directory containing the model files"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Text to synthesize"
    )
    parser.add_argument(
        "--text-file", 
        type=str, 
        help="File containing text to synthesize"
    )
    parser.add_argument(
        "--prompt-path", 
        type=str, 
        required=True,
        help="Path to prompt audio file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to save output audio"
    )
    parser.add_argument(
        "--gender", 
        type=float, 
        default=0.5,
        help="Gender control (0: male, 1: female)"
    )
    parser.add_argument(
        "--pitch", 
        type=float, 
        default=1.0,
        help="Pitch factor"
    )
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0,
        help="Speed factor"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (cuda, mps, cpu)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Check if model directory exists and contains ONNX models
    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    
    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist.")
        sys.exit(1)
    
    if not onnx_dir.exists():
        print(f"Error: ONNX model directory {onnx_dir} does not exist.")
        print("Please run convert_to_onnx.py to generate ONNX models first.")
        sys.exit(1)
    
    # Check if prompt file exists
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        print(f"Error: Prompt audio file {prompt_path} does not exist.")
        sys.exit(1)
    
    # Initialize TTS model
    print(f"Initializing Spark-TTS ONNX model from {args.model_dir}")
    try:
        tts = SparkTTSONNX(
            model_dir=args.model_dir,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)
    
    # Process prompt
    print(f"Processing prompt audio: {args.prompt_path}")
    tts.process_prompt(args.prompt_path)
    
    # Apply voice controls
    tts.process_prompt_control(
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed
    )
    
    # Get text from argument or file
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        print("Error: Either --text or --text-file must be provided.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Measure inference time
    start_time = time.time()
    
    # Generate speech
    wav, info = tts.inference(
        text=text,
        save_dir=args.output_dir,
    )
    
    inference_time = time.time() - start_time
    
    # Print stats
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Audio length: {len(wav) / 16000:.2f} seconds")
    print(f"Real-time factor: {(len(wav) / 16000) / inference_time:.2f}x")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
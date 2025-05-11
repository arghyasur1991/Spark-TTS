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

# Set MPS fallback for operations not supported by MPS
# This addresses the "Output channels > 65536 not supported at the MPS device" error
# that occurs with wav2vec2 models on Apple Silicon (M1/M2/M3) hardware
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform
import json
import time

from cli.SparkTTS import SparkTTS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--text", type=str, help="Text for TTS generation"
    )
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run benchmark comparing standard and optimized flows"
    )
    parser.add_argument(
        "--benchmark_texts", 
        type=str, 
        nargs="+", 
        default=["This is the first sentence to synthesize.",
                "Here's another sentence with the same voice.",
                "And one more with the same voice characteristics."],
        help="List of texts to use for benchmarking"
    )
    parser.add_argument(
        "--benchmark_runs", 
        type=int, 
        default=3, 
        help="Number of benchmark runs to average"
    )
    parser.add_argument(
        "--reuse_tokenization", 
        action="store_true", 
        help="Use optimized flow with tokenization reuse"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "fp16", "int8"],
        default="none",
        help="Model quantization method (none, fp16, or int8)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Whether to use sampling for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile for model acceleration (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="Compilation mode for torch.compile"
    )
    parser.add_argument(
        "--use_wav2vec2_onnx",
        action="store_true",
        help="Use ONNX for Wav2Vec2 feature extraction in BiCodecTokenizer."
    )
    parser.add_argument(
        "--use_bicodec_onnx",
        action="store_true",
        help="Use ONNX for BiCodec vocoder inference."
    )
    parser.add_argument(
        "--use_speaker_encoder_tokenizer_onnx",
        action="store_true",
        help="Use ONNX for Speaker Encoder tokenizer."
    )
    parser.add_argument(
        "--use_llm_onnx",
        action="store_true",
        help="Use ONNX for the main Language Model.",
    )
    parser.add_argument(
        "--use_mel_spectrogram_onnx",
        action="store_true",
        help="Use ONNX for Mel Spectrogram generation."
    )
    return parser.parse_args()


def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # macOS with MPS support (Apple Silicon)
        logging.info("MPS fallback to CPU enabled for unsupported operations")
        device = torch.device(f"mps:{args.device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    model = SparkTTS(
        args.model_dir, 
        device,
        quantization=args.quantization if args.quantization != "none" else None,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        use_wav2vec2_onnx=args.use_wav2vec2_onnx,
        use_bicodec_onnx=args.use_bicodec_onnx,
        use_speaker_encoder_tokenizer_onnx=args.use_speaker_encoder_tokenizer_onnx,
        use_llm_onnx=args.use_llm_onnx,
        use_mel_spectrogram_onnx=args.use_mel_spectrogram_onnx,
    )

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(model, args)
        return

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Check if we should use the optimized flow
    if args.reuse_tokenization and args.text and len(args.text.split(".")) > 1:
        # For demonstration, split the text into sentences
        sentences = [s.strip() + "." for s in args.text.split(".") if s.strip()]
        
        logging.info(f"Using optimized flow with {len(sentences)} sentences")
        
        # Initial tokenization
        start_time = time.time()
        model_inputs, global_token_ids = model.tokenize_inputs(
            text=sentences[0],
            prompt_speech_path=args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
        )
        
        # Process first sentence
        wav, model_inputs, global_token_ids = model.inference(
            model_inputs=model_inputs,
            global_token_ids=global_token_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )
        
        # Save first sentence
        sf.write(save_path, wav, samplerate=16000)
        logging.info(f"First sentence audio saved at: {save_path}")
        
        # Process remaining sentences
        for i, sentence in enumerate(sentences[1:], 1):
            sentence_start = time.time()
            
            # Update text in existing tokenized inputs
            is_control_mode = args.gender is not None
            if is_control_mode:
                updated_inputs = model.update_text_in_tokenized_inputs(
                    model_inputs, sentence, True, args.gender, args.pitch, args.speed
                )
            else:
                updated_inputs = model.update_text_in_tokenized_inputs(
                    model_inputs, sentence
                )
            
            # Generate with updated inputs
            wav, model_inputs, _ = model.inference(
                model_inputs=updated_inputs,
                global_token_ids=global_token_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            
            # Save subsequent sentences
            next_save_path = os.path.join(args.save_dir, f"{timestamp}_sentence_{i}.wav")
            sf.write(next_save_path, wav, samplerate=16000)
            logging.info(f"Sentence {i+1} audio saved at: {next_save_path} (took {time.time() - sentence_start:.2f}s)")
        
        logging.info(f"Total time for optimized flow: {time.time() - start_time:.2f}s")
    else:
        # Perform standard inference and save the output audio
        with torch.no_grad():
            wav = model.inference(
                args.text,
                args.prompt_speech_path,
                prompt_text=args.prompt_text,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )[0]  # Extract wav from tuple
            sf.write(save_path, wav, samplerate=16000)

        logging.info(f"Audio saved at: {save_path}")


def run_benchmark(model, args):
    """Run benchmark comparing standard and optimized flows."""
    logging.info("Running benchmark...")
    logging.info(f"Number of texts: {len(args.benchmark_texts)}")
    logging.info(f"Number of runs: {args.benchmark_runs}")
    
    # Run benchmark
    results = model.benchmark_performance(
        text_list=args.benchmark_texts,
        prompt_speech_path=args.prompt_speech_path,
        prompt_text=args.prompt_text,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
        num_runs=args.benchmark_runs,
    )
    
    # Save benchmark results
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_file = os.path.join(args.save_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display results
    logging.info("\n=== BENCHMARK RESULTS ===")
    logging.info(f"Standard flow total time: {results['standard_flow']['total_time']:.2f}s")
    logging.info(f"Optimized flow total time: {results['optimized_flow']['total_time']:.2f}s")
    logging.info(f"Speedup factor: {results['speedup_factor']:.2f}x")
    logging.info(f"Initial tokenization overhead: {results['first_utterance_overhead']:.2f}s")
    logging.info(f"Subsequent utterance speedup: {results['subsequent_utterance_speedup']:.2f}x")
    
    # Display operation timing analysis
    logging.info("\n=== OPERATION TIMING ANALYSIS ===")
    
    # Standard flow analysis
    std_analysis = results['operation_analysis']['standard_flow']
    if std_analysis:
        logging.info("Standard Flow Breakdown:")
        for op_name in sorted(
            [k for k in std_analysis.keys() if not k.endswith('_percent') 
             and k not in ['total_time', 'heaviest_operation', 'heaviest_operation_time', 'heaviest_operation_percent']],
            key=lambda x: std_analysis[x],
            reverse=True
        ):
            if op_name.endswith('_time'):
                op_display = op_name.replace('_time', '')
                logging.info(f"  {op_display:20s}: {std_analysis[op_name]:.2f}s ({std_analysis[op_name+'_percent']:.1f}%)")
        
        logging.info(f"  Heaviest operation: {std_analysis['heaviest_operation'].replace('_time', '')} "
                    f"({std_analysis['heaviest_operation_percent']:.1f}% of total time)")
    
    # Optimized flow analysis
    opt_analysis = results['operation_analysis']['optimized_flow']
    if opt_analysis:
        logging.info("\nOptimized Flow Breakdown:")
        for op_name in sorted(
            [k for k in opt_analysis.keys() if not k.endswith('_percent') 
             and k not in ['total_time', 'heaviest_operation', 'heaviest_operation_time', 'heaviest_operation_percent']],
            key=lambda x: opt_analysis[x],
            reverse=True
        ):
            if op_name.endswith('_time'):
                op_display = op_name.replace('_time', '')
                logging.info(f"  {op_display:20s}: {opt_analysis[op_name]:.2f}s ({opt_analysis[op_name+'_percent']:.1f}%)")
        
        logging.info(f"  Heaviest operation: {opt_analysis['heaviest_operation'].replace('_time', '')} "
                    f"({opt_analysis['heaviest_operation_percent']:.1f}% of total time)")
    
    logging.info(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    run_tts(args)

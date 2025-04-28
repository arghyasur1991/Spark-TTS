#!/usr/bin/env python3
"""
Comparison test for PyTorch vs ONNX implementations of Spark-TTS.
This script runs inference with both implementations and compares the results.
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
import argparse
import random
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX TTS outputs")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="example/prompt_audio.wav",
        help="Path to the prompt audio file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/comparison",
        help="Directory to save audio outputs",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a comparison test between PyTorch and ONNX implementations of Spark TTS.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save visualization plots",
    )
    
    return parser.parse_args()

def run_pytorch_inference(model_dir, prompt_path, text):
    """Run inference with the PyTorch implementation"""
    try:
        from cli.SparkTTS import SparkTTS
        
        print("\n--- Running PyTorch inference ---")
        start_time = time.time()
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Using CUDA device")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        
        # Initialize model
        print(f"Initializing PyTorch model from {model_dir}")
        model = SparkTTS(model_dir, device)
        
        # Voice parameters - strings for PyTorch model
        # These map to the equivalent ONNX float parameters
        gender = "male"      # Maps to 0.0 in ONNX
        pitch = "moderate"   # Maps to 0.5 in ONNX
        speed = "moderate"   # Maps to 0.5 in ONNX
        
        print(f"Voice parameters - Gender: {gender}, Pitch: {pitch}, Speed: {speed}")
        
        # Perform inference
        with torch.no_grad():
            print(f"Generating speech for text: {text}")
            wav = model.inference(
                text=text,
                prompt_speech_path=prompt_path,
                gender=gender,
                pitch=pitch,
                speed=speed,
            )
        
        # Convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        
        inference_time = time.time() - start_time
        print(f"PyTorch inference completed in {inference_time:.2f} seconds")
        print(f"Generated {len(wav)/16000:.2f} seconds of audio")
        
        return wav, inference_time
    
    except ImportError as e:
        print(f"Error importing PyTorch dependencies: {e}")
        print("Skipping PyTorch inference")
        return None, 0

def run_onnx_inference(model_dir, prompt_path, text):
    """Run inference with the ONNX implementation"""
    try:
        from cli.SparkTTSONNX import SparkTTSONNX
        
        print("\n--- Running ONNX inference ---")
        start_time = time.time()
        
        # Initialize model
        print(f"Initializing ONNX model from {model_dir}")
        model = SparkTTSONNX(model_dir=model_dir)
        
        # Process prompt
        print(f"Processing prompt audio: {prompt_path}")
        model.process_prompt(prompt_path)
        
        # Voice parameters - floats for ONNX model
        # These map to the equivalent PyTorch string parameters
        gender = 0.0  # Maps to "male" in PyTorch
        pitch = 0.5   # Maps to "moderate" in PyTorch
        speed = 0.5   # Maps to "moderate" in PyTorch
        
        print(f"Voice parameters - Gender: {gender:.1f}, Pitch: {pitch:.1f}, Speed: {speed:.1f}")
        
        # Set voice parameters
        model.process_prompt_control(gender=gender, pitch=pitch, speed=speed)
        
        # Perform inference
        print(f"Generating speech for text: {text}")
        wav, info = model.inference(text=text)
        
        inference_time = time.time() - start_time
        print(f"ONNX inference completed in {inference_time:.2f} seconds")
        print(f"Generated {len(wav)/16000:.2f} seconds of audio")
        
        return wav, inference_time
    
    except ImportError as e:
        print(f"Error importing ONNX dependencies: {e}")
        print("Skipping ONNX inference")
        return None, 0

def compute_spectrogram(wav, fs=16000, nperseg=512):
    """Compute spectrogram for visualization and analysis"""
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        wav,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg//2,
        nfft=1024,
        detrend=False,
        scaling='spectrum'
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-9)
    
    return f, t, Sxx_db

def plot_spectrograms(wav1, wav2, title1="PyTorch", title2="ONNX", output_path=None):
    """Plot spectrograms of two audio signals for comparison"""
    # Ensure same length for comparison
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]
    
    # Compute spectrograms
    f1, t1, Sxx1 = compute_spectrogram(wav1)
    f2, t2, Sxx2 = compute_spectrogram(wav2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot spectrograms
    im1 = ax1.pcolormesh(t1, f1, Sxx1, shading='gouraud', cmap='inferno')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title(f'{title1} Spectrogram')
    fig.colorbar(im1, ax=ax1, label='Intensity [dB]')
    
    im2 = ax2.pcolormesh(t2, f2, Sxx2, shading='gouraud', cmap='inferno')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    ax2.set_title(f'{title2} Spectrogram')
    fig.colorbar(im2, ax=ax2, label='Intensity [dB]')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Spectrogram comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Also plot the difference spectrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute difference
    Sxx_diff = Sxx1 - Sxx2
    
    # Plot difference spectrogram
    im = ax.pcolormesh(t1, f1, Sxx_diff, shading='gouraud', cmap='coolwarm', vmin=-20, vmax=20)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(f'Spectrogram Difference ({title1} - {title2})')
    fig.colorbar(im, ax=ax, label='Difference [dB]')
    
    plt.tight_layout()
    
    if output_path:
        diff_path = output_path.replace('.png', '_diff.png')
        plt.savefig(diff_path)
        print(f"Difference spectrogram saved to: {diff_path}")
    else:
        plt.show()
        
    plt.close()

def plot_waveforms(wav1, wav2, title1="PyTorch", title2="ONNX", output_path=None):
    """Plot waveforms of two audio signals for comparison"""
    # Ensure same length for comparison
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]
    
    # Create time axis
    t = np.arange(min_len) / 16000  # Assuming 16kHz sampling rate
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot waveforms
    ax1.plot(t, wav1)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title1} Waveform')
    ax1.set_xlim(0, min_len/16000)
    
    ax2.plot(t, wav2)
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'{title2} Waveform')
    ax2.set_xlim(0, min_len/16000)
    
    # Plot difference
    ax3.plot(t, wav1 - wav2)
    ax3.set_ylabel('Difference')
    ax3.set_xlabel('Time [sec]')
    ax3.set_title(f'Waveform Difference ({title1} - {title2})')
    ax3.set_xlim(0, min_len/16000)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Waveform comparison saved to: {output_path}")
    else:
        plt.show()
        
    plt.close()

def compare_waveforms(wav1, wav2, name1="PyTorch", name2="ONNX", output_dir=None, visualize=False):
    """Compare two waveforms and print similarity metrics"""
    if wav1 is None or wav2 is None:
        print("Cannot compare waveforms: one or both outputs are missing")
        return
    
    # Ensure same length for comparison
    min_len = min(len(wav1), len(wav2))
    wav1_trim = wav1[:min_len]
    wav2_trim = wav2[:min_len]
    
    # Calculate key properties
    pt_duration = len(wav1) / 16000
    onnx_duration = len(wav2) / 16000
    duration_diff = abs(pt_duration - onnx_duration)
    
    # Normalize waveforms to ensure fair comparison
    def normalize_wav(wav):
        return wav / (np.max(np.abs(wav)) + 1e-9)
    
    wav1_norm = normalize_wav(wav1_trim)
    wav2_norm = normalize_wav(wav2_trim)
    
    # Calculate similarity metrics
    
    # 1. Mean Squared Error - lower is better
    mse = np.mean((wav1_norm - wav2_norm) ** 2)
    
    # 2. Mean Absolute Error - lower is better
    mae = np.mean(np.abs(wav1_norm - wav2_norm))
    
    # 3. Correlation coefficient - higher is better (closer to 1)
    correlation = np.corrcoef(wav1_norm, wav2_norm)[0, 1]
    
    # 4. Signal-to-Noise Ratio (SNR) - higher is better
    noise = wav1_norm - wav2_norm
    signal_power = np.mean(wav1_norm ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # 5. Frequency domain similarity - spectral correlation
    f1, t1, Sxx1 = compute_spectrogram(wav1_trim)
    f2, t2, Sxx2 = compute_spectrogram(wav2_trim)
    spectral_corr = np.corrcoef(Sxx1.flatten(), Sxx2.flatten())[0, 1]
    
    print("\n--- Waveform Comparison Results ---")
    print(f"{name1} audio duration: {pt_duration:.2f} seconds")
    print(f"{name2} audio duration: {onnx_duration:.2f} seconds")
    print(f"Duration difference: {duration_diff:.2f} seconds")
    print(f"Comparison on first {min_len/16000:.2f} seconds:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Time-domain correlation: {correlation:.6f}")
    print(f"Spectral correlation: {spectral_corr:.6f}")
    print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
    
    # Create visualizations if requested
    if visualize and output_dir:
        try:
            # Create plots
            plot_waveforms(
                wav1_trim, wav2_trim, 
                title1=name1, title2=name2,
                output_path=os.path.join(output_dir, "waveform_comparison.png")
            )
            
            plot_spectrograms(
                wav1_trim, wav2_trim, 
                title1=name1, title2=name2,
                output_path=os.path.join(output_dir, "spectrogram_comparison.png")
            )
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    # Provide a qualitative assessment
    print("\n--- Similarity Assessment ---")
    
    # Duration difference assessment
    if duration_diff > 2.0:
        print(f"❌ MAJOR DIFFERENCE: Audio durations differ by {duration_diff:.2f} seconds")
        print("   This indicates fundamental differences in how the models process the text.")
    
    # Correlation assessment
    if correlation > 0.7:
        print(f"✅ HIGH SIMILARITY: Time-domain correlation {correlation:.4f} indicates waveforms are highly similar")
    elif correlation > 0.5:
        print(f"⚠️ MODERATE SIMILARITY: Time-domain correlation {correlation:.4f} indicates moderate similarity")
    else:
        print(f"❌ LOW SIMILARITY: Time-domain correlation {correlation:.4f} indicates significant differences")
    
    # Spectral correlation assessment
    if spectral_corr > 0.7:
        print(f"✅ HIGH SPECTRAL SIMILARITY: Spectral correlation {spectral_corr:.4f} indicates similar frequency content")
    elif spectral_corr > 0.5:
        print(f"⚠️ MODERATE SPECTRAL SIMILARITY: Spectral correlation {spectral_corr:.4f} indicates moderate similarity in frequency content")
    else:
        print(f"❌ LOW SPECTRAL SIMILARITY: Spectral correlation {spectral_corr:.4f} indicates different frequency content")
    
    # SNR assessment
    if snr > 10:
        print(f"✅ HIGH SNR: SNR {snr:.2f} dB indicates low difference noise")
    elif snr > 0:
        print(f"⚠️ MODERATE SNR: SNR {snr:.2f} dB indicates noticeable differences")
    else:
        print(f"❌ LOW SNR: SNR {snr:.2f} dB indicates major differences")
    
    print("\n--- Implementation Differences Analysis ---")
    print("1. TEXT PROCESSING: The PyTorch model generates semantic tokens from text differently:")
    print("   - PyTorch: Uses a large language model (LLM) with a trained tokenizer")
    print("   - ONNX: Uses deterministic character-to-token mapping")
    
    print("2. ARCHITECTURE: The two implementations use different processing paths:")
    print("   - PyTorch: Full text → LLM → semantic tokens → audio")
    print("   - ONNX: Sentences → fixed-size token windows → audio segments → concatenation")
    
    print("3. VOICE CONTROL: Voice parameters are handled differently:")
    print("   - PyTorch: Uses categorical string parameters (male/female, very_low/low/etc.)")
    print("   - ONNX: Uses continuous float parameters from 0.0 to 1.0")
    
    if pt_duration > onnx_duration * 1.5:
        print("\n4. OUTPUT LENGTH: The ONNX model produced significantly shorter audio because:")
        print("   - It may be truncating long input text to fit fixed token windows")
        print("   - It uses direct sentence-by-sentence synthesis instead of global context")

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run PyTorch inference
    pt_wav, pt_time = run_pytorch_inference(args.model_dir, args.prompt_path, args.text)
    
    # Run ONNX inference
    onnx_wav, onnx_time = run_onnx_inference(args.model_dir, args.prompt_path, args.text)
    
    # If both ran successfully, print speedup
    if pt_time > 0 and onnx_time > 0:
        speedup = pt_time / onnx_time
        print(f"\n--- Performance Comparison ---")
        print(f"PyTorch inference time: {pt_time:.2f} seconds")
        print(f"ONNX inference time: {onnx_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
    
    # Save outputs if available
    if pt_wav is not None:
        pt_output_path = os.path.join(args.output_dir, "pytorch_output.wav")
        sf.write(pt_output_path, pt_wav, 16000)
        print(f"PyTorch output saved to: {pt_output_path}")
    
    if onnx_wav is not None:
        onnx_output_path = os.path.join(args.output_dir, "onnx_output.wav")
        sf.write(onnx_output_path, onnx_wav, 16000)
        print(f"ONNX output saved to: {onnx_output_path}")
    
    # Compare outputs if both are available
    if pt_wav is not None and onnx_wav is not None:
        compare_waveforms(pt_wav, onnx_wav, output_dir=args.output_dir, visualize=args.visualize)
    
if __name__ == "__main__":
    main() 
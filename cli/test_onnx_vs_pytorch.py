import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import warnings

# Suppress torchaudio sox warning if needed
warnings.filterwarnings("ignore", message=".*sox_io.load_audio_file failed UI_INITIALIZE*.")
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy argument number*.")


# Attempt to import necessary project modules
try:
    from cli.SparkTTS import SparkTTS # Assuming SparkTTS.py is in cli/
    from cli.run_onnx_inference import ONNXSparkTTSPredictor # Assuming run_onnx_inference.py is in cli/
    from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP # For controllable TTS args
except ImportError as e:
    print(f"Error importing project-specific modules: {e}. ")
    print("Please ensure that SparkTTS.py and run_onnx_inference.py are in the 'cli' directory relative to this script,")
    print("and the SparkTTS project is in PYTHONPATH or the script is run from project root.")
    # Define placeholders if import fails for basic script parsing
    LEVELS_MAP = {} 
    GENDER_MAP = {}
    # Define dummy classes if imports fail to allow script to be parsed
    class SparkTTS:
        def __init__(self, *args, **kwargs):
            raise ImportError("Original SparkTTS class could not be imported.")
        def inference(self, *args, **kwargs):
            raise ImportError("Original SparkTTS class could not be imported.")
    class ONNXSparkTTSPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("ONNXSparkTTSPredictor class could not be imported.")
        def inference(self, *args, **kwargs):
            raise ImportError("ONNXSparkTTSPredictor class could not be imported.")


def calculate_mse(audio1: np.ndarray, audio2: np.ndarray) -> float:
    min_len = min(len(audio1), len(audio2))
    if min_len == 0:
        return float('inf')
    return np.mean((audio1[:min_len] - audio2[:min_len])**2)

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    min_len = min(len(signal), len(noise))
    if min_len == 0:
        return 0.0 # Or float('-inf') for undefined
    signal = signal[:min_len]
    noise = noise[:min_len]
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0 or signal_power == 0: # Avoid log(0) or division by zero
        return float('inf') if noise_power == 0 and signal_power > 0 else 0.0
    return 10 * np.log10(signal_power / noise_power)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare PyTorch SparkTTS with ONNX SparkTTS inference.")
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize.")
    parser.add_argument("--pytorch_model_dir", required=True, type=str, help="Path to the original PyTorch model directory (e.g., pretrained_models/Spark-TTS-0.5B).")
    parser.add_argument("--onnx_model_dir", required=True, type=str, help="Directory containing all ONNX model files (e.g., onnx_models/Spark-TTS-0.5B).")
    parser.add_argument("--llm_tokenizer_dir", required=True, type=str, help="Directory of the LLM tokenizer (e.g., pretrained_models/Spark-TTS-0.5B/LLM).")
    parser.add_argument("--bicodec_config", type=str, default=None, help="Path to BiCodec's config.yaml for ONNX predictor. If None, attempts to find it relative to onnx_model_dir.")

    parser.add_argument("--prompt_audio", type=str, default=None, help="Path to prompt audio file for voice cloning mode.")
    parser.add_argument("--prompt_text", type=str, default=None, help="Transcript of the prompt audio (optional).")
    
    parser.add_argument("--gender", type=str, choices=list(GENDER_MAP.keys()), default=None, help="Gender for controllable TTS mode.")
    parser.add_argument("--pitch", type=str, choices=list(LEVELS_MAP.keys()), default=None, help="Pitch for controllable TTS mode.")
    parser.add_argument("--speed", type=str, choices=list(LEVELS_MAP.keys()), default=None, help="Speed for controllable TTS mode.")

    parser.add_argument("--output_pytorch_wav", type=str, default="output_pytorch_test.wav", help="Path to save PyTorch generated waveform.")
    parser.add_argument("--output_onnx_wav", type=str, default="output_onnx_test.wav", help="Path to save ONNX generated waveform.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for PyTorch and ONNX Runtime.")
    
    # Common inference parameters for fair comparison
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p for LLM.")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens for LLM generation.")

    args = parser.parse_args()

    pytorch_device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {pytorch_device} for PyTorch model.")
    onnx_device_arg = args.device # Passed to ONNXSparkTTSPredictor

    is_voice_cloning_mode = bool(args.prompt_audio)
    is_controllable_mode = all([args.gender, args.pitch, args.speed])

    if not is_voice_cloning_mode and not is_controllable_mode:
        print("Error: Test mode unclear. Provide EITHER --prompt_audio (for voice cloning) OR (--gender, --pitch, --speed) for controllable TTS.")
        exit(1)
    if is_voice_cloning_mode and is_controllable_mode:
        print("Warning: Both prompt_audio and controllable TTS args provided. Prioritizing voice cloning mode.")
        is_controllable_mode = False # Prioritize cloning if both are given

    # 1. Run PyTorch Inference
    print("\n" + "-"*10 + " Running PyTorch SparkTTS Inference " + "-"*10)
    wav_pytorch_np = None
    pytorch_sample_rate = 16000 # Default, will be updated by model
    try:
        pytorch_model = SparkTTS(model_dir=Path(args.pytorch_model_dir), device=pytorch_device)
        pytorch_sample_rate = pytorch_model.sample_rate
        
        common_inference_params = {
            "text": args.text,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens # SparkTTS.py uses `max_new_tokens`
        }

        if is_voice_cloning_mode:
            print("PyTorch Mode: Voice Cloning")
            wav_pytorch_tensor, _, _, _ = pytorch_model.inference(
                **common_inference_params,
                prompt_speech_path=Path(args.prompt_audio),
                prompt_text=args.prompt_text
            )
        elif is_controllable_mode:
            print("PyTorch Mode: Controllable TTS")
            wav_pytorch_tensor, _, _, _ = pytorch_model.inference(
                **common_inference_params,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed
            )
        
        if wav_pytorch_tensor is not None:
            wav_pytorch_np = wav_pytorch_tensor.cpu().numpy().squeeze()
            torchaudio.save(args.output_pytorch_wav, torch.from_numpy(wav_pytorch_np).unsqueeze(0), pytorch_sample_rate)
            print(f"PyTorch audio saved to {args.output_pytorch_wav} (Length: {len(wav_pytorch_np)})")

    except Exception as e:
        print(f"Error during PyTorch inference: {e}")
        if isinstance(e, ImportError):
            print("Skipping PyTorch inference due to import error.")


    # 2. Run ONNX Inference
    print("\n" + "-"*10 + " Running ONNX SparkTTS Inference " + "-"*10)
    wav_onnx_np = None
    onnx_sample_rate = 16000 # Default, will be updated by predictor
    try:
        bicodec_cfg_path_onnx = Path(args.bicodec_config) if args.bicodec_config else Path(args.onnx_model_dir).parent / "BiCodec" / "config.yaml"
        onnx_predictor = ONNXSparkTTSPredictor(
            Path(args.onnx_model_dir),
            Path(args.llm_tokenizer_dir),
            bicodec_config_path=bicodec_cfg_path_onnx,
            device=onnx_device_arg
        )
        onnx_sample_rate = onnx_predictor.sample_rate # Get actual sample rate

        common_onnx_params = {
            "text": args.text,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens_llm": args.max_new_tokens # ONNX script uses `max_new_tokens_llm`
        }

        if is_voice_cloning_mode:
            print("ONNX Mode: Voice Cloning")
            wav_onnx_np = onnx_predictor.inference(
                **common_onnx_params,
                prompt_speech_path=args.prompt_audio,
                prompt_text=args.prompt_text
            )
        elif is_controllable_mode:
            print("ONNX Mode: Controllable TTS")
            wav_onnx_np = onnx_predictor.inference(
                **common_onnx_params,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed
            )

        if wav_onnx_np is not None and wav_onnx_np.size > 0:
            # Ensure waveform is 1D or 2D for torchaudio.save
            if wav_onnx_np.ndim == 1:
                wav_onnx_torch = torch.from_numpy(wav_onnx_np).unsqueeze(0)
            elif wav_onnx_np.ndim == 2:
                wav_onnx_torch = torch.from_numpy(wav_onnx_np)
            else:
                print(f"Error: ONNX generated waveform has unexpected ndim: {wav_onnx_np.ndim}")
                wav_onnx_torch = None
            
            if wav_onnx_torch is not None:
                torchaudio.save(args.output_onnx_wav, wav_onnx_torch, onnx_sample_rate)
                print(f"ONNX audio saved to {args.output_onnx_wav} (Length: {len(wav_onnx_np)})")
        else:
            print("ONNX inference did not produce a valid waveform.")
            wav_onnx_np = None # Ensure it's None if empty or invalid for comparison logic

    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        if isinstance(e, ImportError):
            print("Skipping ONNX inference due to import error of its dependencies.")


    # 3. Compare Outputs
    print("\n" + "-"*10 + " Comparison Results " + "-"*10)
    if pytorch_sample_rate != onnx_sample_rate:
        print(f"Warning: Sample rates differ! PyTorch: {pytorch_sample_rate}, ONNX: {onnx_sample_rate}. Comparison might be less meaningful.")

    if wav_pytorch_np is not None and wav_onnx_np is not None and wav_pytorch_np.size > 0 and wav_onnx_np.size > 0:
        mse = calculate_mse(wav_pytorch_np, wav_onnx_np)
        print(f"Mean Squared Error (MSE) between waveforms: {mse:.6e}")

        noise_signal = wav_pytorch_np - wav_onnx_np
        snr = calculate_snr(wav_pytorch_np, noise_signal)
        print(f"Signal-to-Noise Ratio (SNR) (PyTorch as signal, Diff as noise): {snr:.2f} dB")

        if mse < 1e-5: 
            print("Result: Outputs are numerically very close (MSE < 1e-5).")
        elif mse < 1e-4:
            print("Result: Outputs are reasonably close (MSE < 1e-4).")
        else:
            print("Result: Outputs have noticeable differences (MSE >= 1e-4).")
        
        len_diff = abs(len(wav_pytorch_np) - len(wav_onnx_np))
        print(f"Length: PyTorch: {len(wav_pytorch_np)}, ONNX: {len(wav_onnx_np)} (Difference: {len_diff})")
        if len_diff > 0:
            print("Note: Waveform lengths differ, comparison was done on the shorter segment.")
            if len_diff > 0.1 * max(len(wav_pytorch_np), len(wav_onnx_np)) : # If diff > 10%
                 print("Warning: Significant length difference detected. This might indicate issues in EOS handling or max_tokens.")
    else:
        print("Could not compare audio outputs: one or both failed to generate or produced empty audio.")

    print("\nTest script finished. Check saved .wav files and comparison metrics.") 
import torch
import torchaudio
import torchaudio.transforms as TT
import torchaudio.functional as TF # For melscale_fbanks if MelSpectrogramONNXWrapper is used directly
import numpy as np
import onnxruntime
import argparse
from pathlib import Path
import yaml # For loading config.yaml
import math # For MelSpectrogramONNXWrapper

# --- Helper to load config ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- MelSpectrogramCalculator (adapted from export_mel_spectrogram_onnx.py) ---
# This class is included to ensure Mel calculation matches the ONNX export logic,
# providing the correct input format for the speaker_encoder_tokenizer.onnx.
class MelSpectrogramCalculator(torch.nn.Module):
    def __init__(self, mel_params: dict, device: torch.device):
        super().__init__()
        
        self.n_fft = mel_params['n_fft']
        self.hop_length = mel_params.get('hop_length', self.n_fft // 4)
        self.win_length = mel_params.get('win_length', self.n_fft)
        self.sample_rate = mel_params['sample_rate']
        
        window_fn_name = mel_params.get('window_fn', 'hann_window')
        if window_fn_name == 'hann_window':
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        elif window_fn_name == 'hamming_window':
            window_tensor = torch.hamming_window(self.win_length, periodic=True, dtype=torch.float32)
        else:
            print(f"Warning: Unrecognized window_fn '{window_fn_name}', defaulting to Hann window.")
            window_tensor = torch.hann_window(self.win_length, periodic=True, dtype=torch.float32)
        self.register_buffer('window', window_tensor.to(device))

        self.center = mel_params.get('center', True)
        self.pad_mode = mel_params.get('pad_mode', "reflect") 
        self.power = mel_params.get('power', 1.0) 
        
        n_stft = self.n_fft // 2 + 1
        f_min = mel_params.get('mel_fmin', 0.0)
        f_max_param = mel_params.get('mel_fmax')
        f_max = f_max_param if f_max_param is not None else self.sample_rate / 2.0
        n_mels = mel_params['num_mels']
        mel_norm = mel_params.get('norm', 'slaney') 
        mel_scale_type = mel_params.get('mel_scale', 'slaney')

        mel_fbanks_tensor = TF.melscale_fbanks(
            n_freqs=n_stft, f_min=f_min, f_max=f_max, n_mels=n_mels,
            sample_rate=self.sample_rate, norm=mel_norm, mel_scale=mel_scale_type
        )
        self.register_buffer('mel_fbanks', mel_fbanks_tensor.to(device))

        k = torch.arange(0, self.n_fft // 2 + 1, dtype=torch.float32, device=device)
        n_range = torch.arange(0, self.n_fft, dtype=torch.float32, device=device)
        angle = -2 * math.pi * k.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
        
        self.register_buffer('rfft_mat_real', torch.cos(angle))
        self.register_buffer('rfft_mat_imag', torch.sin(angle))
        self.to(device)


    def forward(self, wav_with_channel: torch.Tensor) -> torch.Tensor:
        wav = wav_with_channel.squeeze(1) # Expect (B, 1, T_audio) -> (B, T_audio)
        batch_size = wav.shape[0]
        
        padded_wav = wav
        if self.center:
            padding_amount = self.n_fft // 2
            padded_wav = torch.nn.functional.pad(wav, (padding_amount, padding_amount), mode=self.pad_mode)
        
        num_frames = (padded_wav.shape[1] - self.win_length) // self.hop_length + 1
        if num_frames <= 0: # Should not happen with typical audio, but good practice
             print(f"Warning: Input audio too short for any frames. num_frames={num_frames}")
             return torch.empty((batch_size, self.mel_fbanks.shape[1], 0), device=wav.device, dtype=wav.dtype) # (B, n_mels, 0_frames)

        frame_list = []
        for i in range(num_frames):
            start = i * self.hop_length
            frame = padded_wav.narrow(1, start, self.win_length)
            frame_list.append(frame)
        frames = torch.stack(frame_list, dim=1) # (B, num_frames, win_length)
        
        windowed_frames = frames * self.window # (B, num_frames, win_length)
        
        fft_ready_frames = windowed_frames
        if self.n_fft > self.win_length: # Pad
            fft_ready_frames = torch.nn.functional.pad(windowed_frames, (0, self.n_fft - self.win_length))
        elif self.n_fft < self.win_length: # Truncate
            fft_ready_frames = windowed_frames[:, :, :self.n_fft]
        # Now fft_ready_frames is (B, num_frames, n_fft)

        rfft_mat_real_t = self.rfft_mat_real.T
        rfft_mat_imag_t = self.rfft_mat_imag.T
        real_part = torch.matmul(fft_ready_frames, rfft_mat_real_t) # (B, num_frames, n_fft // 2 + 1)
        imag_part = torch.matmul(fft_ready_frames, rfft_mat_imag_t) # (B, num_frames, n_fft // 2 + 1)
        
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))
        if self.power != 1.0:
            magnitude = magnitude.pow(self.power)
        
        # self.mel_fbanks is (n_stft_bins, n_mels) e.g. (513, 128) from TF.melscale_fbanks
        # magnitude is (B, num_frames, n_stft_bins)
        # We need matmul: (B, num_frames, n_stft_bins) @ (n_stft_bins, n_mels)
        mel_output = torch.matmul(magnitude, self.mel_fbanks) # (B, num_frames, n_mels)
        
        # Transpose to conventional (B, n_mels, num_frames)
        mel_output = mel_output.transpose(1, 2) 
        return mel_output

def main():
    parser = argparse.ArgumentParser(description="Extract Speaker Tokens from an audio file using ONNX for both Mel and Speaker Encoding.")
    parser.add_argument(
        "--model_base_dir", 
        type=str, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B). Should contain BiCodec/config.yaml for target_sample_rate."
    )
    parser.add_argument(
        "--mel_onnx_path",
        type=str,
        required=True,
        help="Path to the Mel Spectrogram ONNX model."
    )
    parser.add_argument(
        "--speaker_encoder_onnx_path", 
        type=str, 
        required=True, 
        help="Path to the Speaker Encoder Tokenizer ONNX model."
    )
    parser.add_argument(
        "--audio_file_path", 
        type=str, 
        required=True, 
        help="Path to the input audio WAV file."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch processing if any ('cpu' or 'cuda'). ONNX runs on CPU by default unless provider specified.")
    
    args = parser.parse_args()
    # Device for PyTorch (audio loading/resampling)
    pytorch_device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')


    # 1. Load target_sample_rate from BiCodec config
    bicodec_config_path = Path(args.model_base_dir) / "BiCodec" / "config.yaml"
    if not bicodec_config_path.exists():
        print(f"ERROR: BiCodec config.yaml not found at {bicodec_config_path}")
        return
    
    full_config = load_config(bicodec_config_path)
    # We only need mel_params for the target_sample_rate now
    if 'audio_tokenizer' not in full_config or 'mel_params' not in full_config['audio_tokenizer']:
        print(f"ERROR: 'mel_params' not found in {bicodec_config_path} under 'audio_tokenizer' key.")
        return
    mel_parameters = full_config['audio_tokenizer']['mel_params']
    target_sample_rate = mel_parameters['sample_rate']
    # num_mel_channels_expected_by_config = mel_parameters['num_mels'] # Might not be needed if Mel ONNX output is trusted
    print(f"Target sample rate for audio (from config): {target_sample_rate}")

    # 2. Load and preprocess audio (using PyTorch)
    try:
        waveform, sample_rate = torchaudio.load(args.audio_file_path)
        waveform = waveform.to(pytorch_device)
    except Exception as e:
        print(f"Error loading audio file {args.audio_file_path}: {e}")
        return

    if sample_rate != target_sample_rate:
        print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz...")
        resampler = TT.Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to(pytorch_device)
        waveform = resampler(waveform)
    else:
        print(f"Audio sample rate ({sample_rate} Hz) matches target rate ({target_sample_rate} Hz). Skipping resampling.")
    
    if waveform.ndim == 1: 
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > 1 : 
        print("Audio appears to be stereo, taking the first channel.")
        waveform = waveform[0, :].unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] == 1: 
        waveform = waveform.unsqueeze(0) 
    
    if waveform.shape[0] != 1 or waveform.shape[1] != 1:
         print(f"ERROR: Waveform processing resulted in unexpected shape: {waveform.shape}. Expected (1, 1, Time).")
         return
    print(f"Loaded audio waveform shape for Mel ONNX (B, C, T): {waveform.shape}")
    waveform_np = waveform.cpu().numpy() # ONNX runtime expects numpy

    # ---- START: Dump raw audio samples for comparison ----
    print(f"Python Raw Audio (waveform_np) Shape: {waveform_np.shape}")
    print(f"Python Raw Audio (waveform_np) Length: {waveform_np.size}")
    print("Python Raw Audio (waveform_np) Data (first 50 flat, F8 format):")
    flat_waveform_np = waveform_np.flatten()
    for i, val in enumerate(flat_waveform_np[:50]):
        print(f"{val:.8f}", end=", ")
        if (i + 1) % 10 == 0: print("") 
    print("\\n")

    if waveform_np.size > 1000:
        print("Python Raw Audio (waveform_np) Data (middle 10 flat around index 500, F8 format):")
        for i, val in enumerate(flat_waveform_np[500:510]):
            print(f"{val:.8f}", end=", ")
        print("\\n")
    
    np.save("python_audio_for_mel_onnx.npy", waveform_np)
    print("Saved python_audio_for_mel_onnx.npy")
    # ---- END: Dump raw audio samples for comparison ----

    # 3. Calculate Mel Spectrogram using the provided Mel ONNX model
    mel_spectrogram_onnx_output = None
    try:
        print(f"Attempting to load Mel ONNX model: {args.mel_onnx_path}")
        mel_ort_session_options = onnxruntime.SessionOptions()
        mel_ort_providers = ['CPUExecutionProvider'] # Default to CPU for Mel
        mel_ort_session = onnxruntime.InferenceSession(args.mel_onnx_path, sess_options=mel_ort_session_options, providers=mel_ort_providers)
        
        mel_onnx_input_name = mel_ort_session.get_inputs()[0].name
        mel_onnx_output_name = mel_ort_session.get_outputs()[0].name
        
        print(f"Mel ONNX Input Name: '{mel_onnx_input_name}', Mel ONNX Output Name: '{mel_onnx_output_name}'")
        # The input for mel_spectrogram.onnx from export_mel_spectrogram_onnx.py is "raw_waveform_with_channel"
        # which matches our waveform_np shape (1, 1, T_audio)
        mel_ort_inputs = {mel_onnx_input_name: waveform_np}
        mel_spectrogram_onnx_output_list = mel_ort_session.run([mel_onnx_output_name], mel_ort_inputs)
        mel_spectrogram_onnx_output = mel_spectrogram_onnx_output_list[0] # This is a numpy array

        print(f"Mel Spectrogram from ONNX shape (B, n_mels, T_frames): {mel_spectrogram_onnx_output.shape}")

    except Exception as e:
        print(f"Error during Mel ONNX inference: {e}")
        import traceback
        traceback.print_exc()
        return

    if mel_spectrogram_onnx_output is None:
        print("Mel ONNX inference failed to produce output.")
        return

    # The speaker encoder ONNX model expects input shape (batch_size, mel_time_steps, mel_channels/features).
    # The Mel ONNX output is likely (B, n_mels, T_frames).
    # So, transpose (B, n_mels, T_frames) to (B, T_frames, n_mels).
    # n_mels is feature_channels (dim 1 of ONNX output), T_frames is time_steps (dim 2 of ONNX output)
    if mel_spectrogram_onnx_output.ndim == 3:
        mel_input_for_speaker_encoder_onnx = np.ascontiguousarray(np.transpose(mel_spectrogram_onnx_output, (0, 2, 1)))
    else:
        print(f"Mel ONNX output has unexpected ndim: {mel_spectrogram_onnx_output.ndim}. Expected 3. Cannot transpose.")
        return
        
    print(f"Mel spectrogram (from Mel ONNX) reshaped for Speaker Encoder ONNX (B, T_frames, n_mels): {mel_input_for_speaker_encoder_onnx.shape}")
    
    # Save this for comparison with C#
    np.save("python_mel_for_speaker_encoder.npy", mel_input_for_speaker_encoder_onnx)
    print("Saved python_mel_for_speaker_encoder.npy")


    # 4. Run Speaker Encoder ONNX model
    try:
        print(f"Attempting to load Speaker Encoder ONNX model: {args.speaker_encoder_onnx_path}")
        spk_ort_session_options = onnxruntime.SessionOptions()
        spk_ort_providers = ['CPUExecutionProvider']
        if args.device == 'cuda' and onnxruntime.get_device() == 'GPU': # Check if onnxruntime-gpu is installed and CUDA is available
             # Only add CUDAExecutionProvider if a GPU is actually available to ONNXRuntime
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                spk_ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("CUDAExecutionProvider not available in ONNXRuntime. Using CPU for Speaker Encoder.")


        spk_ort_session = onnxruntime.InferenceSession(args.speaker_encoder_onnx_path, sess_options=spk_ort_session_options, providers=spk_ort_providers)
        
        spk_onnx_input_name = spk_ort_session.get_inputs()[0].name
        spk_onnx_output_name = spk_ort_session.get_outputs()[0].name
        # spk_onnx_input_shape_expected = spk_ort_session.get_inputs()[0].shape 
        # print(f"Speaker Encoder ONNX Input Name: '{spk_onnx_input_name}', Expected Shape (from model): {spk_onnx_input_shape_expected}")
        print(f"Speaker Encoder ONNX Input Name: '{spk_onnx_input_name}', Output Name: '{spk_onnx_output_name}'")
        
        # mel_input_for_speaker_encoder_onnx is already a numpy array
        spk_ort_inputs = {spk_onnx_input_name: mel_input_for_speaker_encoder_onnx}
        spk_ort_outputs = spk_ort_session.run([spk_onnx_output_name], spk_ort_inputs)
        
        speaker_tokens_np = spk_ort_outputs[0] 
        print(f"ONNX Speaker Encoder successful. Output tokens name: '{spk_onnx_output_name}', shape: {speaker_tokens_np.shape}")
        print("Speaker Tokens (numeric IDs from Python ONNX):")
        print(speaker_tokens_np)

        if speaker_tokens_np.ndim == 3 and speaker_tokens_np.shape[0] == 1: 
            flattened_tokens = speaker_tokens_np[0].flatten().astype(int)
            print("Flattened Speaker Tokens for easy comparison (IDs as integers):")
            print(flattened_tokens.tolist())
        elif speaker_tokens_np.ndim == 2 and speaker_tokens_np.shape[0] == 1: 
            flattened_tokens = speaker_tokens_np[0].astype(int)
            print("Flattened Speaker Tokens for easy comparison (IDs as integers):")
            print(flattened_tokens.tolist())
        else:
            print("Output token array is not of expected shape (1, Nq, NumTokens) or (1, NumTokensFlat). Manual inspection needed.")

    except Exception as e:
        print(f"Error during ONNX Speaker Encoder inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
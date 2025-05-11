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
from transformers import AutoTokenizer # Added for LLM tokenizer

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

def get_llm_model_dims(ort_session):
    num_layers = 0
    num_heads = 0
    head_dim = 0

    for i_input in ort_session.get_inputs():
        if "past_key_values" in i_input.name and ".key" in i_input.name: # e.g. past_key_values.0.key
            try:
                layer_index = int(i_input.name.split('.')[1])
                if layer_index + 1 > num_layers:
                    num_layers = layer_index + 1
                
                # Shape: (batch_size, num_heads, past_seq_len, head_dim)
                shape = i_input.shape
                if len(shape) == 4:
                    if num_heads == 0: # Infer from first one found
                        num_heads = shape[1] if isinstance(shape[1], int) else 0
                    if head_dim == 0: # Infer from first one found
                        head_dim = shape[3] if isinstance(shape[3], int) else 0
            except Exception as e:
                print(f"Warning: Could not parse layer_index or dims from input {i_input.name}: {e}")

    if num_layers == 0: # Fallback if not found in inputs (e.g. model that only has present_outputs)
         for o_output in ort_session.get_outputs():
            if "present" in o_output.name and ".key" in o_output.name: # e.g. present.0.key
                try:
                    layer_index = int(o_output.name.split('.')[1])
                    if layer_index + 1 > num_layers:
                        num_layers = layer_index + 1
                    shape = o_output.shape
                    if len(shape) == 4: # (batch_size, num_heads, seq_len, head_dim)
                         if num_heads == 0: num_heads = shape[1] if isinstance(shape[1], int) else 0
                         if head_dim == 0: head_dim = shape[3] if isinstance(shape[3], int) else 0
                except:
                    pass # ignore parsing errors

    if num_layers == 0 or num_heads == 0 or head_dim == 0:
        print(f"Warning: Could not reliably infer all LLM dimensions. Found: Layers={num_layers}, Heads={num_heads}, HeadDim={head_dim}. Using defaults if zero.")
        # Provide some defaults if any are zero, though this is a sign of issue.
        if num_layers == 0: num_layers = 32 # Common default
        if num_heads == 0: num_heads = 32 # Common default
        if head_dim == 0: head_dim = 128 # Common default
        print(f"Updated to: Layers={num_layers}, Heads={num_heads}, HeadDim={head_dim} (defaults applied where necessary).")


    print(f"Inferred LLM Dims: Layers={num_layers}, Heads={num_heads}, HeadDim={head_dim}")
    return num_layers, num_heads, head_dim

def main():
    parser = argparse.ArgumentParser(description="Extract Speaker Tokens from an audio file using ONNX for both Mel and Speaker Encoding, and optionally generate LLM input_ids and semantic tokens.")
    parser.add_argument(
        "--model_base_dir", 
        type=str, 
        required=True,
        help="Path to the base SparkTTS model directory (e.g., pretrained_models/Spark-TTS-0.5B). Should contain BiCodec/config.yaml and LLM/tokenizer_config.json."
    )
    parser.add_argument(
        "--mel_onnx_path",
        type=str,
        help="Path to the Mel Spectrogram ONNX model. Required if not using --use_fixed_speaker_tokens for LLM input ID generation, or if not generating LLM IDs at all."
    )
    parser.add_argument(
        "--speaker_encoder_onnx_path", 
        type=str, 
        help="Path to the Speaker Encoder Tokenizer ONNX model. Required if not using --use_fixed_speaker_tokens for LLM input ID generation, or if not generating LLM IDs at all."
    )
    parser.add_argument(
        "--audio_file_path", 
        type=str, 
        help="Path to the input audio WAV file. Required if not using --use_fixed_speaker_tokens for LLM input ID generation, or if not generating LLM IDs at all."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch processing if any ('cpu' or 'cuda'). ONNX runs on CPU by default unless provider specified.")
    
    parser.add_argument(
        "--process_for_llm_input_ids",
        action="store_true",
        help="If set, the script will also generate and print input_ids for the LLM based on the provided text and speaker tokens."
    )
    parser.add_argument(
        "--text_to_synthesize",
        type=str,
        default="Hello world",
        help="Text to use for generating LLM input_ids. Only used if --process_for_llm_input_ids is set."
    )
    parser.add_argument(
        "--use_fixed_speaker_tokens",
        type=str,
        default=None,
        help="Comma-separated string of integer speaker token IDs. If provided and --process_for_llm_input_ids is set, these tokens will be used instead of deriving them from audio."
    )
    parser.add_argument(
        "--llm_onnx_path",
        type=str,
        help="Path to the LLM ONNX model. Required if --generate_semantic_tokens is set."
    )
    parser.add_argument(
        "--generate_semantic_tokens",
        action="store_true",
        help="If set, the script will run the LLM ONNX model to generate semantic tokens."
    )
    parser.add_argument(
        "--max_new_semantic_tokens",
        type=int,
        default=768, # Typical value for semantic tokens
        help="Maximum number of new semantic tokens to generate. Only used if --generate_semantic_tokens is set."
    )
    parser.add_argument(
        "--llm_eos_token_id",
        type=int,
        default=151645, # Default for <|im_end|>
        help="EOS token ID for LLM generation."
    )

    args = parser.parse_args()

    # Determine if audio processing pipeline is needed
    needs_audio_pipeline = False
    if not args.process_for_llm_input_ids: # Original mode always needs audio pipeline
        needs_audio_pipeline = True
    elif args.process_for_llm_input_ids and not args.use_fixed_speaker_tokens: # LLM mode, but need to derive speaker tokens
        needs_audio_pipeline = True

    if needs_audio_pipeline:
        if not all([args.audio_file_path, args.mel_onnx_path, args.speaker_encoder_onnx_path]):
            parser.error("When deriving speaker tokens from audio (either for direct output or for LLM input ID generation without --use_fixed_speaker_tokens), --audio_file_path, --mel_onnx_path, and --speaker_encoder_onnx_path are required.")

    if args.generate_semantic_tokens and not args.llm_onnx_path:
        parser.error("--llm_onnx_path is required if --generate_semantic_tokens is set.")
    if args.generate_semantic_tokens and not args.process_for_llm_input_ids:
        print("Warning: --generate_semantic_tokens is set, but --process_for_llm_input_ids is not. Will proceed but this might not be intended if LLM input_ids are also needed.")

    pytorch_device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # Initialize flattened_tokens for use later
    flattened_tokens = None 

    if needs_audio_pipeline:
        bicodec_config_path = Path(args.model_base_dir) / "BiCodec" / "config.yaml"
        if not bicodec_config_path.exists():
            print(f"ERROR: BiCodec config.yaml not found at {bicodec_config_path}")
            return
        
        full_config = load_config(bicodec_config_path)
        if 'audio_tokenizer' not in full_config or 'mel_params' not in full_config['audio_tokenizer']:
            print(f"ERROR: 'mel_params' not found in {bicodec_config_path} under 'audio_tokenizer' key.")
            return
        mel_parameters = full_config['audio_tokenizer']['mel_params']
        target_sample_rate = mel_parameters['sample_rate']
        print(f"Target sample rate for audio (from config): {target_sample_rate}")

        try:
            waveform, sample_rate = torchaudio.load(args.audio_file_path) # audio_file_path is guaranteed to be non-None if needs_audio_pipeline is true
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
        waveform_np = waveform.cpu().numpy()

        print(f"Python Raw Audio (waveform_np) Shape: {waveform_np.shape}")
        print(f"Python Raw Audio (waveform_np) Length: {waveform_np.size}")
        flat_waveform_np_for_print = waveform_np.flatten() # Create a new variable for printing
        print("Python Raw Audio (waveform_np) Data (first 50 flat, F8 format):")
        for i, val in enumerate(flat_waveform_np_for_print[:50]): # Use the new variable
            print(f"{val:.8f}", end=", ")
            if (i + 1) % 10 == 0: print("") 
        print("\\\\n")
        if waveform_np.size > 1000: # Use original waveform_np for size check
            print("Python Raw Audio (waveform_np) Data (middle 10 flat around index 500, F8 format):")
            for i, val in enumerate(flat_waveform_np_for_print[500:510]): # Use new variable
                print(f"{val:.8f}", end=", ")
            print("\\\\n")
        np.save("python_audio_for_mel_onnx.npy", waveform_np)
        print("Saved python_audio_for_mel_onnx.npy")

        mel_spectrogram_onnx_output = None
        try:
            print(f"Attempting to load Mel ONNX model: {args.mel_onnx_path}")
            mel_ort_session = onnxruntime.InferenceSession(args.mel_onnx_path, providers=['CPUExecutionProvider'])
            mel_onnx_input_name = mel_ort_session.get_inputs()[0].name
            mel_onnx_output_name = mel_ort_session.get_outputs()[0].name
            print(f"Mel ONNX Input Name: '{mel_onnx_input_name}', Mel ONNX Output Name: '{mel_onnx_output_name}'")
            mel_ort_inputs = {mel_onnx_input_name: waveform_np}
            mel_spectrogram_onnx_output = mel_ort_session.run([mel_onnx_output_name], mel_ort_inputs)[0]
            print(f"Mel Spectrogram from ONNX shape (B, n_mels, T_frames): {mel_spectrogram_onnx_output.shape}")
        except Exception as e:
            print(f"Error during Mel ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return

        if mel_spectrogram_onnx_output is None:
            print("Mel ONNX inference failed to produce output.")
            return

        if mel_spectrogram_onnx_output.ndim == 3:
            mel_input_for_speaker_encoder_onnx = np.ascontiguousarray(np.transpose(mel_spectrogram_onnx_output, (0, 2, 1)))
        else:
            print(f"Mel ONNX output has unexpected ndim: {mel_spectrogram_onnx_output.ndim}. Expected 3.")
            return
            
        print(f"Mel spectrogram (from Mel ONNX) reshaped for Speaker Encoder ONNX (B, T_frames, n_mels): {mel_input_for_speaker_encoder_onnx.shape}")
        np.save("python_mel_for_speaker_encoder.npy", mel_input_for_speaker_encoder_onnx)
        print("Saved python_mel_for_speaker_encoder.npy")

        try:
            print(f"Attempting to load Speaker Encoder ONNX model: {args.speaker_encoder_onnx_path}")
            spk_providers = ['CPUExecutionProvider']
            if args.device == 'cuda' and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                spk_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            spk_ort_session = onnxruntime.InferenceSession(args.speaker_encoder_onnx_path, providers=spk_providers)
            spk_onnx_input_name = spk_ort_session.get_inputs()[0].name
            spk_onnx_output_name = spk_ort_session.get_outputs()[0].name
            print(f"Speaker Encoder ONNX Input Name: '{spk_onnx_input_name}', Output Name: '{spk_onnx_output_name}'")
            spk_ort_inputs = {spk_onnx_input_name: mel_input_for_speaker_encoder_onnx}
            speaker_tokens_np = spk_ort_session.run([spk_onnx_output_name], spk_ort_inputs)[0] 
            print(f"ONNX Speaker Encoder successful. Output tokens name: '{spk_onnx_output_name}', shape: {speaker_tokens_np.shape}")
            print("Speaker Tokens (numeric IDs from Python ONNX):")
            print(speaker_tokens_np)

            if speaker_tokens_np.ndim == 3 and speaker_tokens_np.shape[0] == 1: 
                flattened_tokens = speaker_tokens_np[0].flatten().astype(int).tolist()
            elif speaker_tokens_np.ndim == 2 and speaker_tokens_np.shape[0] == 1: 
                flattened_tokens = speaker_tokens_np[0].astype(int).tolist()
            else:
                print("Output token array is not of expected shape. Manual inspection needed.")
                flattened_tokens = [] 
            print("Flattened Speaker Tokens for easy comparison (IDs as integers):")
            print(flattened_tokens)

        except Exception as e:
            print(f"Error during ONNX Speaker Encoder inference: {e}")
            import traceback
            traceback.print_exc()
            return 
    # End of needs_audio_pipeline block

    if args.process_for_llm_input_ids:
        print("\\\\n--- Generating LLM Input IDs ---")
        
        speaker_tokens_for_llm_prompt = []
        if args.use_fixed_speaker_tokens:
            try:
                speaker_tokens_for_llm_prompt = [int(t.strip()) for t in args.use_fixed_speaker_tokens.split(',')]
                print(f"Using fixed speaker tokens for LLM prompt: {speaker_tokens_for_llm_prompt}")
            except ValueError as e:
                print(f"Error parsing --use_fixed_speaker_tokens: {e}. Ensure it's a comma-separated list of integers.")
                return
        elif flattened_tokens is not None: 
            speaker_tokens_for_llm_prompt = flattened_tokens
            print(f"Using speaker tokens derived from audio for LLM prompt: {speaker_tokens_for_llm_prompt}")
        else:
            # This case should ideally be caught by the arg validation if deriving tokens was expected but failed.
            # But if fixed tokens were expected but not provided, this is another fallback.
            print("Error: --process_for_llm_input_ids is set, but no speaker tokens are available. Ensure --use_fixed_speaker_tokens is provided or audio pipeline ran successfully.")
            return

        llm_tokenizer_path = Path(args.model_base_dir) / "LLM"
        if not (llm_tokenizer_path / "tokenizer.json").exists(): # Check for tokenizer.json specifically
            print(f"LLM tokenizer.json not found at {llm_tokenizer_path / 'tokenizer.json'}")
            print(f"Ensure --model_base_dir ('{args.model_base_dir}') points to a directory containing the LLM subdirectory with 'tokenizer.json'.")
            return
            
        try:
            # Pass the directory path to from_pretrained
            llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_tokenizer_path), trust_remote_code=True) # Added trust_remote_code
            print(f"Successfully loaded LLM tokenizer from {llm_tokenizer_path}")
        except Exception as e:
            print(f"Error loading LLM tokenizer from {llm_tokenizer_path}: {e}")
            return
        
        prompt_parts = ["<|task_tts|>", "<|start_content|>", args.text_to_synthesize, "<|end_content|>"]
        if speaker_tokens_for_llm_prompt: # Only add speaker token parts if list is not empty
            prompt_parts.append("<|start_global_token|>")
            for token_id in speaker_tokens_for_llm_prompt:
                prompt_parts.append(f"<|bicodec_global_{token_id}|>")
            prompt_parts.append("<|end_global_token|>")
        
        llm_prompt_string = "".join(prompt_parts)
        print(f"Constructed LLM prompt string: {llm_prompt_string}")
        llm_tokenized_output = llm_tokenizer.encode(llm_prompt_string, add_special_tokens=False)
        
        print(f"LLM Input IDs (from Python HF Tokenizer) for text '{args.text_to_synthesize}':")
        print(llm_tokenized_output)
        print(f"Number of LLM Input IDs: {len(llm_tokenized_output)}")

        if args.generate_semantic_tokens:
            print("\\\\n--- Generating Semantic Tokens using LLM ONNX ---")
            if not args.llm_onnx_path:
                print("Error: --llm_onnx_path not provided. Cannot generate semantic tokens.")
                return

            try:
                print(f"Loading LLM ONNX model from: {args.llm_onnx_path}")
                llm_ort_session = onnxruntime.InferenceSession(args.llm_onnx_path, providers=['CPUExecutionProvider'])
                print("LLM ONNX model loaded.")
            except Exception as e:
                print(f"Error loading LLM ONNX model: {e}")
                return

            num_llm_layers, num_attention_heads, head_dimension = get_llm_model_dims(llm_ort_session)
            
            # Prepare initial inputs for LLM
            initial_input_ids_np = np.array([llm_tokenized_output], dtype=np.int64) # Batch size 1
            initial_seq_len = initial_input_ids_np.shape[1]
            
            attention_mask_np = np.ones_like(initial_input_ids_np, dtype=np.int64)
            position_ids_np = np.arange(initial_seq_len, dtype=np.int64).reshape(1, initial_seq_len)

            # Initial past_key_values (empty)
            batch_size = 1
            past_kv_inputs = {}
            for i in range(num_llm_layers):
                empty_kv_shape = (batch_size, num_attention_heads, 0, head_dimension)
                empty_kv_tensor = np.empty(empty_kv_shape, dtype=np.float32)
                past_kv_inputs[f'past_key_values.{i}.key'] = empty_kv_tensor
                past_kv_inputs[f'past_key_values.{i}.value'] = empty_kv_tensor
            
            llm_inputs = {
                'input_ids': initial_input_ids_np,
                'attention_mask': attention_mask_np,
                'position_ids': position_ids_np,
                **past_kv_inputs
            }
            
            print("Running initial prompt pass through LLM ONNX...")
            try:
                llm_outputs_initial_pass = llm_ort_session.run(None, llm_inputs)
            except Exception as e:
                print(f"Error during initial LLM ONNX run: {e}")
                return

            # Output names might vary, inspect or assume standard ones.
            # Expected outputs: logits, present.0.key, present.0.value, ..., present.N-1.value
            output_names = [o.name for o in llm_ort_session.get_outputs()]
            logits_output_name = next((name for name in output_names if "logits" in name), None)
            if not logits_output_name:
                print("Error: Could not find 'logits' in LLM ONNX model output names.")
                return

            logits_idx = output_names.index(logits_output_name)
            initial_logits = llm_outputs_initial_pass[logits_idx] # Shape: (batch, seq_len, vocab_size)
            
            # Prepare past_kv for the generation loop from the initial pass's present_kv
            current_past_kv = {}
            for i in range(num_llm_layers):
                present_key_name = f'present.{i}.key'
                present_value_name = f'present.{i}.value'
                if present_key_name in output_names and present_value_name in output_names:
                    key_idx = output_names.index(present_key_name)
                    value_idx = output_names.index(present_value_name)
                    current_past_kv[f'past_key_values.{i}.key'] = llm_outputs_initial_pass[key_idx]
                    current_past_kv[f'past_key_values.{i}.value'] = llm_outputs_initial_pass[value_idx]
                else:
                    print(f"Error: Could not find present K/V for layer {i} in LLM ONNX model output names.")
                    return
            
            # Get the next token from the initial pass (logits of the last token of the prompt)
            next_token_id = np.argmax(initial_logits[0, -1, :])
            generated_semantic_tokens = [next_token_id]
            print(f"Token after initial prompt: {next_token_id}")

            # Auto-regressive generation loop
            print("Starting auto-regressive generation...")
            for step in range(args.max_new_semantic_tokens - 1): # -1 because one token already generated
                if next_token_id == args.llm_eos_token_id:
                    print(f"EOS token ({args.llm_eos_token_id}) reached at step {step}. Stopping.")
                    break

                current_input_ids_np = np.array([[next_token_id]], dtype=np.int64) # (1,1)
                
                # The past_kv_tensors now have a sequence length from the prompt.
                # The attention_mask and position_ids need to account for this.
                past_kv_seq_len = current_past_kv[f'past_key_values.0.key'].shape[2] # (B, Heads, SeqLen, Dim)
                
                # Attention mask should cover all past_kv tokens + the current token.
                current_attention_mask_len = past_kv_seq_len + 1
                current_attention_mask_np = np.ones((1, current_attention_mask_len), dtype=np.int64)
                
                # Position ID for the current token is its absolute position.
                current_position_id_val = past_kv_seq_len
                current_position_ids_np = np.array([[current_position_id_val]], dtype=np.int64) # (1,1)

                iter_llm_inputs = {
                    'input_ids': current_input_ids_np,
                    'attention_mask': current_attention_mask_np,
                    'position_ids': current_position_ids_np,
                    **current_past_kv 
                }
                
                try:
                    llm_outputs_iter = llm_ort_session.run(None, iter_llm_inputs)
                except Exception as e:
                    print(f"Error during LLM ONNX run at generation step {step}: {e}")
                    break
                
                iter_logits = llm_outputs_iter[logits_idx] # Shape (1, 1, vocab_size) for single token input
                next_token_id = np.argmax(iter_logits[0, 0, :])
                generated_semantic_tokens.append(next_token_id)
                # print(f"Step {step}: Generated token {next_token_id}") # Verbose

                # ---- DEBUG: Save logits and print Top-N at the diverging step ----
                if step == 3: # This is when the 5th token (index 4) is decided
                    print(f"DEBUG: Processing logits for step {step} (decision for token index 4, which is the 5th token)")
                    np.save("python_logits_for_5th_token.npy", iter_logits)
                    print("Saved python_logits_for_5th_token.npy")
                    
                    # Get Top-5 logits and their token IDs
                    current_logits_flat = iter_logits[0, 0, :]
                    top_n = 5
                    top_n_indices = np.argsort(current_logits_flat)[-top_n:][::-1] # Get indices of top N, then reverse for descending order
                    top_n_values = current_logits_flat[top_n_indices]
                    
                    print(f"Python Top-{top_n} Logits for 5th token decision:")
                    for i in range(top_n):
                        print(f"  Rank {i+1}: Token ID = {top_n_indices[i]}, Logit = {top_n_values[i]:.8f}")
                # ---- END DEBUG ----

                # Update past_kv for next iteration
                for i in range(num_llm_layers):
                    present_key_name = f'present.{i}.key'
                    present_value_name = f'present.{i}.value'
                    key_idx = output_names.index(present_key_name)
                    value_idx = output_names.index(present_value_name)
                    current_past_kv[f'past_key_values.{i}.key'] = llm_outputs_iter[key_idx]
                    current_past_kv[f'past_key_values.{i}.value'] = llm_outputs_iter[value_idx]

            print("\\\\n--- Generated Semantic Tokens (Python ONNX) ---")
            print(f"Number of generated semantic tokens: {len(generated_semantic_tokens)}")
            print(generated_semantic_tokens)

if __name__ == "__main__":
    main() 
import onnxruntime
import torch
import numpy as np
from pathlib import Path
import argparse
import warnings
import time

# Suppress torchaudio sox warning if needed
warnings.filterwarnings("ignore", message=".*sox_io.load_audio_file failed UI_INITIALIZE*.")
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy argument number*.")


# Attempt to import necessary pre/post-processing from SparkTTS (or replicate logic)
try:
    from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
    # from sparktts.utils.audio import load_audio # Using torchaudio directly for wider compatibility
    from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP # For special tokens
except ImportError as e:
    print(f"Error importing project-specific modules: {e}. Ensure SparkTTS is in PYTHONPATH or script is run from project root.")
    print("Some functionalities like advanced token parsing might be limited.")
    # Define placeholders if import fails, so the script can be parsed
    LEVELS_MAP = {} 
    GENDER_MAP = {}
    TASK_TOKEN_MAP = {"tts": "<|tts|>", "controllable_tts": "<|controllable_tts|>"}


import torchaudio
import torchaudio.transforms as T
import re # For parsing LLM output

class ONNXSparkTTSPredictor:
    def __init__(self, onnx_model_dir: Path, llm_tokenizer_path: Path, bicodec_config_path: Path = None, device:str = "cpu"):
        self.onnx_model_dir = onnx_model_dir
        self.sessions = {}
        self.provider = ["CPUExecutionProvider"]
        if device == "cuda" and onnxruntime.get_device() == "GPU":
            self.provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        print(f"Using ONNX Runtime provider: {self.provider}")

        self.load_models()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_tokenizer_path))
        
        # Special tokens from SparkTTS (ensure these match your model's vocabulary)
        self.START_CONTENT = "<|start_content|>"
        self.END_CONTENT = "<|end_content|>"
        self.START_SEMANTIC = "<|start_semantic_token|>"
        self.END_SEMANTIC = "<|end_semantic_token|>"
        self.START_GLOBAL = "<|start_global_token|>"
        self.END_GLOBAL = "<|end_global_token|>"
        self.SEMANTIC_TOKEN_PREFIX = "bicodec_semantic_"
        self.GLOBAL_TOKEN_PREFIX = "bicodec_global_"

        # Load BiCodec config for Mel spectrogram parameters
        self.bicodec_config = {}
        if bicodec_config_path and bicodec_config_path.exists():
            try:
                import yaml
                with open(bicodec_config_path, 'r') as f:
                    full_bicodec_config = yaml.safe_load(f)
                # Try to find audio_tokenizer, then mel_params
                if "audio_tokenizer" in full_bicodec_config:
                    self.bicodec_config = full_bicodec_config["audio_tokenizer"].get("mel_params", {})
                elif "model" in full_bicodec_config and "audio_tokenizer" in full_bicodec_config["model"]: # another common pattern
                     self.bicodec_config = full_bicodec_config["model"]["audio_tokenizer"].get("mel_params", {})
                if not self.bicodec_config:
                    print(f"Warning: 'mel_params' not found in BiCodec config {bicodec_config_path}. Using default Mel params.")
            except Exception as e:
                print(f"Warning: Could not load or parse BiCodec config {bicodec_config_path}: {e}. Using default Mel params.")

        self.sample_rate = self.bicodec_config.get("sample_rate", 16000)
        self.n_fft = self.bicodec_config.get("n_fft", 1024)
        self.win_length = self.bicodec_config.get("win_length", self.n_fft)
        self.hop_length = self.bicodec_config.get("hop_length", 256)
        self.n_mels = self.bicodec_config.get("num_mels", 100) # num_mels in config often
        self.mel_fmin = self.bicodec_config.get("mel_fmin", 0)
        self.mel_fmax = self.bicodec_config.get("mel_fmax", int(self.sample_rate / 2))


        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            power=1, # Matches SparkTTS BiCodec default
            norm="slaney", # Matches SparkTTS BiCodec default
            mel_scale="slaney" # Matches SparkTTS BiCodec default
        )
        
        # Wav2Vec2 Processor (needs to be loaded from where the ONNX model was derived)
        # Assuming wav2vec2-large-xlsr-53 is in the parent of onnx_model_dir, or at a known path
        w2v2_processor_path = self.onnx_model_dir.parent / "wav2vec2-large-xlsr-53"
        if w2v2_processor_path.exists():
            self.w2v2_processor = Wav2Vec2FeatureExtractor.from_pretrained(str(w2v2_processor_path))
        else:
            print(f"Warning: Wav2Vec2FeatureExtractor not found at {w2v2_processor_path}. Prompt audio processing might fail.")
            self.w2v2_processor = None


    def load_models(self):
        model_files = [
            "llm.onnx", "wav2vec2_feature_extractor.onnx",
            "bicodec_encoder.onnx",
            "bicodec_factorized_vq_tokenize.onnx",
            "bicodec_speaker_encoder_tokenize.onnx",
            "bicodec_factorized_vq_detokenize.onnx",
            "bicodec_speaker_encoder_detokenize.onnx",
            "bicodec_prenet.onnx", "bicodec_wavegenerator.onnx"
        ]
        for mf in model_files:
            path = self.onnx_model_dir / mf
            if path.exists():
                print(f"Loading ONNX model: {mf}")
                try:
                    self.sessions[mf.split('.')[0]] = onnxruntime.InferenceSession(str(path), providers=self.provider)
                except Exception as e:
                    print(f"Error loading ONNX model {mf}: {e}")
                    print(f"Ensure your ONNX Runtime version is compatible with the model opset and CUDA (if using GPU).")
            else:
                print(f"Warning: ONNX model {mf} not found at {path}. Inference might fail.")

    def _run_onnx(self, model_key, inputs_dict):
        if model_key not in self.sessions:
            raise ValueError(f"ONNX session for {model_key} not loaded or failed to load.")
        return self.sessions[model_key].run(None, inputs_dict)

    def _load_audio_np(self, audio_path: str, target_sr: int):
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.ndim > 1: # Get first channel if stereo
                wav = wav[0, :].unsqueeze(0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.squeeze().numpy() # Return as numpy array
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None


    def get_prompt_tokens_onnx(self, prompt_speech_path: str):
        wav_np = self._load_audio_np(prompt_speech_path, self.sample_rate)
        if wav_np is None: return None, None
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)

        mel_features = self.mel_spectrogram_transform(wav_tensor)
        
        global_tokens_indices_onnx = self._run_onnx("bicodec_speaker_encoder_tokenize", {"mels": mel_features.numpy()})[0]
        
        if not self.w2v2_processor:
            print("Error: Wav2Vec2Processor not available for prompt feature extraction.")
            return None, None
        
        w2v2_input_values = self.w2v2_processor(wav_np, sampling_rate=self.sample_rate, return_tensors="np").input_values
        
        w2v2_hidden_states_tuple = self._run_onnx("wav2vec2_feature_extractor", {"input_values": w2v2_input_values})
        combined_feat = (w2v2_hidden_states_tuple[0] + w2v2_hidden_states_tuple[1] + w2v2_hidden_states_tuple[2]) / 3.0
        # combined_feat is (B, D, T_feat), bicodec_encoder expects (B, D, T_feat)

        encoded_z = self._run_onnx("bicodec_encoder", {"feat_input": combined_feat})[0]
        semantic_tokens_prompt = self._run_onnx("bicodec_factorized_vq_tokenize", {"encoded_z": encoded_z})[0]

        return torch.from_numpy(global_tokens_indices_onnx), torch.from_numpy(semantic_tokens_prompt)


    def generate_llm_output_onnx(self, text_input_ids_np, attention_mask_np, 
                                 max_new_tokens=1000, temperature=0.7, top_k=0, top_p=0.9, do_sample=True,
                                 eos_token_id=None):
        if eos_token_id is None:
            eos_token_id = self.llm_tokenizer.eos_token_id
            if eos_token_id is None: # Fallback if not defined in tokenizer
                 # A common EOS token for many models is the ID for "<|endoftext|>" or similar.
                 # This needs to be set correctly for the specific LLM.
                 # Using a high number as a placeholder if completely unknown, but this is risky.
                warnings.warn("EOS token ID not found in tokenizer, generation might not stop correctly.")
                # Attempt to get from special tokens map or use a known one if possible
                eos_token_id = self.llm_tokenizer.convert_tokens_to_ids(self.END_SEMANTIC) # Example
                if not isinstance(eos_token_id, int): eos_token_id = 50256 # fallback to a common one

        print(f"LLM Generation: Max new tokens: {max_new_tokens}, EOS token ID: {eos_token_id}")

        current_input_ids_np = text_input_ids_np.copy()
        current_attention_mask_np = attention_mask_np.copy()
        
        generated_token_ids_list = []

        for i in range(max_new_tokens):
            onnx_inputs = {
                "input_ids": current_input_ids_np,
                "attention_mask": current_attention_mask_np
            }
            logits_onnx = self._run_onnx("llm", onnx_inputs)[0] # (B, Seq, Vocab)
            
            next_token_logits_torch = torch.from_numpy(logits_onnx[:, -1, :])

            if do_sample:
                if temperature != 1.0:
                    next_token_logits_torch = next_token_logits_torch / temperature
                
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits_torch, top_k)
                    min_val_top_k = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits_torch[next_token_logits_torch < min_val_top_k] = -float('Inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits_torch, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits_torch[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(next_token_logits_torch, dim=-1)
                next_token_id_torch = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id_torch = torch.argmax(next_token_logits_torch, dim=-1, keepdim=True)

            next_token_id_item = next_token_id_torch.item()
            generated_token_ids_list.append(next_token_id_item)
            
            if next_token_id_item == eos_token_id:
                print(f"EOS token {eos_token_id} reached at step {i+1}. Stopping generation.")
                break

            current_input_ids_np = np.concatenate([current_input_ids_np, next_token_id_torch.numpy()], axis=1)
            current_attention_mask_np = np.concatenate([current_attention_mask_np, np.ones_like(next_token_id_torch.numpy())], axis=1)
        else: # Loop finished without break
            print(f"Max new tokens ({max_new_tokens}) reached.")


        return np.array(generated_token_ids_list, dtype=np.int64).reshape(1, -1)


    def parse_llm_tokens(self, llm_output_str: str, prefix: str, end_token: str):
        tokens = []
        # Simplified parsing, assumes format like "<prefix>_DDD<prefix>_DDD...<end_token>"
        # This needs to be robust. Example: re.findall(r"bicodec_semantic_(\d+)", llm_output_str)
        pattern = re.compile(f"{re.escape(prefix)}(\d+)")
        for match in pattern.finditer(llm_output_str):
            tokens.append(int(match.group(1)))
        # Check if end_token is present to consider parsing complete
        if end_token not in llm_output_str and len(tokens)>0 :
            print(f"Warning: End token '{end_token}' not found in LLM output substring used for parsing. Parsed {len(tokens)} tokens.")

        return np.array(tokens, dtype=np.int64).reshape(1, -1) if tokens else np.array([], dtype=np.int64).reshape(1,0)


    def detokenize_to_audio_onnx(self, llm_generated_ids_str: str, global_tokens_from_prompt_onnx: np.ndarray):
        # 1. Parse semantic and global tokens from LLM output string
        pred_semantic_ids_np = self.parse_llm_tokens(llm_generated_ids_str, self.SEMANTIC_TOKEN_PREFIX, self.END_SEMANTIC)
        
        if pred_semantic_ids_np.size == 0:
            print("Error: No semantic tokens parsed from LLM output. Cannot generate audio.")
            return None

        # Global tokens: if voice cloning, use from prompt. If controllable, parse from LLM.
        # For now, this function assumes voice cloning, global_tokens_from_prompt_onnx is primary.
        # If global_tokens_from_prompt_onnx is None (e.g. controllable mode starting from scratch), then try parsing.
        
        final_global_tokens_indices_np = None
        if global_tokens_from_prompt_onnx is not None:
            final_global_tokens_indices_np = global_tokens_from_prompt_onnx
            # Expected shape for speaker_encoder_detokenize: (B, G_spk, T_spk_token)
        else: # Controllable TTS: parse global tokens from LLM string
            print("Attempting to parse global tokens from LLM output for controllable TTS.")
            parsed_global_tokens_np = self.parse_llm_tokens(llm_generated_ids_str, self.GLOBAL_TOKEN_PREFIX, self.END_GLOBAL)
            if parsed_global_tokens_np.size == 0:
                 print("Error: No global tokens provided by prompt or parsed from LLM. Cannot generate audio.")
                 return None
            # The parsed global tokens are likely (B, T_flat_global). Reshape for ResidualFSQ if needed.
            # This depends on how SpeakerEncoder.detokenize (and its ONNX wrapper) expects indices.
            # Assuming SpeakerEncoder.detokenize ONNX wrapper takes (B, G, T_spk_token)
            # This step is complex and model-dependent. Placeholder:
            # TODO: Reshape parsed_global_tokens_np to match speaker_encoder_detokenize input
            # For now, if controllable, it might error out if shape is not (B,G,T)
            final_global_tokens_indices_np = parsed_global_tokens_np 
            # This will likely fail if not (B,G,T), needs config for G and T_spk_token to reshape.
            print(f"Warning: Using directly parsed global tokens for controllable TTS. Shape might be incorrect: {final_global_tokens_indices_np.shape}")


        if final_global_tokens_indices_np is None:
            print("Error: Final global token indices are unavailable.")
            return None

        # 2. Detokenize semantic tokens
        z_q_onnx = self._run_onnx("bicodec_factorized_vq_detokenize", {"semantic_tokens": pred_semantic_ids_np})[0]

        # 3. Detokenize global tokens
        d_vector_onnx = self._run_onnx("bicodec_speaker_encoder_detokenize", {"global_tokens_indices": final_global_tokens_indices_np})[0]
        
        # 4. Run Prenet
        prenet_output_x_onnx = self._run_onnx("bicodec_prenet", {"z_q": z_q_onnx, "d_vector_condition": d_vector_onnx})[0]

        # 5. Run WaveGenerator
        wavegen_input_onnx = prenet_output_x_onnx + d_vector_onnx[:, :, np.newaxis]
        wav_recon_onnx = self._run_onnx("bicodec_wavegenerator", {"wavegen_input": wavegen_input_onnx})[0]

        return wav_recon_onnx.squeeze()


    def inference(self, text: str, 
                  prompt_speech_path: str = None, prompt_text: str = None,
                  gender: str = None, pitch: str = None, speed: str = None, # For controllable TTS
                  temperature: float = 0.7, top_p: float = 0.9, max_new_tokens_llm: int = 1000
                  ):
        
        llm_input_parts = []
        global_tokens_for_detok_onnx = None # This will hold the (B,G,T) indices for speaker_encoder_detokenize

        if prompt_speech_path: # Voice cloning
            print("Mode: Voice Cloning")
            llm_input_parts.append(TASK_TOKEN_MAP["tts"])
            llm_input_parts.append(self.START_CONTENT)
            if prompt_text:
                llm_input_parts.append(prompt_text)
            llm_input_parts.append(text)
            llm_input_parts.append(self.END_CONTENT)

            # Get global tokens from prompt audio via ONNX
            # global_tokens_indices_prompt_onnx is (B, G_spk, T_spk_token)
            # semantic_tokens_indices_prompt_onnx is (B, T_sem_prompt)
            global_tokens_indices_prompt_onnx, semantic_tokens_indices_prompt_onnx = self.get_prompt_tokens_onnx(prompt_speech_path)

            if global_tokens_indices_prompt_onnx is None or semantic_tokens_indices_prompt_onnx is None:
                print("Failed to get tokens from prompt audio.")
                return None
            
            global_tokens_for_detok_onnx = global_tokens_indices_prompt_onnx.numpy() # Keep as numpy for ONNX run

            # Format tokens for LLM input string (as SparkTTS does)
            # This requires knowing how SparkTTS converts these indices back to <|bicodec_...|> strings
            # Placeholder string formatting:
            llm_input_parts.append(self.START_GLOBAL)
            for g_idx in range(global_tokens_indices_prompt_onnx.shape[1]): # Iterate over groups
                 for t_idx in range(global_tokens_indices_prompt_onnx.shape[2]): # Iterate over tokens in group
                    llm_input_parts.append(f"{self.GLOBAL_TOKEN_PREFIX}{global_tokens_indices_prompt_onnx[0, g_idx, t_idx].item()}")
            llm_input_parts.append(self.END_GLOBAL)
            
            llm_input_parts.append(self.START_SEMANTIC)
            for s_idx in range(semantic_tokens_indices_prompt_onnx.shape[1]):
                llm_input_parts.append(f"{self.SEMANTIC_TOKEN_PREFIX}{semantic_tokens_indices_prompt_onnx[0, s_idx].item()}")
            # LLM should generate from here
            # No END_SEMANTIC here, LLM generates more semantic tokens
            
        else: # Controllable TTS
            print("Mode: Controllable TTS")
            if not all([gender, pitch, speed]):
                raise ValueError("Gender, pitch, and speed must be provided for controllable TTS if no prompt audio.")
            
            llm_input_parts.append(TASK_TOKEN_MAP["controllable_tts"])
            llm_input_parts.append(self.START_CONTENT)
            llm_input_parts.append(text)
            llm_input_parts.append(self.END_CONTENT)
            
            # Add style tokens
            llm_input_parts.append("<|start_style_label|>")
            llm_input_parts.append(f"<|gender_{GENDER_MAP.get(gender, -1)}|>")
            llm_input_parts.append(f"<|pitch_label_{LEVELS_MAP.get(pitch, -1)}|>")
            llm_input_parts.append(f"<|speed_label_{LEVELS_MAP.get(speed, -1)}|>")
            llm_input_parts.append("<|end_style_label|>")
            # LLM should generate both global and semantic tokens in this mode
            # global_tokens_for_detok_onnx will be parsed from LLM output later.
            
        # Common for both: LLM needs to start generating semantic tokens
        llm_input_parts.append(self.START_SEMANTIC) # Prompt LLM to start generating semantic tokens

        llm_input_str = "".join(llm_input_parts)
        print(f"LLM Input String (Prefix): {llm_input_str[:500]} ...")
        
        tokenized_llm_input = self.llm_tokenizer(llm_input_str, return_tensors="np", padding=True, truncation=True, max_length=1024) # Max length for prefix
        input_ids_np = tokenized_llm_input.input_ids
        attention_mask_np = tokenized_llm_input.attention_mask

        # Generate with LLM ONNX (this generates the token IDs, not the string with <|...|>)
        # It should generate the sequence of numbers for semantic tokens, and for global if controllable mode.
        # And also the EOS token for semantic part.
        llm_generated_ids_numbers = self.generate_llm_output_onnx(
            input_ids_np, attention_mask_np, 
            max_new_tokens=max_new_tokens_llm, 
            temperature=temperature, top_p=top_p,
            eos_token_id=self.llm_tokenizer.convert_tokens_to_ids(self.END_SEMANTIC) # LLM should stop at end of semantic
        )
        
        # Convert these numbers back to the <|bicodec_semantic_XXX|> <|bicodec_global_XXX|> string format
        # This is what detokenize_to_audio_onnx expects for parsing.
        # This step is crucial: the generated IDs are raw numbers.
        generated_tokens_str_parts = []
        # Assuming LLM generates semantic tokens first, then global if controllable mode (needs verification based on SparkTTS training)
        for token_id in llm_generated_ids_numbers.squeeze():
            # This reconstruction assumes LLM directly outputs indices that correspond to bicodec tokens.
            # It's more likely LLM outputs its own vocabulary, which then contains these special bicodec tokens.
            # The current generate_llm_output_onnx gives raw token IDs. We need to decode them first using llm_tokenizer.
            # Then parse bicodec tokens from that decoded string.
            pass # This needs to be re-thought.

        # --- Re-thinking LLM output processing ---
        # 1. Get raw token IDs from LLM ONNX
        # 2. Decode these raw token IDs to a string using self.llm_tokenizer.batch_decode()
        # 3. Then, parse the <|bicodec_semantic_XXX|> and <|bicodec_global_XXX|> from this decoded string.
        
        llm_output_decoded_string = self.llm_tokenizer.decode(llm_generated_ids_numbers.squeeze(), skip_special_tokens=False)
        print(f"LLM Output Decoded String (Raw): {llm_output_decoded_string[:500]} ...")
        
        # Now pass this string to detokenize_to_audio_onnx
        # global_tokens_for_detok_onnx is already set if voice cloning.
        # If controllable TTS, global_tokens_for_detok_onnx is None, so detokenize_to_audio_onnx will try to parse them.
        waveform = self.detokenize_to_audio_onnx(llm_output_decoded_string, global_tokens_for_detok_onnx)
        
        return waveform

# main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Spark-TTS inference using ONNX models.")
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize.")
    parser.add_argument("--onnx_model_dir", required=True, type=str, help="Directory containing all ONNX model files.")
    parser.add_argument("--llm_tokenizer_dir", required=True, type=str, help="Directory of the HuggingFace LLM tokenizer (part of original model/LLM).")
    parser.add_argument("--bicodec_config", type=str, default=None, help="Path to BiCodec's config.yaml (for Mel parameters). Usually ONNX_MODEL_DIR/../BiCodec/config.yaml")
    parser.add_argument("--prompt_audio", type=str, default=None, help="Path to prompt audio file for voice cloning.")
    parser.add_argument("--prompt_text", type=str, default=None, help="Transcript of the prompt audio.")
    parser.add_argument("--gender", type=str, choices=GENDER_MAP.keys(), default=None, help="Gender for controllable TTS.")
    parser.add_argument("--pitch", type=str, choices=LEVELS_MAP.keys(), default=None, help="Pitch for controllable TTS.")
    parser.add_argument("--speed", type=str, choices=LEVELS_MAP.keys(), default=None, help="Speed for controllable TTS.")
    parser.add_argument("--output_wav", type=str, default="output_onnx.wav", help="Path to save the generated waveform.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p for LLM.")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens for LLM generation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for ONNX Runtime (cpu or cuda).")


    args = parser.parse_args()

    bicodec_cfg_path = Path(args.bicodec_config) if args.bicodec_config else Path(args.onnx_model_dir).parent / "BiCodec" / "config.yaml"

    predictor = ONNXSparkTTSPredictor(Path(args.onnx_model_dir), Path(args.llm_tokenizer_dir), bicodec_config_path=bicodec_cfg_path, device=args.device)
    
    print("\nStarting ONNX inference...")
    waveform_onnx = None
    if args.prompt_audio:
        waveform_onnx = predictor.inference(
            args.text, 
            prompt_speech_path=args.prompt_audio, 
            prompt_text=args.prompt_text,
            temperature=args.temperature, top_p=args.top_p, max_new_tokens_llm=args.max_new_tokens
        )
    elif args.gender and args.pitch and args.speed:
         waveform_onnx = predictor.inference(
            args.text,
            gender=args.gender, pitch=args.pitch, speed=args.speed,
            temperature=args.temperature, top_p=args.top_p, max_new_tokens_llm=args.max_new_tokens
        )
    else:
        print("Error: Either prompt_audio (for voice cloning) or (gender, pitch, speed) (for controllable TTS) must be provided.")
        exit(1)


    if waveform_onnx is not None and waveform_onnx.size > 0 :
        # Ensure waveform is 1D or 2D (for torchaudio.save)
        if waveform_onnx.ndim == 1:
            waveform_onnx_torch = torch.from_numpy(waveform_onnx).unsqueeze(0)
        elif waveform_onnx.ndim == 2:
             waveform_onnx_torch = torch.from_numpy(waveform_onnx)
        else:
            print(f"Error: Generated waveform has unexpected ndim: {waveform_onnx.ndim}")
            waveform_onnx_torch = None

        if waveform_onnx_torch is not None:
            torchaudio.save(args.output_wav, waveform_onnx_torch, predictor.sample_rate)
            print(f"Generated audio saved to {args.output_wav}")
    else:
        print("ONNX inference failed to produce a valid waveform.") 
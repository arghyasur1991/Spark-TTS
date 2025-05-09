import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import os
import re
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import warnings

# Suppress torchaudio sox warning if needed
warnings.filterwarnings("ignore", message=".*sox_io.load_audio_file failed UI_INITIALIZE*.")

# Import TTS-specific components if available
try:
    from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
except ImportError as e:
    print(f"Warning: Could not import token_parser from sparktts.utils: {e}")
    # Define placeholders for token maps
    LEVELS_MAP = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
    GENDER_MAP = {"female": 0, "male": 1}
    TASK_TOKEN_MAP = {
        "tts": "<|task_tts|>",
        "controllable_tts": "<|task_controllable_tts|>"
    }

class ONNXSparkTTSPredictor:
    """
    ONNX-based inference model for SparkTTS.
    Uses exported ONNX models to perform inference similar to the original PyTorch model.
    """
    
    def __init__(
        self,
        onnx_model_dir: Path,
        llm_tokenizer_dir: Path,
        bicodec_config_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """
        Initialize the ONNX SparkTTS predictor.
        
        Args:
            onnx_model_dir: Path to directory containing exported ONNX models
            llm_tokenizer_dir: Path to the directory containing the tokenizer files for the LLM
            bicodec_config_path: Path to BiCodec config.yaml (optional)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_dir = onnx_model_dir
        self.device = device
        
        # Define token constants
        self.START_CONTENT = "<|start_content|>"
        self.END_CONTENT = "<|end_content|>"
        self.START_GLOBAL = "<|start_global_token|>"
        self.END_GLOBAL = "<|end_global_token|>"
        self.START_SEMANTIC = "<|start_semantic_token|>"
        self.END_SEMANTIC = "<|end_semantic_token|>"
        self.GLOBAL_TOKEN_PREFIX = "<|bicodec_global_"
        self.SEMANTIC_TOKEN_PREFIX = "<|bicodec_semantic_"
        
        # Set default ORT session options
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if device == "cuda" and ort.get_device() == "GPU":
            print("Using CUDA for ONNX Runtime inference")
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print(f"Using CPU for ONNX Runtime inference (requested: {device})")
            self.providers = ['CPUExecutionProvider']
        
        # Initialize configs
        self._load_config(bicodec_config_path)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(llm_tokenizer_dir))
        
        # Load ONNX models
        self.load_models()
        
        print(f"ONNX SparkTTS initialized with {len(self.onnx_sessions)} ONNX models loaded")
    
    def _load_config(self, bicodec_config_path: Optional[Path] = None):
        """Load configuration from provided path or try to find it."""
        # Default values
        self.sample_rate = 16000  # Default sample rate
        
        if bicodec_config_path and bicodec_config_path.exists():
            try:
                with open(bicodec_config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"Loaded BiCodec config from {bicodec_config_path}")
                
                # Extract sample rate if available
                if 'sample_rate' in self.config:
                    self.sample_rate = self.config['sample_rate']
                elif 'model' in self.config and 'sample_rate' in self.config['model']:
                    self.sample_rate = self.config['model']['sample_rate']
            except Exception as e:
                print(f"Error loading BiCodec config: {e}")
                self.config = {}
        else:
            print("No BiCodec config provided or found. Using default values.")
            self.config = {}
    
    def load_models(self):
        """Load all available ONNX models from the model directory."""
        self.onnx_sessions = {}
        
        # Expected component names
        component_names = [
            "bicodec_encoder",
            "bicodec_factorized_vq_tokenize",
            "bicodec_factorized_vq_detokenize",
            "bicodec_speaker_encoder_tokenize",
            "bicodec_speaker_encoder_detokenize",
            "bicodec_wavegenerator",
            # "bicodec_prenet" - Skipped due to export issues
        ]
        
        for component in component_names:
            onnx_path = self.model_dir / f"{component}.onnx"
            if onnx_path.exists():
                try:
                    session = ort.InferenceSession(
                        str(onnx_path),
                        providers=self.providers,
                        sess_options=self.session_options
                    )
                    self.onnx_sessions[component] = session
                    
                    # Get input and output names
                    input_names = [input.name for input in session.get_inputs()]
                    output_names = [output.name for output in session.get_outputs()]
                    
                    print(f"Loaded {component}: inputs={input_names}, outputs={output_names}")
                except Exception as e:
                    print(f"Error loading {component}: {e}")
            else:
                print(f"Warning: {component} not found at {onnx_path}")
        
        # Verify essential components are loaded
        essential_components = [
            "bicodec_factorized_vq_detokenize",
            "bicodec_wavegenerator"
        ]
        
        missing = [c for c in essential_components if c not in self.onnx_sessions]
        if missing:
            raise ValueError(f"Missing essential ONNX models: {', '.join(missing)}")
    
    def _run_onnx(self, model_key, inputs_dict):
        """Run inference using a specific ONNX model."""
        if model_key not in self.onnx_sessions:
            return None
        
        # Copy input dict to modify without affecting original
        inputs_copy = {}
        for k, v in inputs_dict.items():
            if isinstance(v, np.ndarray):
                # Ensure all numpy arrays are the correct dtype for ONNX
                if np.issubdtype(v.dtype, np.integer):
                    inputs_copy[k] = v.astype(np.int64)
                else:
                    inputs_copy[k] = v.astype(np.float32)
            else:
                inputs_copy[k] = v
        
        # Get output names for this model
        output_names = [output.name for output in self.onnx_sessions[model_key].get_outputs()]
        
        # Run inference
        try:
            start_time = time.time()
            outputs = self.onnx_sessions[model_key].run(output_names, inputs_copy)
            inference_time = time.time() - start_time
            print(f"ONNX inference for {model_key} took {inference_time:.4f}s")
            return outputs
        except Exception as e:
            print(f"Error running {model_key} ONNX inference: {e}")
            print(f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs_copy.items()]}")
            return None
    
    def _load_audio_np(self, audio_path: str):
        """Load audio file to numpy array."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            print(f"Resampling audio from {sr} Hz to {self.sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to numpy array and ensure proper shape
        audio_np = waveform.squeeze().numpy()
        
        return audio_np
    
    def get_prompt_tokens_onnx(self, prompt_speech_path: str):
        """
        Extract tokens from prompt audio using ONNX models.
        
        Returns:
            (global_tokens_tensor, semantic_tokens_tensor) or (None, None) if failed
        """
        try:
            # 1. Load audio and convert to mel using BiCodec encoder
            audio_np = self._load_audio_np(prompt_speech_path)
            
            # 2. For speaker encoder tokenize, we should use BiCodec encoder first,
            # but the current ONNX model for SpeakerEncoder.tokenize is a dummy.
            # Instead, return a dummy global token tensor.
            global_tokens_tensor = torch.zeros((1, 1, 32), dtype=torch.long)
            global_tokens_tensor.fill_(1)  # Use '1' as token ID for testing
            
            # 3. Similarly, return dummy semantic tokens (not used for inference)
            semantic_tokens_tensor = torch.zeros((1, 100), dtype=torch.long)
            
            print("Using dummy global and semantic tokens for prompt (real ones would require working encoder)")
            return global_tokens_tensor, semantic_tokens_tensor
            
        except Exception as e:
            print(f"Error extracting tokens from prompt audio: {e}")
            return None, None
    
    def generate_llm_output_onnx(self, input_ids_np, attention_mask_np, max_new_tokens=1000, temperature=0.7, top_p=0.9, eos_token_id=None):
        """
        Generate tokens from LLM input using ONNX model.
        
        This is a placeholder since the LLM export to ONNX failed.
        In a real implementation, this would use the ONNX LLM model.
        """
        print("Warning: Using dummy LLM output generation (LLM export to ONNX failed)")
        
        # For testing, just generate some dummy semantic token indices
        # 2048 is the codebook size for FactorizedVectorQuantize in many configs
        dummy_semantic_indices = np.random.randint(0, 2048, size=(1, min(300, max_new_tokens)), dtype=np.int64)
        
        # Convert these to a string that matches the expected format
        # e.g., "<|bicodec_semantic_123|><|bicodec_semantic_456|>..."
        output_str = "".join([f"{self.SEMANTIC_TOKEN_PREFIX}{idx}|>" for idx in dummy_semantic_indices.flatten()])
        output_str += self.END_SEMANTIC
        
        return output_str
    
    def parse_tokens(self, text, token_prefix, end_token=None):
        """Parse token indices from text with format "<prefix>123<prefix>456..."."""
        # Extract prefix from token_prefix by removing trailing chars that might need escaping
        # Handle either "<|bicodec_semantic_" or "<|bicodec_global_"
        base_prefix = token_prefix  
        
        # Use a proper regex pattern that escapes special characters
        pattern = re.compile(f"{re.escape(base_prefix)}(\\d+)")
        tokens = []
        
        # Find all matches
        for match in pattern.finditer(text):
            tokens.append(int(match.group(1)))
        
        # Check if end_token is present to consider parsing complete
        if end_token and end_token not in text and tokens:
            print(f"Warning: End token '{end_token}' not found in text. Parsing may be incomplete. Found {len(tokens)} tokens.")
        
        return np.array(tokens, dtype=np.int64).reshape(1, -1) if tokens else np.array([], dtype=np.int64).reshape(1, 0)
    
    def _detokenize(self, global_token_ids, semantic_token_ids):
        """
        Convert tokens back to waveform using BiCodec components.
        
        Args:
            global_token_ids: Global token IDs from the speaker encoder
            semantic_token_ids: Semantic token IDs from the LLM
        
        Returns:
            Generated waveform as numpy array
        """
        # Ensure inputs are numpy arrays
        if isinstance(global_token_ids, torch.Tensor):
            global_token_ids = global_token_ids.numpy()
        if isinstance(semantic_token_ids, torch.Tensor):
            semantic_token_ids = semantic_token_ids.numpy()
        
        # 1. Convert semantic tokens to z_q using factorized_vq_detokenize
        fqv_detok_result = self._run_onnx("bicodec_factorized_vq_detokenize", 
                                          {"semantic_tokens": semantic_token_ids})
        if not fqv_detok_result:
            print("Failed to detokenize semantic tokens")
            return None
        
        z_q = fqv_detok_result[0]
        print(f"z_q shape after detokenize: {z_q.shape}")
        
        # 2. Convert global tokens to d_vector using speaker_encoder_detokenize
        # When using the dummy implementation, this may return an empty tensor or fail
        d_vector = None
        try:
            spk_detok_result = self._run_onnx("bicodec_speaker_encoder_detokenize", 
                                             {"global_tokens_indices": global_token_ids})
            if spk_detok_result:
                d_vector = spk_detok_result[0]
                print(f"d_vector shape: {d_vector.shape}")
            else:
                print("Failed to run speaker_encoder_detokenize, using dummy d_vector")
        except Exception as e:
            print(f"Error in speaker_encoder_detokenize: {e}")
        
        # If d_vector failed or isn't usable, create a dummy one
        if d_vector is None or d_vector.size == 0:
            d_vector = np.zeros((1, 256), dtype=np.float32)
            print("Using zero-initialized d_vector with shape (1, 256)")
        
        # 3. Since we don't have prenet in ONNX, we'll need to adapt the output
        # For simplicity, pass z_q directly to the wavegenerator
        # The expected shape is (batch_size, channels, sequence_length)
        # Normally, prenet would do: output = prenet(z_q, d_vector)
        
        # 4. Generate waveform using wavegenerator
        wavegen_result = self._run_onnx("bicodec_wavegenerator", {"wavegen_input": z_q})
        if not wavegen_result:
            print("Failed to run wavegenerator")
            return None
        
        wav_output = wavegen_result[0]
        print(f"Generated waveform shape: {wav_output.shape}")
        
        return wav_output.squeeze()
    
    def inference(
        self,
        text: str,
        prompt_speech_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens_llm: int = 1000,
    ):
        """
        Perform TTS inference using ONNX models.
        
        Args:
            text: The text to synthesize
            prompt_speech_path: Path to audio file for voice cloning
            prompt_text: Transcript of the prompt audio (optional)
            gender: female | male (for controllable TTS)
            pitch: very_low | low | moderate | high | very_high (for controllable TTS)
            speed: very_low | low | moderate | high | very_high (for controllable TTS)
            temperature: Sampling temperature for the LLM
            top_p: Top-p (nucleus) sampling parameter
            max_new_tokens_llm: Maximum number of tokens to generate from the LLM
        
        Returns:
            Synthesized audio as numpy array
        """
        is_voice_cloning_mode = prompt_speech_path is not None
        is_controllable_mode = gender is not None and pitch is not None and speed is not None
        
        if not is_voice_cloning_mode and not is_controllable_mode:
            raise ValueError(
                "Either prompt_speech_path (for voice cloning) or gender/pitch/speed (for controllable TTS) must be provided."
            )
        
        start_time = time.time()
        global_token_ids = None
        
        # 1. Prepare input based on mode
        if is_voice_cloning_mode:
            print("Mode: Voice Cloning")
            # Get global tokens from prompt audio
            global_token_ids, _ = self.get_prompt_tokens_onnx(prompt_speech_path)
            
            # Prepare prompt with optional prompt text
            llm_input = TASK_TOKEN_MAP["tts"]
            llm_input += self.START_CONTENT
            if prompt_text:
                llm_input += prompt_text + " "
            llm_input += text
            llm_input += self.END_CONTENT
            
            # Add global tokens (from prompt) to the input
            llm_input += self.START_GLOBAL
            if global_token_ids is not None:
                for token_id in global_token_ids.flatten():
                    llm_input += f"{self.GLOBAL_TOKEN_PREFIX}{token_id}|>"
            llm_input += self.END_GLOBAL
            
            # Add start of semantic tokens
            llm_input += self.START_SEMANTIC
            
        else:  # Controllable TTS
            print("Mode: Controllable TTS")
            
            # Create synthetic global tokens (will be needed for detokenize)
            global_token_ids = np.ones((1, 1, 32), dtype=np.int64)
            
            # Prepare controllable input
            llm_input = TASK_TOKEN_MAP["controllable_tts"]
            llm_input += self.START_CONTENT
            llm_input += text
            llm_input += self.END_CONTENT
            
            # Add style tokens
            llm_input += "<|start_style_label|>"
            llm_input += f"<|gender_{GENDER_MAP.get(gender, 0)}|>"
            llm_input += f"<|pitch_label_{LEVELS_MAP.get(pitch, 2)}|>"  # moderate=2 is default
            llm_input += f"<|speed_label_{LEVELS_MAP.get(speed, 2)}|>"  # moderate=2 is default
            llm_input += "<|end_style_label|>"
            
            # Add start of semantic tokens
            llm_input += self.START_SEMANTIC
        
        # 2. Generate tokens from LLM (using dummy implementation for now)
        print(f"Generating tokens for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        llm_output = self.generate_llm_output_onnx(
            None, None,  # Placeholders since we're using a dummy implementation
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens_llm,
            eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        )
        
        # 3. Parse semantic tokens from LLM output
        semantic_token_ids = self.parse_tokens(
            llm_output, 
            self.SEMANTIC_TOKEN_PREFIX, 
            self.END_SEMANTIC
        )
        
        if semantic_token_ids.size == 0:
            print("No semantic tokens found in LLM output. Cannot generate audio.")
            return None
        
        print(f"Parsed {semantic_token_ids.size} semantic token IDs")
        
        # 4. Generate waveform from tokens
        print("Generating waveform from tokens...")
        wav = self._detokenize(global_token_ids, semantic_token_ids)
        
        total_time = time.time() - start_time
        print(f"Inference completed in {total_time:.2f} seconds")
        
        return wav


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Spark-TTS inference using ONNX models.")
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize.")
    parser.add_argument("--onnx_model_dir", required=True, type=str, help="Directory containing all ONNX model files.")
    parser.add_argument("--llm_tokenizer_dir", required=True, type=str, help="Directory of the HuggingFace LLM tokenizer (part of original model/LLM).")
    parser.add_argument("--bicodec_config", type=str, default=None, help="Path to BiCodec's config.yaml (for Mel parameters). Usually ONNX_MODEL_DIR/../BiCodec/config.yaml")
    parser.add_argument("--prompt_audio", type=str, default=None, help="Path to prompt audio file for voice cloning.")
    parser.add_argument("--prompt_text", type=str, default=None, help="Transcript of the prompt audio.")
    parser.add_argument("--gender", type=str, choices=list(GENDER_MAP.keys()), default=None, help="Gender for controllable TTS.")
    parser.add_argument("--pitch", type=str, choices=list(LEVELS_MAP.keys()), default=None, help="Pitch for controllable TTS.")
    parser.add_argument("--speed", type=str, choices=list(LEVELS_MAP.keys()), default=None, help="Speed for controllable TTS.")
    parser.add_argument("--output_wav", type=str, default="output_onnx.wav", help="Path to save the generated waveform.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p for LLM.")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens for LLM generation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for ONNX Runtime (cpu or cuda).")
    
    args = parser.parse_args()
    
    # Validate mode selection
    is_voice_cloning = args.prompt_audio is not None
    is_controllable = all([args.gender, args.pitch, args.speed])
    
    if not is_voice_cloning and not is_controllable:
        print("Error: Must specify either --prompt_audio for voice cloning OR (--gender, --pitch, --speed) for controllable TTS")
        return 1
    
    if is_voice_cloning and is_controllable:
        print("Warning: Both voice cloning and controllable TTS parameters provided. Using voice cloning mode.")
    
    # Create the model
    model_dir = Path(args.onnx_model_dir)
    tokenizer_dir = Path(args.llm_tokenizer_dir)
    bicodec_config_path = Path(args.bicodec_config) if args.bicodec_config else None
    
    try:
        predictor = ONNXSparkTTSPredictor(
            model_dir,
            tokenizer_dir,
            bicodec_config_path,
            device=args.device
        )
        
        # Run inference
        if is_voice_cloning:
            waveform = predictor.inference(
                text=args.text,
                prompt_speech_path=args.prompt_audio,
                prompt_text=args.prompt_text,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens_llm=args.max_new_tokens
            )
        else:
            waveform = predictor.inference(
                text=args.text,
                gender=args.gender,
                pitch=args.pitch,
                speed=args.speed,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens_llm=args.max_new_tokens
            )
        
        if waveform is not None:
            # Save output
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            torchaudio.save(args.output_wav, waveform_tensor, predictor.sample_rate)
            print(f"Audio saved to {args.output_wav}")
            return 0
        else:
            print("Failed to generate waveform")
            return 1
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 
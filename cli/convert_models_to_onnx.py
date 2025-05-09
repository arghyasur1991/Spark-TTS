import torch
import torch.nn as nn
import os
from pathlib import Path
import warnings
import yaml # For loading model configurations

# Suppress specific warnings during ONNX export if necessary
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to import necessary project modules
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from sparktts.utils.file import load_config as sparktts_load_config # Alias to avoid confusion
    from sparktts.models.bicodec import BiCodec
    from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
    from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
    from sparktts.modules.fsq.residual_fsq import ResidualFSQ # For type hinting/checking
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from a context where Spark-TTS modules are accessible (e.g., project root or with PYTHONPATH set).")
    exit(1)


# --- Configuration ---
DEFAULT_MODEL_DIR_NAME = "Spark-TTS-0.5B"
DEFAULT_ONNX_OUTPUT_DIR = "onnx_models"
DEFAULT_OPSET_VERSION = 14 # Common opset, can be overridden

# --- Helper function for ONNX export ---
def export_model_to_onnx(model, dummy_inputs, input_names, output_names, dynamic_axes, onnx_path, opset_version):
    """Exports a PyTorch model to ONNX."""
    model.eval()
    if not isinstance(dummy_inputs, tuple):
        dummy_inputs = (dummy_inputs,)

    device = torch.device("cpu")
    model.to(device)
    dummy_inputs_cpu = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in dummy_inputs)

    os.makedirs(Path(onnx_path).parent, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy_inputs_cpu,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
        )
        print(f"Successfully exported: {onnx_path}")
    except Exception as e:
        print(f"Error exporting {model.__class__.__name__} to {onnx_path}: {e}")
        print("  Inputs provided:")
        for name, inp in zip(input_names, dummy_inputs_cpu):
            if isinstance(inp, torch.Tensor):
                print(f"    {name}: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"    {name}: value={inp}")
        return False
    return True

# --- Config Loading Utilities ---
def get_config_value(cfg, key_path, default=None, expected_type=None, model_name_for_error=""):
    """Safely retrieves a value from a nested config dictionary."""
    current = cfg
    path_list = key_path.split('/')
    for key in path_list:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # print(f"Warning: Key '{key}' not found in path '{key_path}' for {model_name_for_error}. Using default: {default}")
            return default
    if expected_type and not isinstance(current, expected_type):
        print(f"Warning: Config value for '{key_path}' in {model_name_for_error} is type {type(current)}, expected {expected_type}. Using default: {default}")
        return default
    return current

def get_model_configs(model_root_dir: Path) -> dict:
    """Loads main and BiCodec specific YAML configurations."""
    configs = {"main": None, "bicodec_specific": None, "audio_tokenizer": None}
    main_config_path = model_root_dir / "config.yaml"
    bicodec_config_path = model_root_dir / "BiCodec" / "config.yaml"

    if main_config_path.exists():
        configs["main"] = sparktts_load_config(str(main_config_path))
        if configs["main"] and "audio_tokenizer" in configs["main"]:
            configs["audio_tokenizer"] = configs["main"]["audio_tokenizer"]
    else:
        print(f"Warning: Main config.yaml not found at {main_config_path}")

    if bicodec_config_path.exists():
        configs["bicodec_specific"] = sparktts_load_config(str(bicodec_config_path))
        # Prioritize BiCodec specific config for audio_tokenizer if main one wasn't found or BiCodec one exists
        if configs["bicodec_specific"]:
            if "audio_tokenizer" in configs["bicodec_specific"]:
                configs["audio_tokenizer"] = configs["bicodec_specific"]["audio_tokenizer"]
            elif "model" in configs["bicodec_specific"] and "audio_tokenizer" in configs["bicodec_specific"]["model"]:
                 configs["audio_tokenizer"] = configs["bicodec_specific"]["model"]["audio_tokenizer"]
    else:
        print(f"Warning: BiCodec specific config.yaml not found at {bicodec_config_path}")
    
    if not configs["audio_tokenizer"]:
        print(f"Critical Warning: Could not find 'audio_tokenizer' configuration in {main_config_path} or {bicodec_config_path}. Many dimensions will use defaults.")
    
    return configs

# --- 1. Convert LLM (Largely unchanged, self-contained) ---
def convert_llm(model_root_dir: Path, onnx_output_dir: Path, opset_version: int):
    print("\n--- Converting LLM ---")
    llm_model_path = model_root_dir / "LLM"
    if not llm_model_path.exists():
        print(f"LLM path {llm_model_path} not found. Skipping LLM conversion.")
        return
    try:
        llm = AutoModelForCausalLM.from_pretrained(str(llm_model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
    except Exception as e:
        print(f"Error loading LLM from {llm_model_path}: {e}")
        return

    dummy_text = "Hello world"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    onnx_path = onnx_output_dir / "llm.onnx"
    export_model_to_onnx(llm, (dummy_input_ids, dummy_attention_mask), input_names, output_names, dynamic_axes, str(onnx_path), opset_version)


# --- 2. Convert Wav2Vec2 Feature Extractor (Largely unchanged) ---
class Wav2Vec2Wrapper(nn.Module):
    def __init__(self, model, layers_to_extract):
        super().__init__()
        self.model = model
        self.layers_to_extract = layers_to_extract
        self.model.config.output_hidden_states = True
    def forward(self, input_values):
        outputs = self.model(input_values)
        extracted_states = []
        for layer_idx in self.layers_to_extract:
            if 0 <= layer_idx < len(outputs.hidden_states):
                extracted_states.append(outputs.hidden_states[layer_idx])
            else:
                raise ValueError(f"Invalid layer index {layer_idx} for Wav2Vec2 ({len(outputs.hidden_states)} hidden states).")
        return tuple(extracted_states)

def convert_wav2vec2(model_root_dir: Path, onnx_output_dir: Path, opset_version: int):
    print("\n--- Converting Wav2Vec2 Feature Extractor ---")
    w2v2_model_name = "wav2vec2-large-xlsr-53"
    w2v2_model_path = model_root_dir / w2v2_model_name
    if not w2v2_model_path.exists():
        print(f"Wav2Vec2 path {w2v2_model_path} not found. Skipping.")
        return
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(w2v2_model_path))
        feature_extractor_model = Wav2Vec2Model.from_pretrained(str(w2v2_model_path))
    except Exception as e:
        print(f"Error loading Wav2Vec2 from {w2v2_model_path}: {e}")
        return

    layers_to_extract = [11, 14, 16]
    wrapped_model = Wav2Vec2Wrapper(feature_extractor_model, layers_to_extract)
    dummy_raw_audio = torch.randn(1, 16000)
    dummy_input_values = processor(dummy_raw_audio, sampling_rate=16000, return_tensors="pt").input_values
    input_names = ["input_values"]
    output_names = [f"hidden_state_{i}" for i in layers_to_extract]
    dynamic_axes = {"input_values": {0: "batch_size", 1: "sequence_length"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size", 1: "feature_seq_len"}
    onnx_path = onnx_output_dir / "wav2vec2_feature_extractor.onnx"
    export_model_to_onnx(wrapped_model, dummy_input_values, input_names, output_names, dynamic_axes, str(onnx_path), opset_version)


# --- 3. Convert BiCodec Components ---
class FactorizedVQTokenizeWrapper(nn.Module):
    """Replicates FactorizedVectorQuantize.tokenize logic without einops."""
    def __init__(self, fqv_module: FactorizedVectorQuantize):
        super().__init__()
        self.in_project = fqv_module.in_project
        self.codebook_weight = fqv_module.codebook.weight # Buffer, not a submodule for ONNX

    def forward(self, z): # z: (B, D, T)
        z_e = self.in_project(z) # (B, codebook_dim, T)
        
        B_lat, D_cb, T_lat = z_e.shape
        # Reshape latents: (B, codebook_dim, T) -> (B, T, codebook_dim) -> (B*T, codebook_dim)
        encodings = z_e.permute(0, 2, 1).reshape(B_lat * T_lat, D_cb)
        codebook = self.codebook_weight # (codebook_size, codebook_dim)

        encodings_norm = F.normalize(encodings, p=2, dim=1)
        codebook_norm = F.normalize(codebook, p=2, dim=1)

        dist = (
            encodings_norm.pow(2).sum(1, keepdim=True)
            - 2 * (encodings_norm @ codebook_norm.t()) # Matmul
            + codebook_norm.pow(2).sum(1, keepdim=True).t() # Transpose sum for broadcasting
        )
        indices_flat = (-dist).max(1)[1] # Shape (B*T)
        indices = indices_flat.reshape(B_lat, T_lat) # Shape (B, T)
        return indices

class SpeakerEncoderTokenizeWrapper(nn.Module):
    """Wraps SpeakerEncoder.tokenize. Relies on ONNX opset for internal einops (in ResidualFSQ)."""
    def __init__(self, speaker_encoder_module: SpeakerEncoder):
        super().__init__()
        self.speaker_encoder_module = speaker_encoder_module
        if isinstance(self.speaker_encoder_module.quantizer, ResidualFSQ):
            print("Warning: SpeakerEncoder uses ResidualFSQ which may contain 'einops'. Export success depends on ONNX opset compatibility.")

    def forward(self, mels): # mels: (B, D_mel, T_mel)
        return self.speaker_encoder_module.tokenize(mels)

class FQVDetokenizeWrapper(nn.Module):
    def __init__(self, fqv_module: FactorizedVectorQuantize):
        super().__init__()
        self.out_project = fqv_module.out_project
        self.codebook_weight = fqv_module.codebook.weight
        # Replicate fqv_module.decode_code behavior for ONNX
    def _decode_code(self, embed_id):
         # embed_id: (B, T) or (B, G, T) for FSQ
         # FactorizedVQ expects (B,T)
        return F.embedding(embed_id, self.codebook_weight).permute(0,2,1) # (B,D,T)

    def forward(self, indices): # indices: (B,T) for FactorizedVQ
        z_q_no_proj = self._decode_code(indices)
        z_q = self.out_project(z_q_no_proj)
        return z_q

class SpkEncDetokenizeWrapper(nn.Module):
    def __init__(self, spk_enc_module: SpeakerEncoder):
        super().__init__()
        self.spk_enc_module = spk_enc_module # The detokenize method is hopefully simple enough
        if isinstance(self.spk_enc_module.quantizer, ResidualFSQ):
            print("Warning: SpeakerEncoder.detokenize uses ResidualFSQ which may use 'einops' in get_output_from_indices. Export depends on ONNX opset.")

    def forward(self, codes): # codes for SpeakerEncoder/ResidualFSQ: (B, NumQuantizers, T_token)
        return self.spk_enc_module.detokenize(codes)

def convert_bicodec_components(model_root_dir: Path, onnx_output_dir: Path, configs: dict, opset_version: int):
    print("\n--- Converting BiCodec Components ---")
    bicodec_model_dir = model_root_dir / "BiCodec"
    if not bicodec_model_dir.exists():
        print(f"BiCodec model directory {bicodec_model_dir} not found. Skipping.")
        return

    audio_cfg = configs.get("audio_tokenizer")
    if not audio_cfg:
        print("CRITICAL Error: audio_tokenizer config not found. Cannot determine dimensions. Skipping BiCodec components.")
        return

    cpu_device = torch.device("cpu")
    try:
        bicodec_full_model = BiCodec.load_from_checkpoint(bicodec_model_dir, device=cpu_device)
        bicodec_full_model.to(cpu_device)
    except Exception as e:
        print(f"Error loading BiCodec model from {bicodec_model_dir}: {e}")
        return

    dummy_batch_size = 1
    dummy_feat_dim = 1024 
    dummy_feat_seq_len = 150 # Example, should be dynamic

    quantizer_cfg = audio_cfg.get("quantizer", {}) # For FactorizedVectorQuantize (semantic)
    # Encoder output dim feeds into FQV input_dim
    # Fallback to actual model attribute if possible, then to a default (dummy_feat_dim)
    enc_out_dim_actual = bicodec_full_model.encoder.bottleneck_channels if hasattr(bicodec_full_model.encoder, 'bottleneck_channels') else dummy_feat_dim
    dummy_quantizer_input_dim = get_config_value(quantizer_cfg, "input_dim", enc_out_dim_actual, int, "FQV")
    dummy_fqv_codebook_size = get_config_value(quantizer_cfg, "codebook_size", 2048, int, "FQV")
    if dummy_fqv_codebook_size is None or dummy_quantizer_input_dim is None:
        print("Critical Error: Missing codebook_size or input_dim for FactorizedVQ. Skipping FQV components.")
        return

    speaker_encoder_cfg = audio_cfg.get("speaker_encoder", {})
    dummy_mel_dim = get_config_value(speaker_encoder_cfg, "input_dim", 100, int, "SpeakerEncoder")
    dummy_mel_seq_len = 250
    
    # SpeakerEncoder.quantizer is ResidualFSQ.
    actual_se_quantizer = bicodec_full_model.speaker_encoder.quantizer
    dummy_rfs_num_quantizers = get_config_value(speaker_encoder_cfg, "fsq_num_quantizers", actual_se_quantizer.num_quantizers if hasattr(actual_se_quantizer, 'num_quantizers') else 1, int, "SpeakerEncoder.ResidualFSQ")
    # For randint, max value is product of levels for FSQ, simpler to use a safe small number for dummy data if levels not parsed.
    # Max index for ResidualFSQ: product of levels. For simplicity, use a fixed small int for dummy data if not parsed.
    # Example: levels = [8,8,8,8], max_val = 8^4 -1. This is complex to get from config generally.
    # Max index for FQV is dummy_fqv_codebook_size - 1
    dummy_speaker_token_len = get_config_value(speaker_encoder_cfg, "token_num", 32, int, "SpeakerEncoder")

    prenet_cfg = audio_cfg.get("prenet", {})
    # FQV detokenize output_dim is quantizer_cfg["codebook_dim"] (not input_dim for FQV)
    fqv_codebook_dim = get_config_value(quantizer_cfg, "codebook_dim", 256, int, "FQV_codebook_dim")
    dummy_quantized_dim_for_prenet = fqv_codebook_dim 

    actual_se_out_dim = bicodec_full_model.speaker_encoder.out_dim if hasattr(bicodec_full_model.speaker_encoder, 'out_dim') else 256
    dummy_speaker_embed_dim_for_prenet = get_config_value(speaker_encoder_cfg, "out_dim", actual_se_out_dim, int, "SpeakerEncoder_out_dim")
    if None in [dummy_mel_dim, dummy_rfs_num_quantizers, dummy_speaker_token_len, fqv_codebook_dim, dummy_speaker_embed_dim_for_prenet]:
        print("Critical Error: One or more BiCodec dimensions could not be determined from config or defaults. Skipping BiCodec components.")
        return

    # a) BiCodec.encoder
    print("  Converting BiCodec.encoder...")
    dummy_feat_input_enc = torch.randn(dummy_batch_size, dummy_feat_dim, dummy_feat_seq_len)
    onnx_path_enc = onnx_output_dir / "bicodec_encoder.onnx"
    export_model_to_onnx(bicodec_full_model.encoder, dummy_feat_input_enc, ["feat_input"], ["encoded_z"],
                         {"feat_input": {0: "batch_size", 2: "feat_sequence"},
                          "encoded_z": {0: "batch_size", 2: "encoded_sequence"}},
                         str(onnx_path_enc), opset_version)
    
    # b) FactorizedVectorQuantize.tokenize
    print("  Converting FactorizedVectorQuantize.tokenize...")
    dummy_z_for_fqv_tokenize = torch.randn(dummy_batch_size, dummy_quantizer_input_dim, dummy_feat_seq_len) # Encoder output
    fqv_tokenize_wrapper = FactorizedVQTokenizeWrapper(bicodec_full_model.quantizer)
    onnx_path_fqv_tok = onnx_output_dir / "bicodec_factorized_vq_tokenize.onnx"
    export_model_to_onnx(fqv_tokenize_wrapper, dummy_z_for_fqv_tokenize, ["encoded_z"], ["semantic_tokens"],
                         {"encoded_z": {0: "batch_size", 2: "encoded_sequence"},
                          "semantic_tokens": {0: "batch_size", 1: "encoded_sequence"}},
                         str(onnx_path_fqv_tok), opset_version)

    # c) SpeakerEncoder.tokenize
    print("  Converting SpeakerEncoder.tokenize...")
    dummy_mels_for_se_tokenize = torch.randn(dummy_batch_size, dummy_mel_dim, dummy_mel_seq_len)
    se_tokenize_wrapper = SpeakerEncoderTokenizeWrapper(bicodec_full_model.speaker_encoder)
    onnx_path_se_tok = onnx_output_dir / "bicodec_speaker_encoder_tokenize.onnx"
    export_model_to_onnx(se_tokenize_wrapper, dummy_mels_for_se_tokenize, ["mels"], ["global_tokens_indices"],
                         {"mels": {0: "batch_size", 2: "mel_sequence"},
                          "global_tokens_indices": {0: "batch_size", 1:"num_quantizers", 2: "speaker_token_sequence"}},
                         str(onnx_path_se_tok), opset_version)

    # d) FactorizedVectorQuantize.detokenize
    print("  Converting FactorizedVectorQuantize.detokenize...")
    dummy_semantic_tokens_fqv = torch.randint(0, dummy_fqv_codebook_size, (dummy_batch_size, dummy_feat_seq_len), dtype=torch.long)
    fqv_detok_wrapper = FQVDetokenizeWrapper(bicodec_full_model.quantizer)
    onnx_path_fqv_detok = onnx_output_dir / "bicodec_factorized_vq_detokenize.onnx"
    export_model_to_onnx(fqv_detok_wrapper, dummy_semantic_tokens_fqv, ["semantic_tokens"], ["z_q"],
                         {"semantic_tokens": {0: "batch_size", 1: "sequence_tokens"},
                          "z_q": {0: "batch_size", 2: "sequence_quantized"}},
                         str(onnx_path_fqv_detok), opset_version)

    # e) SpeakerEncoder.detokenize
    print("  Converting SpeakerEncoder.detokenize...")
    # For ResidualFSQ indices, max value depends on product of levels. Use small placeholder.
    dummy_global_tokens_se = torch.randint(0, 5, (dummy_batch_size, dummy_rfs_num_quantizers, dummy_speaker_token_len), dtype=torch.long)
    spk_enc_detok_wrapper = SpkEncDetokenizeWrapper(bicodec_full_model.speaker_encoder)
    onnx_path_se_detok = onnx_output_dir / "bicodec_speaker_encoder_detokenize.onnx"
    export_model_to_onnx(spk_enc_detok_wrapper, dummy_global_tokens_se, ["global_tokens_indices"], ["d_vector"],
                         {"global_tokens_indices": {0: "batch_size", 1:"num_quantizers", 2: "speaker_token_sequence"},
                          "d_vector": {0: "batch_size"}},
                         str(onnx_path_se_detok), opset_version)

    # f) BiCodec.prenet
    print("  Converting BiCodec.prenet...")
    dummy_z_q_prenet = torch.randn(dummy_batch_size, dummy_quantized_dim_for_prenet, dummy_feat_seq_len)
    dummy_d_vector_prenet = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet)
    onnx_path_prenet = onnx_output_dir / "bicodec_prenet.onnx"
    export_model_to_onnx(bicodec_full_model.prenet, (dummy_z_q_prenet, dummy_d_vector_prenet), ["z_q", "d_vector_condition"], ["prenet_output_x"],
                         {"z_q": {0: "batch_size", 2: "sequence"}, "d_vector_condition": {0: "batch_size"},
                          "prenet_output_x": {0: "batch_size", 2: "sequence"}},
                         str(onnx_path_prenet), opset_version)

    # g) BiCodec.decoder (WaveGenerator)
    print("  Converting BiCodec.decoder (WaveGenerator)...")
    dummy_prenet_output_x_wg = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet, dummy_feat_seq_len)
    dummy_d_vector_wg = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet)
    dummy_wavegen_input = dummy_prenet_output_x_wg + dummy_d_vector_wg.unsqueeze(-1)
    onnx_path_wavegen = onnx_output_dir / "bicodec_wavegenerator.onnx"
    export_model_to_onnx(bicodec_full_model.decoder, dummy_wavegen_input, ["wavegen_input"], ["wav_recon"],
                         {"wavegen_input": {0: "batch_size", 2: "sequence"},
                          "wav_recon": {0: "batch_size", 1: "audio_sequence"}},
                         str(onnx_path_wavegen), opset_version)
    
    print("\n--- Einops and Complex Logic Note ---")
    print("FactorizedVQTokenizeWrapper includes a direct replication of logic to avoid 'einops'.")
    print("SpeakerEncoder and its internal ResidualFSQ may use 'einops'. Direct export success depends on ONNX opset compatibility.")
    print("If 'einops' errors occur for speaker_encoder_tokenize, that model may need 'ResidualFSQ' to be made ONNX-friendly manually or the wrapper to fully replicate its logic.")


# --- Main script execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Spark-TTS PyTorch models to ONNX.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_DIR_NAME,
                        help=f"Name of the model directory under pretrained_models (default: {DEFAULT_MODEL_DIR_NAME})")
    parser.add_argument("--model_base_dir", type=str, default=None,
                        help="Absolute path to the 'pretrained_models' directory if not in standard locations. Overrides auto-detection.")
    parser.add_argument("--opset_version", type=int, default=DEFAULT_OPSET_VERSION, help="ONNX opset version.")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM conversion.")
    parser.add_argument("--skip_wav2vec2", action="store_true", help="Skip Wav2Vec2 conversion.")
    parser.add_argument("--skip_bicodec", action="store_true", help="Skip BiCodec components conversion.")

    args = parser.parse_args()

    print(f"Starting ONNX Model Conversion for Spark-TTS ('{args.model_name}')")
    print(f"Using PyTorch version: {torch.__version__}")
    try:
        import onnx
        print(f"Using ONNX library version: {onnx.__version__}")
    except ImportError:
        print("ONNX library not found. Please install with: pip install onnx")
        exit(1)
    print(f"Target ONNX Opset: {args.opset_version}")

    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
    except NameError:
        project_root = Path(".").resolve()
        print(f"Warning: __file__ not defined, using current directory as project root: {project_root}")

    model_root_dir_to_convert = None
    if args.model_base_dir:
        model_root_dir_to_convert = Path(args.model_base_dir) / args.model_name
        if not (model_root_dir_to_convert.exists() and (model_root_dir_to_convert / "LLM").exists()): # Basic check
            print(f"ERROR: Provided model_base_dir + model_name ({model_root_dir_to_convert}) does not seem to be a valid model directory.")
            exit(1)
    else:
        # Auto-detection logic
        possible_model_base_paths = [
            project_root / "pretrained_models",
            project_root.parent / "pretrained_models",
            Path.home() / ".cache" / "sparktts_models",
            project_root # Check project root itself for model_name folder
        ]
        for base_path in possible_model_base_paths:
            candidate_path = base_path / args.model_name
            if candidate_path.exists() and (candidate_path / "LLM").exists() and (candidate_path / "BiCodec").exists():
                model_root_dir_to_convert = candidate_path
                break
    
    if not model_root_dir_to_convert:
        print(f"ERROR: Could not find or validate model directory for '{args.model_name}'. Searched in provided/default locations.")
        print("Please ensure the model directory exists and contains subfolders like 'LLM' and 'BiCodec'.")
        exit(1)
    
    print(f"Using model directory at: {model_root_dir_to_convert.resolve()}")
    
    model_configs = get_model_configs(model_root_dir_to_convert)
    if not args.skip_bicodec and not model_configs["audio_tokenizer"]:
        print("Fatal: 'audio_tokenizer' config could not be loaded, and BiCodec conversion is not skipped. This config is essential for BiCodec component dimensions.")
        exit(1)

    onnx_output_base_dir = project_root / DEFAULT_ONNX_OUTPUT_DIR
    onnx_model_specific_output_dir = onnx_output_base_dir / args.model_name
    os.makedirs(onnx_model_specific_output_dir, exist_ok=True)
    print(f"ONNX models will be saved to: {onnx_model_specific_output_dir.resolve()}")
    print("-" * 30)

    if not args.skip_llm:
        convert_llm(model_root_dir_to_convert, onnx_model_specific_output_dir, args.opset_version)
    if not args.skip_wav2vec2:
        convert_wav2vec2(model_root_dir_to_convert, onnx_model_specific_output_dir, args.opset_version)
    if not args.skip_bicodec:
        convert_bicodec_components(model_root_dir_to_convert, onnx_model_specific_output_dir, model_configs, args.opset_version)

    print("\n" + "="*30)
    print("ONNX Conversion Script Finished.")
    print(f"Models saved in: {onnx_model_specific_output_dir.resolve()}")
    print("IMPORTANT: Review logs for errors. Verify 'einops' related warnings if any.")
    print("Thoroughly test the exported ONNX models with the inference and comparison scripts.") 
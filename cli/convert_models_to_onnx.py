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
    # For einops replacement:
    # from einops import rearrange as einops_rearrange
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
def get_model_configs(model_root_dir: Path) -> dict:
    """Loads main and BiCodec specific YAML configurations."""
    configs = {"main": None, "bicodec": None, "audio_tokenizer": None}
    main_config_path = model_root_dir / "config.yaml"
    bicodec_config_path = model_root_dir / "BiCodec" / "config.yaml"

    if main_config_path.exists():
        configs["main"] = sparktts_load_config(str(main_config_path))
        # audio_tokenizer config is often nested in main config or BiCodec config
        if configs["main"] and "audio_tokenizer" in configs["main"]:
            configs["audio_tokenizer"] = configs["main"]["audio_tokenizer"]
    else:
        print(f"Warning: Main config.yaml not found at {main_config_path}")

    if bicodec_config_path.exists():
        configs["bicodec"] = sparktts_load_config(str(bicodec_config_path))
        if configs["bicodec"] and "audio_tokenizer" in configs["bicodec"] and not configs["audio_tokenizer"]:
             configs["audio_tokenizer"] = configs["bicodec"]["audio_tokenizer"]
        elif configs["bicodec"] and "model" in configs["bicodec"] and "audio_tokenizer" in configs["bicodec"]["model"]: # another common pattern
             configs["audio_tokenizer"] = configs["bicodec"]["model"]["audio_tokenizer"]


    if not configs["audio_tokenizer"]:
        print(f"Warning: Could not find 'audio_tokenizer' configuration in {main_config_path} or {bicodec_config_path}")
    
    return configs

# --- 1. Convert LLM ---
def convert_llm(model_root_dir: Path, onnx_output_dir: Path, opset_version: int):
    # (Same as before, no major changes for config loading needed here as it's self-contained with tokenizer)
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


# --- 2. Convert Wav2Vec2 Feature Extractor ---
class Wav2Vec2Wrapper(nn.Module):
    """Wrapper for Wav2Vec2Model to output specific hidden layers."""
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
                raise ValueError(f"Invalid layer index {layer_idx} for Wav2Vec2 model with {len(outputs.hidden_states)} hidden states.")
        return tuple(extracted_states)

def convert_wav2vec2(model_root_dir: Path, onnx_output_dir: Path, opset_version: int):
    # (Same as before)
    print("\n--- Converting Wav2Vec2 Feature Extractor ---")
    w2v2_model_name = "wav2vec2-large-xlsr-53"
    w2v2_model_path = model_root_dir / w2v2_model_name
    if not w2v2_model_path.exists():
        print(f"Wav2Vec2 model path {w2v2_model_path} not found. Skipping Wav2Vec2 conversion.")
        return

    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(w2v2_model_path))
        feature_extractor_model = Wav2Vec2Model.from_pretrained(str(w2v2_model_path))
    except Exception as e:
        print(f"Error loading Wav2Vec2 model from {w2v2_model_path}: {e}")
        return

    layers_to_extract = [11, 14, 16] # As used in BiCodecTokenizer
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

# Einops replacement helper (conceptual)
# For actual use, these would need to be robustly implemented if einops is not ONNX friendly
def replace_einops_rearrange_for_decode_latents(tensor, pattern, b):
    # Specific to: rearrange(latents, "b d t -> (b t) d")
    if pattern == "b d t -> (b t) d":
        B, D, T = tensor.shape
        return tensor.permute(0, 2, 1).reshape(B * T, D)
    # Specific to: rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
    elif pattern == "(b t) -> b t": # input is (B*T)
        # b is batch_size for latents
        BT = tensor.shape[0]
        T = BT // b
        return tensor.reshape(b, T)
    raise NotImplementedError(f"Einops pattern not implemented for ONNX: {pattern}")


class FactorizedVQTokenizeWrapper(nn.Module):
    def __init__(self, fqv_module: FactorizedVectorQuantize):
        super().__init__()
        self.fqv_module = fqv_module
        # TODO: Critical: If fqv_module.decode_latents uses einops directly,
        # this wrapper won't save it. The fqv_module itself needs to be made
        # ONNX-friendly or its logic replicated here without einops.
        # For now, assuming fqv_module.tokenize() might be simple enough or
        # that einops issues are handled elsewhere or by newer ONNX opsets.

    def forward(self, z): # z: (B, D, T)
        # Replicate FactorizedVectorQuantize.tokenize(z)
        z_e = self.fqv_module.in_project(z)

        # --- Start of replicated decode_latents logic (einops replaced) ---
        # encodings = einops_rearrange(z_e, "b d t -> (b t) d")
        B_lat, D_lat, T_lat = z_e.shape
        encodings = z_e.permute(0, 2, 1).reshape(B_lat * T_lat, D_lat)

        codebook = self.fqv_module.codebook.weight

        encodings_norm = F.normalize(encodings)
        codebook_norm = F.normalize(codebook)

        dist = (
            encodings_norm.pow(2).sum(1, keepdim=True)
            - 2 * encodings_norm @ codebook_norm.t()
            + codebook_norm.pow(2).sum(1, keepdim=True).t()
        )
        
        # indices_flat = einops_rearrange((-dist).max(1)[1], "(b t) -> b t", b=B_lat)
        indices_flat_arg = (-dist).max(1)[1] # shape (B*T)
        indices = indices_flat_arg.reshape(B_lat, T_lat)
        # --- End of replicated decode_latents logic ---
        return indices


class SpeakerEncoderTokenizeWrapper(nn.Module):
    def __init__(self, speaker_encoder_module: SpeakerEncoder):
        super().__init__()
        self.speaker_encoder_module = speaker_encoder_module
        # ResidualFSQ within SpeakerEncoder also uses einops for rearrange.
        # This is a deeper issue. For now, we export the tokenize method as is
        # and hope the opset or a future ONNX version handles it, or it fails gracefully.

    def forward(self, mels): # mels: (B, D_mel, T_mel)
        return self.speaker_encoder_module.tokenize(mels)


class FQVDetokenizeWrapper(nn.Module):
    def __init__(self, fqv_module):
        super().__init__()
        self.fqv_module = fqv_module
    def forward(self, codes):
        return self.fqv_module.detokenize(codes)

class SpkEncDetokenizeWrapper(nn.Module):
    def __init__(self, spk_enc_module):
        super().__init__()
        self.spk_enc_module = spk_enc_module
    def forward(self, codes):
        return self.spk_enc_module.detokenize(codes)


def convert_bicodec_components(model_root_dir: Path, onnx_output_dir: Path, configs: dict, opset_version: int):
    print("\n--- Converting BiCodec Components ---")
    bicodec_model_dir = model_root_dir / "BiCodec"
    if not bicodec_model_dir.exists():
        print(f"BiCodec model directory {bicodec_model_dir} not found. Skipping.")
        return

    audio_cfg = configs.get("audio_tokenizer")
    if not audio_cfg:
        print("Error: audio_tokenizer config not found for BiCodec. Cannot determine dimensions. Skipping BiCodec components.")
        return

    cpu_device = torch.device("cpu")
    try:
        bicodec_full_model = BiCodec.load_from_checkpoint(bicodec_model_dir, device=cpu_device)
        bicodec_full_model.to(cpu_device)
    except Exception as e:
        print(f"Error loading BiCodec model from {bicodec_model_dir}: {e}")
        return

    # --- Get dimensions from config ---
    dummy_batch_size = 1
    # Feature Encoder input
    dummy_feat_dim = 1024 # Typically from Wav2Vec2 XLS-R large
    dummy_feat_seq_len = 150  # Example, should be dynamic for ONNX

    # Quantizer related dimensions (FactorizedVectorQuantize)
    quantizer_cfg = audio_cfg.get("quantizer", {})
    dummy_quantizer_input_dim = quantizer_cfg.get("input_dim", bicodec_full_model.quantizer.input_dim if hasattr(bicodec_full_model.quantizer, 'input_dim') else dummy_feat_dim) # Encoder output dim
    dummy_codebook_size = quantizer_cfg.get("codebook_size", bicodec_full_model.quantizer.codebook_size)
    # FactorizedVQ doesn't have num_groups directly, it's one VQ. ResidualFSQ (used by SpkEnc) has groups/num_quantizers.
    # For FactorizedVQ, semantic tokens are (B, T_codes). For ResidualFSQ, (B, G, T_codes)
    # The existing FQVDetokenizeWrapper takes (B,G,T), this needs to align with FactorizedVQ's actual output for tokenize.
    # FactorizedVectorQuantize.tokenize returns (B, T)
    # FactorizedVectorQuantize.detokenize expects (B, T)
    # So, FQVDetokenizeWrapper should take (B,T) if it's for FactorizedVQ.
    # The dummy_semantic_tokens for FQV.detokenize should be (B, T_encoded_seq_len)

    # Speaker Encoder related dimensions
    speaker_encoder_cfg = audio_cfg.get("speaker_encoder", {})
    dummy_mel_dim = speaker_encoder_cfg.get("input_dim", 100) # e.g., n_mels
    dummy_mel_seq_len = 250 # Example
    # SpeakerEncoder uses ResidualFSQ. ResidualFSQ.num_quantizers is like 'groups'.
    rfsquantizer_cfg = speaker_encoder_cfg.get("quantizer", {}) # if speaker encoder has its own quantizer config
    if not rfsquantizer_cfg and hasattr(bicodec_full_model.speaker_encoder, 'quantizer'): # if quantizer is direct attribute
         rfsquantizer_cfg = {'num_quantizers': bicodec_full_model.speaker_encoder.quantizer.num_quantizers,
                             'levels': bicodec_full_model.speaker_encoder.quantizer.levels}


    dummy_speaker_encoder_groups = rfsquantizer_cfg.get("num_quantizers", 1)
    # For speaker_encoder.tokenize output: (B, num_quantizers, T_spk_token)
    # For speaker_encoder.detokenize input: (B, num_quantizers, T_spk_token) (FSQ indices)
    # The T_spk_token comes from PerceiverResampler's num_latents.
    dummy_speaker_token_len = speaker_encoder_cfg.get("token_num", 32)


    # Prenet/Decoder dimensions
    prenet_cfg = audio_cfg.get("prenet", {})
    dummy_quantized_dim_for_prenet = prenet_cfg.get("input_dims", bicodec_full_model.prenet.input_dims if hasattr(bicodec_full_model.prenet, 'input_dims') else 256) # Output of FQV.detokenize
    
    # Speaker embedding dim for prenet condition
    # This is output of speaker_encoder.detokenize, which is speaker_encoder.out_dim
    dummy_speaker_embed_dim_for_prenet = speaker_encoder_cfg.get("out_dim", bicodec_full_model.speaker_encoder.out_dim if hasattr(bicodec_full_model.speaker_encoder, 'out_dim') else 256)


    # --- a) BiCodec.encoder (feat_encoder.Encoder) ---
    print("  Converting BiCodec.encoder...")
    bicodec_encoder = bicodec_full_model.encoder
    dummy_feat_input_enc = torch.randn(dummy_batch_size, dummy_feat_dim, dummy_feat_seq_len)
    # Output z of encoder should have dummy_quantizer_input_dim as its channel dim
    onnx_path_enc = onnx_output_dir / "bicodec_encoder.onnx"
    export_model_to_onnx(bicodec_encoder, dummy_feat_input_enc, ["feat_input"], ["encoded_z"],
                         {"feat_input": {0: "batch_size", 2: "feat_sequence"},
                          "encoded_z": {0: "batch_size", 2: "encoded_sequence"}},
                         str(onnx_path_enc), opset_version)
    
    # --- b) FactorizedVectorQuantize.tokenize ---
    print("  Converting FactorizedVectorQuantize.tokenize...")
    bicodec_quantizer_fqv = bicodec_full_model.quantizer # This is a FactorizedVectorQuantize
    # Input 'z' from encoder: (B, dummy_quantizer_input_dim, T_encoded_seq_len)
    # T_encoded_seq_len depends on encoder's downsampling, assume it's dummy_feat_seq_len for now (can be dynamic)
    dummy_z_for_fqv_tokenize = torch.randn(dummy_batch_size, dummy_quantizer_input_dim, dummy_feat_seq_len)
    fqv_tokenize_wrapper = FactorizedVQTokenizeWrapper(bicodec_quantizer_fqv)
    onnx_path_fqv_tok = onnx_output_dir / "bicodec_factorized_vq_tokenize.onnx"
    export_model_to_onnx(fqv_tokenize_wrapper, dummy_z_for_fqv_tokenize, ["encoded_z"], ["semantic_tokens"],
                         {"encoded_z": {0: "batch_size", 2: "encoded_sequence"},
                          "semantic_tokens": {0: "batch_size", 1: "encoded_sequence"}}, # Output (B, T)
                         str(onnx_path_fqv_tok), opset_version)

    # --- c) SpeakerEncoder.tokenize ---
    print("  Converting SpeakerEncoder.tokenize...")
    bicodec_speaker_encoder_se = bicodec_full_model.speaker_encoder # This is a SpeakerEncoder
    dummy_mels_for_se_tokenize = torch.randn(dummy_batch_size, dummy_mel_dim, dummy_mel_seq_len)
    se_tokenize_wrapper = SpeakerEncoderTokenizeWrapper(bicodec_speaker_encoder_se)
    onnx_path_se_tok = onnx_output_dir / "bicodec_speaker_encoder_tokenize.onnx"
    # Output indices from SpeakerEncoder.tokenize (via ResidualFSQ) is (B, num_quantizers, T_spk_token)
    export_model_to_onnx(se_tokenize_wrapper, dummy_mels_for_se_tokenize, ["mels"], ["global_tokens_indices"],
                         {"mels": {0: "batch_size", 2: "mel_sequence"},
                          "global_tokens_indices": {0: "batch_size", 2: "speaker_token_sequence"}},
                         str(onnx_path_se_tok), opset_version)

    # --- d) FactorizedVectorQuantize.detokenize ---
    print("  Converting FactorizedVectorQuantize.detokenize...")
    # Input semantic_tokens for FQV.detokenize is (B, T_encoded_seq_len)
    dummy_semantic_tokens_fqv = torch.randint(0, dummy_codebook_size -1, (dummy_batch_size, dummy_feat_seq_len), dtype=torch.long)
    fqv_detok_wrapper = FQVDetokenizeWrapper(bicodec_quantizer_fqv) # bicodec_quantizer_fqv is FactorizedVectorQuantize
    onnx_path_fqv_detok = onnx_output_dir / "bicodec_factorized_vq_detokenize.onnx"
    # Output z_q from FQV.detokenize is (B, dummy_quantized_dim_for_prenet, T_encoded_seq_len)
    export_model_to_onnx(fqv_detok_wrapper, dummy_semantic_tokens_fqv, ["semantic_tokens"], ["z_q"],
                         {"semantic_tokens": {0: "batch_size", 1: "sequence_tokens"},
                          "z_q": {0: "batch_size", 2: "sequence_quantized"}},
                         str(onnx_path_fqv_detok), opset_version)

    # --- e) SpeakerEncoder.detokenize ---
    print("  Converting SpeakerEncoder.detokenize...")
    # Input global_tokens_indices for SE.detokenize is (B, dummy_speaker_encoder_groups, dummy_speaker_token_len)
    dummy_global_tokens_se = torch.randint(0, 10, (dummy_batch_size, dummy_speaker_encoder_groups, dummy_speaker_token_len), dtype=torch.long) # Max val depends on FSQ levels
    spk_enc_detok_wrapper = SpkEncDetokenizeWrapper(bicodec_speaker_encoder_se) # bicodec_speaker_encoder_se is SpeakerEncoder
    onnx_path_se_detok = onnx_output_dir / "bicodec_speaker_encoder_detokenize.onnx"
    # Output d_vector from SE.detokenize is (B, dummy_speaker_embed_dim_for_prenet)
    export_model_to_onnx(spk_enc_detok_wrapper, dummy_global_tokens_se, ["global_tokens_indices"], ["d_vector"],
                         {"global_tokens_indices": {0: "batch_size", 2: "speaker_token_sequence"},
                          "d_vector": {0: "batch_size"}},
                         str(onnx_path_se_detok), opset_version)

    # --- f) BiCodec.prenet (feat_decoder.Decoder) ---
    print("  Converting BiCodec.prenet...")
    bicodec_prenet = bicodec_full_model.prenet
    dummy_z_q_prenet = torch.randn(dummy_batch_size, dummy_quantized_dim_for_prenet, dummy_feat_seq_len)
    dummy_d_vector_prenet = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet)
    onnx_path_prenet = onnx_output_dir / "bicodec_prenet.onnx"
    export_model_to_onnx(bicodec_prenet, (dummy_z_q_prenet, dummy_d_vector_prenet), ["z_q", "d_vector_condition"], ["prenet_output_x"],
                         {"z_q": {0: "batch_size", 2: "sequence"}, "d_vector_condition": {0: "batch_size"},
                          "prenet_output_x": {0: "batch_size", 2: "sequence"}},
                         str(onnx_path_prenet), opset_version)

    # --- g) BiCodec.decoder (WaveGenerator) ---
    print("  Converting BiCodec.decoder (WaveGenerator)...")
    bicodec_wavegen = bicodec_full_model.decoder
    # Input to WaveGenerator: prenet_output_x + d_vector.unsqueeze(-1)
    # Assuming prenet_output_x channel dim matches dummy_speaker_embed_dim_for_prenet
    dummy_prenet_output_x_wg = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet, dummy_feat_seq_len)
    dummy_d_vector_wg = torch.randn(dummy_batch_size, dummy_speaker_embed_dim_for_prenet)
    dummy_wavegen_input = dummy_prenet_output_x_wg + dummy_d_vector_wg.unsqueeze(-1)
    onnx_path_wavegen = onnx_output_dir / "bicodec_wavegenerator.onnx"
    export_model_to_onnx(bicodec_wavegen, dummy_wavegen_input, ["wavegen_input"], ["wav_recon"],
                         {"wavegen_input": {0: "batch_size", 2: "sequence"},
                          "wav_recon": {0: "batch_size", 1: "audio_sequence"}},
                         str(onnx_path_wavegen), opset_version)
    
    print("\n--- Einops and Complex Logic Note ---")
    print("FactorizedVQTokenizeWrapper includes a conceptual replacement for 'einops'. Review for correctness.")
    print("SpeakerEncoder and its ResidualFSQ use 'einops'. Direct export might fail or require newer ONNX opsets.")
    print("Thorough testing of exported tokenize models is essential.")


# --- Main script execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Spark-TTS PyTorch models to ONNX.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_DIR_NAME,
                        help=f"Name of the model directory under pretrained_models (default: {DEFAULT_MODEL_DIR_NAME})")
    parser.add_argument("--model_base_dir", type=str, default=None,
                        help="Absolute path to the 'pretrained_models' directory if not in standard locations.")
    parser.add_argument("--opset_version", type=int, default=DEFAULT_OPSET_VERSION, help="ONNX opset version.")
    args = parser.parse_args()

    print(f"Starting ONNX Model Conversion for Spark-TTS ({args.model_name})")
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
        candidate_path = Path(args.model_base_dir) / args.model_name
        if candidate_path.exists() and (candidate_path / "LLM").exists() and (candidate_path / "BiCodec").exists():
            model_root_dir_to_convert = candidate_path
    else:
        possible_model_base_paths = [
            project_root / "pretrained_models",
            project_root.parent / "pretrained_models",
            Path.home() / ".cache" / "sparktts_models"
        ]
        for base_path in possible_model_base_paths:
            candidate_path = base_path / args.model_name
            if candidate_path.exists() and (candidate_path / "LLM").exists() and (candidate_path / "BiCodec").exists():
                model_root_dir_to_convert = candidate_path
                break
    
    if not model_root_dir_to_convert:
        print(f"ERROR: Could not find model directory for '{args.model_name}'. Searched in provided/default locations.")
        exit(1)
    
    print(f"Found model directory at: {model_root_dir_to_convert.resolve()}")
    
    # Load configurations
    model_configs = get_model_configs(model_root_dir_to_convert)
    if not model_configs["audio_tokenizer"]: # Critical for BiCodec dimensions
        print("Fatal: 'audio_tokenizer' config could not be loaded. This is essential for BiCodec component dimensions.")
        print("Ensure config.yaml exists in model root and/or BiCodec subdirectory and contains 'audio_tokenizer' section.")
        exit(1)


    onnx_output_base_dir = project_root / DEFAULT_ONNX_OUTPUT_DIR
    onnx_model_specific_output_dir = onnx_output_base_dir / args.model_name
    os.makedirs(onnx_model_specific_output_dir, exist_ok=True)
    print(f"ONNX models will be saved to: {onnx_model_specific_output_dir.resolve()}")
    print("-" * 30)

    # Run conversions
    # convert_llm(model_root_dir_to_convert, onnx_model_specific_output_dir, args.opset_version)
    # convert_wav2vec2(model_root_dir_to_convert, onnx_model_specific_output_dir, args.opset_version)
    convert_bicodec_components(model_root_dir_to_convert, onnx_model_specific_output_dir, model_configs, args.opset_version)

    print("\n" + "="*30)
    print("ONNX Conversion Script Finished.")
    print(f"Models saved in: {onnx_model_specific_output_dir.resolve()}")
    print("IMPORTANT: Review logs for errors. Dummy input dimensions are now read from configs where possible,")
    print("but verify against actual model architecture. 'Einops' usage in some modules might pose challenges for ONNX export.")
    print("Thoroughly test the exported ONNX models.") 
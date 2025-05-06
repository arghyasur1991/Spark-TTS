import torch
import torch.nn as nn
import os
from pathlib import Path
import warnings

# Suppress specific warnings during ONNX export if necessary
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Attempt to import necessary project modules
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from sparktts.utils.file import load_config
    from sparktts.models.bicodec import BiCodec
    # The following are submodules of BiCodec, but might be needed for type hinting or direct instantiation if not using BiCodec.load_from_checkpoint
    # from sparktts.modules.encoder_decoder.feat_encoder import Encoder
    # from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
    # from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
    # from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
    # from sparktts.modules.encoder_decoder.feat_decoder import Decoder as SparkDecoder # Alias to avoid clash if torch.nn.Decoder is used
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from a context where Spark-TTS modules are accessible (e.g., project root or with PYTHONPATH set).")
    exit(1)


# --- Configuration ---
# MODEL_DIR should point to the root directory of a specific SparkTTS model,
# e.g., "pretrained_models/Spark-TTS-0.5B"
DEFAULT_MODEL_DIR_NAME = "Spark-TTS-0.5B" # Default name of the model folder
DEFAULT_ONNX_OUTPUT_DIR = "onnx_models"

# --- Helper function for ONNX export ---
def export_model_to_onnx(model, dummy_inputs, input_names, output_names, dynamic_axes, onnx_path, opset_version=14):
    """Exports a PyTorch model to ONNX."""
    model.eval()
    if not isinstance(dummy_inputs, tuple):
        dummy_inputs = (dummy_inputs,)

    device = torch.device("cpu")
    model.to(device)
    dummy_inputs_cpu = []
    for t_input in dummy_inputs:
        if isinstance(t_input, torch.Tensor):
            dummy_inputs_cpu.append(t_input.to(device))
        else:
            dummy_inputs_cpu.append(t_input) # For non-tensor inputs if any (e.g. static parameters)
    dummy_inputs_cpu = tuple(dummy_inputs_cpu)


    # Ensure target directory exists
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
            verbose=False, # Set to True for detailed export logging
            # example_outputs=model(*dummy_inputs_cpu) # Can help with some complex models
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


# --- 1. Convert LLM ---
def convert_llm(model_root_dir: Path, onnx_output_dir: Path):
    print("\\n--- Converting LLM ---")
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
    export_model_to_onnx(llm, (dummy_input_ids, dummy_attention_mask), input_names, output_names, dynamic_axes, str(onnx_path))

# --- 2. Convert Wav2Vec2 Feature Extractor ---
class Wav2Vec2Wrapper(nn.Module):
    """Wrapper for Wav2Vec2Model to output specific hidden layers."""
    def __init__(self, model, layers_to_extract):
        super().__init__()
        self.model = model
        self.layers_to_extract = layers_to_extract
        # Ensure model's config is set to output hidden states
        self.model.config.output_hidden_states = True

    def forward(self, input_values):
        outputs = self.model(input_values)
        # hidden_states is a tuple: (embeddings, layer1_out, layer2_out, ...)
        # Ensure requested layer indices are valid
        extracted_states = []
        for layer_idx in self.layers_to_extract:
            if 0 <= layer_idx < len(outputs.hidden_states):
                extracted_states.append(outputs.hidden_states[layer_idx])
            else:
                # This should not happen if layers_to_extract is correct
                raise ValueError(f"Invalid layer index {layer_idx} for Wav2Vec2 model with {len(outputs.hidden_states)} hidden states.")
        return tuple(extracted_states) # Return as tuple for multiple outputs in ONNX

def convert_wav2vec2(model_root_dir: Path, onnx_output_dir: Path):
    print("\\n--- Converting Wav2Vec2 Feature Extractor ---")
    w2v2_model_name = "wav2vec2-large-xlsr-53"
    w2v2_model_path = model_root_dir / w2v2_model_name
    if not w2v2_model_path.exists():
        print(f"Wav2Vec2 model path {w2v2_model_path} not found. Skipping Wav2Vec2 conversion.")
        return

    try:
        # feature_extractor_model = Wav2Vec2Model.from_pretrained(str(w2v2_model_path), output_hidden_states=True)
        # Wav2Vec2FeatureExtractor is for pre-processing, Wav2Vec2Model is the actual model
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(w2v2_model_path))
        feature_extractor_model = Wav2Vec2Model.from_pretrained(str(w2v2_model_path))
    except Exception as e:
        print(f"Error loading Wav2Vec2 model from {w2v2_model_path}: {e}")
        return

    # Indices used in BiCodecTokenizer: 11, 14, 16 from feat.hidden_states
    # Assuming these are 0-indexed from the 'hidden_states' tuple.
    # The tuple structure is typically: (input_embeddings, layer1_output, ..., layerN_output)
    # If XLS-R 53 has N layers, hidden_states[0] is emb, hidden_states[i] is layer i output.
    # So index 11 = layer 11 output.
    layers_to_extract = [11, 14, 16]
    wrapped_model = Wav2Vec2Wrapper(feature_extractor_model, layers_to_extract)

    # Dummy input: raw waveform, processed by Wav2Vec2FeatureExtractor
    # A typical audio might be 1 second at 16kHz -> 16000 samples
    dummy_raw_audio = torch.randn(1, 16000) # 1 batch, 1 sec audio at 16kHz
    dummy_input_values = processor(dummy_raw_audio, sampling_rate=16000, return_tensors="pt").input_values

    input_names = ["input_values"]
    output_names = [f"hidden_state_{i}" for i in layers_to_extract]
    dynamic_axes = {"input_values": {0: "batch_size", 1: "sequence_length"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size", 1: "feature_seq_len"}

    onnx_path = onnx_output_dir / "wav2vec2_feature_extractor.onnx"
    export_model_to_onnx(wrapped_model, dummy_input_values, input_names, output_names, dynamic_axes, str(onnx_path))


# --- 3. Convert BiCodec Components ---
# Helper Wrappers for BiCodec sub-modules that have complex methods or loops
class FQVDetokenizeWrapper(nn.Module):
    def __init__(self, fqv_module):
        super().__init__()
        self.fqv_module = fqv_module
    def forward(self, codes): # codes is semantic_tokens (B, G, T_codes)
        return self.fqv_module.detokenize(codes)

class SpkEncDetokenizeWrapper(nn.Module):
    def __init__(self, spk_enc_module):
        super().__init__()
        self.spk_enc_module = spk_enc_module
    def forward(self, codes): # codes is global_tokens (B, G_spk)
        return self.spk_enc_module.detokenize(codes)

# NOTE: For FactorizedVectorQuantize.tokenize and SpeakerEncoder.tokenize:
# These methods involve finding nearest neighbors (codebook lookup), often using `argmin`.
# While `argmin` is ONNX-exportable, the full data flow and any Python logic within
# `tokenize` or `encode` methods need careful wrapping to be purely tensor operations.
# This might require creating dedicated nn.Module wrappers for those specific paths.
# For this script, we'll focus on detokenize and other main components.

def convert_bicodec_components(model_root_dir: Path, onnx_output_dir: Path):
    print("\\n--- Converting BiCodec Components ---")
    bicodec_model_dir = model_root_dir / "BiCodec"
    if not bicodec_model_dir.exists():
        print(f"BiCodec model directory {bicodec_model_dir} not found. Skipping BiCodec components.")
        return

    # Load BiCodec which initializes sub-modules
    # The BiCodec load_from_checkpoint handles device logic carefully for MPS.
    # For export, we standardize to CPU.
    cpu_device = torch.device("cpu")
    try:
        # Pass device=cpu_device to ensure it loads with CPU in mind, esp. for MPS avoidance logic
        bicodec_full_model = BiCodec.load_from_checkpoint(bicodec_model_dir, device=cpu_device)
        bicodec_full_model.to(cpu_device) # Ensure all submodules also on CPU
    except Exception as e:
        print(f"Error loading BiCodec model from {bicodec_model_dir}: {e}")
        return

    # Dummy data dimensions (VERIFY AND ADJUST THESE BASED ON YOUR MODEL CONFIG)
    # These should ideally be read from the model's config.yaml if possible
    # Or by inspecting tensor shapes during a PyTorch run.
    dummy_batch_size = 1
    # For feat_encoder input (output of Wav2Vec2 feature extraction, then processed)
    # BiCodecTokenizer.extract_wav2vec2_features mixes hidden_states[11,14,16]
    # All these layers in wav2vec2-large-xlsr-53 have dim 1024.
    dummy_feat_dim = 1024
    dummy_feat_seq_len = 150  # Example sequence length for features
    # For quantizer.detokenize input (semantic_tokens)
    # From a sample BiCodec config: quantizer.num_groups might be 4
    dummy_quantizer_groups = 4 # Typically from BiCodec config['audio_tokenizer']['quantizer']['num_groups']
    dummy_codebook_size = 2048 # Typically from config, e.g. quantizer.codebook_size
    # For speaker_encoder.detokenize input (global_tokens)
    # From a sample BiCodec config: speaker_encoder.quantizer.num_groups might be 2
    dummy_speaker_encoder_groups = 2 # Typically from config
    # Dimensions for prenet input (z_q, d_vector)
    # z_q output dim from quantizer: quantizer.output_dims (e.g., 256)
    # d_vector output dim from speaker_encoder: speaker_encoder.output_dims (e.g., 256)
    dummy_quantized_dim = 256 # Example, check config: quantizer.output_dims
    dummy_speaker_embed_dim = 256 # Example, check config: speaker_encoder.output_dims


    # a) BiCodec.encoder (feat_encoder.Encoder)
    print("  Converting BiCodec.encoder...")
    bicodec_encoder = bicodec_full_model.encoder
    # Input to BiCodec.encoder is feat.transpose(1,2) which is (B, C_feat, T_feat)
    # The Encoder module's forward is `pre(x.transpose(1,2))` then ConvNet, so input (B, C, T) is fine.
    dummy_feat_input = torch.randn(dummy_batch_size, dummy_feat_dim, dummy_feat_seq_len)
    input_names_enc = ["feat_input"]
    output_names_enc = ["encoded_z"] # Output 'z'
    dynamic_axes_enc = {
        "feat_input": {0: "batch_size", 2: "feat_sequence"},
        "encoded_z": {0: "batch_size", 2: "encoded_sequence"}, # z has shape (B, quantizer_input_dim, T_encoded)
    }
    onnx_path_enc = onnx_output_dir / "bicodec_encoder.onnx"
    export_model_to_onnx(bicodec_encoder, dummy_feat_input, input_names_enc, output_names_enc, dynamic_axes_enc, str(onnx_path_enc))

    # b) BiCodec.quantizer.detokenize (FactorizedVectorQuantize.detokenize)
    print("  Converting BiCodec.quantizer.detokenize...")
    bicodec_quantizer = bicodec_full_model.quantizer
    # semantic_tokens shape (B, G, T_codes), type long
    dummy_semantic_tokens = torch.randint(0, dummy_codebook_size -1, (dummy_batch_size, dummy_quantizer_groups, dummy_feat_seq_len), dtype=torch.long)
    fqv_detok_wrapper = FQVDetokenizeWrapper(bicodec_quantizer)
    input_names_q_detok = ["semantic_tokens"]
    output_names_q_detok = ["z_q"] # Output z_q (quantized vectors)
    dynamic_axes_q_detok = {
        "semantic_tokens": {0: "batch_size", 2: "sequence_tokens"},
        "z_q": {0: "batch_size", 2: "sequence_quantized"}, # z_q shape (B, C_quantized, T_quantized)
    }
    onnx_path_q_detok = onnx_output_dir / "bicodec_quantizer_detokenize.onnx"
    export_model_to_onnx(fqv_detok_wrapper, dummy_semantic_tokens, input_names_q_detok, output_names_q_detok, dynamic_axes_q_detok, str(onnx_path_q_detok))

    # c) BiCodec.speaker_encoder.detokenize (SpeakerEncoder.detokenize)
    print("  Converting BiCodec.speaker_encoder.detokenize...")
    bicodec_speaker_encoder = bicodec_full_model.speaker_encoder
    # global_tokens shape (B, G_spk_groups), type long
    dummy_global_tokens = torch.randint(0, dummy_codebook_size - 1, (dummy_batch_size, dummy_speaker_encoder_groups), dtype=torch.long)
    spk_enc_detok_wrapper = SpkEncDetokenizeWrapper(bicodec_speaker_encoder)
    input_names_spk_detok = ["global_tokens"]
    output_names_spk_detok = ["d_vector"] # Output d_vector (speaker embedding)
    dynamic_axes_spk_detok = {
        "global_tokens": {0: "batch_size"}, # d_vector shape (B, C_speaker_embed)
        "d_vector": {0: "batch_size"},
    }
    onnx_path_spk_detok = onnx_output_dir / "bicodec_speaker_encoder_detokenize.onnx"
    export_model_to_onnx(spk_enc_detok_wrapper, dummy_global_tokens, input_names_spk_detok, output_names_spk_detok, dynamic_axes_spk_detok, str(onnx_path_spk_detok))

    # d) BiCodec.prenet (feat_decoder.Decoder)
    print("  Converting BiCodec.prenet...")
    bicodec_prenet = bicodec_full_model.prenet
    # Inputs: z_q from quantizer.detokenize, d_vector from speaker_encoder.detokenize
    dummy_z_q = torch.randn(dummy_batch_size, dummy_quantized_dim, dummy_feat_seq_len)
    dummy_d_vector = torch.randn(dummy_batch_size, dummy_speaker_embed_dim)
    # Prenet in BiCodec.detokenize is called as: self.prenet(z_q, d_vector)
    input_names_prenet = ["z_q", "d_vector_condition"]
    output_names_prenet = ["prenet_output_x"]
    dynamic_axes_prenet = {
        "z_q": {0: "batch_size", 2: "sequence"},
        "d_vector_condition": {0: "batch_size"},
        "prenet_output_x": {0: "batch_size", 2: "sequence"}, # Output x shape (B, C_prenet_out, T_out)
    }
    onnx_path_prenet = onnx_output_dir / "bicodec_prenet.onnx"
    export_model_to_onnx(bicodec_prenet, (dummy_z_q, dummy_d_vector), input_names_prenet, output_names_prenet, dynamic_axes_prenet, str(onnx_path_prenet))

    # e) BiCodec.decoder (WaveGenerator)
    print("  Converting BiCodec.decoder (WaveGenerator)...")
    bicodec_wavegen = bicodec_full_model.decoder # This is the WaveGenerator instance
    # Input to WaveGenerator in BiCodec.detokenize: x_combined = prenet_output_x + d_vector.unsqueeze(-1)
    # prenet_output_x shape e.g. (B, C_prenet_out, T_prenet_out) -> (1, 256, 150)
    # d_vector shape e.g. (B, C_speaker_embed) -> (1, 256)
    # So, d_vector.unsqueeze(-1) gives (B, C_speaker_embed, 1)
    # The addition implies C_prenet_out == C_speaker_embed
    dummy_prenet_output_x = torch.randn(dummy_batch_size, dummy_speaker_embed_dim, dummy_feat_seq_len) # Assuming C_prenet_out = C_speaker_embed
    dummy_d_vector_for_wavegen_input = torch.randn(dummy_batch_size, dummy_speaker_embed_dim)
    dummy_wavegen_input = dummy_prenet_output_x + dummy_d_vector_for_wavegen_input.unsqueeze(-1)

    input_names_wavegen = ["wavegen_input"]
    output_names_wavegen = ["wav_recon"] # Output waveform
    dynamic_axes_wavegen = {
        "wavegen_input": {0: "batch_size", 2: "sequence"},
        "wav_recon": {0: "batch_size", 1: "audio_sequence"}, # wav_recon shape (B, T_audio)
    }
    onnx_path_wavegen = onnx_output_dir / "bicodec_wavegenerator.onnx"
    export_model_to_onnx(bicodec_wavegen, dummy_wavegen_input, input_names_wavegen, output_names_wavegen, dynamic_axes_wavegen, str(onnx_path_wavegen))

    print("\\n--- Note on Tokenize Operations ---")
    print("BiCodec.quantizer.tokenize and BiCodec.speaker_encoder.tokenize involve codebook lookups.")
    print("These might require custom wrappers or careful implementation if their internal Python logic")
    print("is complex. Exporting them was skipped in this script but may be necessary for a full pipeline.")
    print("The core operation (argmin over distances) is ONNX-exportable, so it's feasible with effort.")


# --- Main script execution ---
if __name__ == "__main__":
    print("Starting ONNX Model Conversion Script for Spark-TTS")

    # Determine project root assuming script is in cli/ or similar structure
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent # Assumes cli/convert_script.py -> project_root/
    except NameError: # __file__ is not defined if run in some interactive environments
        project_root = Path(".").resolve() # Fallback to current directory
        print(f"Warning: __file__ not defined, using current directory as project root: {project_root}")


    # --- Configuration for Model and Output Directories ---
    # User should set this to the specific model directory, e.g., /path/to/Spark-TTS/pretrained_models/Spark-TTS-0.5B
    model_name_to_convert = DEFAULT_MODEL_DIR_NAME # Or get from args
    
    # Try to find a 'pretrained_models' directory up from the project root or common locations
    possible_model_base_paths = [
        project_root / "pretrained_models",
        project_root.parent / "pretrained_models", # If project_root is e.g. Spark-TTS/cli
        Path.home() / ".cache" / "sparktts_models" # A potential global cache
    ]
    
    model_root_dir_to_convert = None
    for base_path in possible_model_base_paths:
        candidate_path = base_path / model_name_to_convert
        if candidate_path.exists() and (candidate_path / "LLM").exists() and (candidate_path / "BiCodec").exists():
            model_root_dir_to_convert = candidate_path
            print(f"Found model directory at: {model_root_dir_to_convert}")
            break
    
    if model_root_dir_to_convert is None:
        print(f"ERROR: Could not automatically find model directory for '{model_name_to_convert}'.")
        print("Please set 'model_root_dir_to_convert' manually in the script, or ensure your models are in one of these locations:")
        for base_path in possible_model_base_paths:
             print(f"  - {base_path / model_name_to_convert}")
        print(f"Current project root considered: {project_root}")
        exit(1)

    # ONNX models will be saved in a subfolder of the main project directory
    onnx_output_base_dir = project_root / DEFAULT_ONNX_OUTPUT_DIR
    onnx_model_specific_output_dir = onnx_output_base_dir / model_name_to_convert
    os.makedirs(onnx_model_specific_output_dir, exist_ok=True)
    print(f"ONNX models will be saved to: {onnx_model_specific_output_dir.resolve()}")
    print("-" * 30)


    # Run conversions
    print(f"Using PyTorch version: {torch.__version__}")
    # It's good practice to also log onnx and onnxruntime versions if they are used for verification later
    try:
        import onnx
        print(f"Using ONNX version: {onnx.__version__}")
    except ImportError:
        print("ONNX library not found. `pip install onnx`")
    
    # convert_llm(model_root_dir_to_convert, onnx_model_specific_output_dir)
    # convert_wav2vec2(model_root_dir_to_convert, onnx_model_specific_output_dir)
    convert_bicodec_components(model_root_dir_to_convert, onnx_model_specific_output_dir)

    print("\\n" + "="*30)
    print("ONNX Conversion Script Finished.")
    print(f"Models saved in: {onnx_model_specific_output_dir.resolve()}")
    print("IMPORTANT: Review any error messages above. ")
    print("The dummy input dimensions and model paths might need adjustment for your specific setup.")
    print("The VQ 'tokenize' parts were skipped and require further work if needed for your pipeline.")
    print("Ensure you have the necessary 'opset_version' for torch.onnx.export that matches your ONNX Runtime.") 
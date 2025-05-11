
from transformers import AutoTokenizer

# --- USER CONFIGURATION ---\n# TODO: Adjust this path to your HuggingFace LLM tokenizer directory.
# This is the directory that contains tokenizer.json, vocab.json, etc.,
# for the LLM model you exported to ONNX and are using in Unity.
# Example: "/Users/sur/Personal/Projects/ML/Spark-TTS/LLM_onnx"
# or the specific model checkpoint folder like "/Users/sur/Personal/Projects/ML/Spark-TTS/some_model_checkpoint_for_llm"
python_tokenizer_path = "/Users/sur/Personal/Projects/ML/Spark-TTS/onnx_models/LLM_onnx" # <<< PLEASE VERIFY THIS PATH
# --- END USER CONFIGURATION ---

try:
    # trust_remote_code=True might be needed if the tokenizer uses custom code.
    # SparkTTS tokenizers are often standard, but it's safer to include.
    tokenizer = AutoTokenizer.from_pretrained(python_tokenizer_path, trust_remote_code=True)
except Exception as e_tokenizer:
    print(f"Error initializing AutoTokenizer from path: {python_tokenizer_path}")
    print(f"Error: {e_tokenizer}")
    print("Please ensure `python_tokenizer_path` is correct and the directory contains the necessary HuggingFace tokenizer files.")
    tokenizer = None

if tokenizer:
    text_to_synthesize = "Hello world" # Same as in TTSTestHarness
    
    # Speaker ID from your C# logs
    global_speaker_numeric_id = 112 
    
    # Construct the prompt string exactly as logged in C#
    # C# constructed: <|task_tts|><|start_content|>Hello world<|end_content|><|start_global_token|><|bicodec_global_112|><|end_global_token|>
    # C# normalized:  <|task_tts|><|start_content|>hello world<|end_content|><|start_global_token|><|bicodec_global_112|><|end_global_token|>
    # HuggingFace tokenizers handle normalization internally. We provide the un-normalized text.

    # Based on SparkTTS.py process_prompt logic for when prompt_text is None:
    task_token = "<|task_tts|>" # Assuming this mapping
    start_content_token = "<|start_content|>"
    end_content_token = "<|end_content|>"
    start_global_token = "<|start_global_token|>"
    end_global_token = "<|end_global_token|>"
    
    speaker_token_str = f"<|bicodec_global_{global_speaker_numeric_id}|>"
    
    # The text part should be the original case, tokenizer will handle normalization
    input_prompt_parts = [
        task_token,
        start_content_token,
        text_to_synthesize, # "Hello world" (original case)
        end_content_token,
        start_global_token,
        speaker_token_str,
        end_global_token,
    ]
    input_prompt_string_for_python = "".join(input_prompt_parts)
    
    print(f"Python constructed input_prompt_string: '{input_prompt_string_for_python}'")

    # Tokenize using the HuggingFace tokenizer.
    # add_special_tokens=False is important because we've manually constructed the full sequence with all special tokens.
    # The C# TokenizerService.EncodeForLLM also effectively does this by tokenizing the fully constructed string.
    tokenized_output = tokenizer(input_prompt_string_for_python, add_special_tokens=False)
    python_input_ids = tokenized_output.input_ids
    
    print(f"Python generated input_ids: {python_input_ids}")
    print(f"Python input_ids length: {len(python_input_ids)}")
    
    csharp_input_ids = [151633, 151638, 31373, 1025, 151639, 151641, 100112, 151642]
    print(f"C# logged input_ids:      {csharp_input_ids}")
    
    if python_input_ids == csharp_input_ids:
        print("\nToken IDs MATCH between Python and C#.")
    else:
        print("\nToken IDs DO NOT MATCH.")
        print("Differences:")
        for i in range(max(len(python_input_ids), len(csharp_input_ids))):
            py_id = python_input_ids[i] if i < len(python_input_ids) else "N/A"
            cs_id = csharp_input_ids[i] if i < len(csharp_input_ids) else "N/A"
            if py_id != cs_id:
                py_token_str = tokenizer.decode([python_input_ids[i]]) if i < len(python_input_ids) else "N/A"
                # Decoding C# token requires C# tokenizer - just show ID for now
                print(f"  Index {i}: Python ID {py_id} (Decodes to: '{py_token_str}') --- C# ID {cs_id}")

else:
    print("Python tokenizer could not be initialized. Cannot perform comparison.")
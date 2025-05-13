#!/bin/bash

# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ANSI color codes for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname "$script_dir")

# Set default parameters
device=0
save_dir='example/results'
model_dir="pretrained_models/Spark-TTS-0.5B"
text="Hello world, How are you? I am also good. So I am the narrator"
# text="Did you not *just* state it? 'Arghya', was it? Yes, I heard you. I acknowledged it."
prompt_speech_path="/Users/sur/Personal/Projects/ML/MysteryAI/Assets/Resources/audio/narrator_voice_prompt.wav"
# prompt_speech_path="example/results/prompt.wav"

# Generation parameters
quantization="none"  # none, fp16, int8
max_new_tokens=3000
do_sample=true
temperature=0.8
top_k=50
top_p=0.95
reuse_tokenization=false

# Acceleration options
use_compile=true
compile_mode="reduce-overhead"  # default, reduce-overhead, max-autotune

# --- Test ONNX for Wav2Vec2 --- #
use_wav2vec2_onnx=true # Set to true to test ONNX, false for PyTorch
use_bicodec_onnx_model=true  # Set to true to use ONNX for BiCodec Vocoder
use_speaker_encoder_tokenizer_onnx=true  # Set to true to use ONNX for Speaker Encoder Tokenizer
use_llm_onnx=true # New flag for ONNX LLM
use_mel_spectrogram_onnx=true # New flag for ONNX Mel Spectrogram, default to false
use_bicodec_encoder_quantizer_onnx=true # Added flag for Encoder/Quantizer ONNX
# ----------------------------- #

# Change directory to the root directory
cd "$root_dir" || exit

source sparktts/utils/parse_options.sh

# Display configuration
echo -e "${BLUE}SparkTTS Inference${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "• Model: $model_dir"
echo "• Device: $device"
echo "• Quantization: $quantization"
echo "• Text: $text"
echo "• Prompt: $prompt_speech_path"
echo "• Generation Parameters:"
echo "  - Max Tokens: $max_new_tokens"
echo "  - Temperature: $temperature"
echo "  - Top-k: $top_k"
echo "  - Top-p: $top_p"
echo "  - Sampling: $do_sample"
echo "  - Reuse Tokenization: $reuse_tokenization"
echo "  - Use Compile: $use_compile"
echo "  - Compile Mode: $compile_mode"
echo "  - Use Wav2Vec2 ONNX: $use_wav2vec2_onnx"
echo "  - Use BiCodec ONNX: $use_bicodec_onnx_model"
echo "  - Use Speaker Encoder Tokenizer ONNX: $use_speaker_encoder_tokenizer_onnx"
echo "  - Use LLM ONNX: $use_llm_onnx"
echo "  - Use Mel Spectrogram ONNX: $use_mel_spectrogram_onnx"
echo "  - Use BiCodec Encoder/Quantizer ONNX: $use_bicodec_encoder_quantizer_onnx"
echo

# Set up additional arguments
additional_args=""
if [ "$reuse_tokenization" = true ]; then
  additional_args="$additional_args --reuse_tokenization"
fi

if [ "$do_sample" = true ]; then
  additional_args="$additional_args --do_sample"
fi

if [ "$use_compile" = true ]; then
  additional_args="$additional_args --use_compile"
fi

# Add ONNX argument if enabled
onnx_arg=""
if [ "$use_wav2vec2_onnx" = true ]; then
  onnx_arg="$onnx_arg --use_wav2vec2_onnx"
fi
if [[ "$use_bicodec_onnx_model" == "true" ]]; then # This is the new conditional block
  onnx_arg="$onnx_arg --use_bicodec_onnx"
fi
if [ "$use_speaker_encoder_tokenizer_onnx" = true ]; then
  onnx_arg="$onnx_arg --use_speaker_encoder_tokenizer_onnx"
fi
if [ "$use_llm_onnx" = true ]; then
  onnx_arg="$onnx_arg --use_llm_onnx"
fi
if [ "$use_mel_spectrogram_onnx" = true ]; then
  onnx_arg="$onnx_arg --use_mel_spectrogram_onnx"
fi
if [ "$use_bicodec_encoder_quantizer_onnx" = true ]; then # Added logic for new flag
  onnx_arg="$onnx_arg --use_bicodec_encoder_quantizer_onnx"
fi

echo -e "${GREEN}Starting inference...${NC}"

# Run inference
python -m cli.inference \
    --text "${text}" \
    --device "${device}" \
    --save_dir "${save_dir}" \
    --model_dir "${model_dir}" \
    --prompt_speech_path "${prompt_speech_path}" \
    --quantization "${quantization}" \
    --max_new_tokens "${max_new_tokens}" \
    --temperature "${temperature}" \
    --top_k "${top_k}" \
    --top_p "${top_p}" \
    --compile_mode "${compile_mode}" \
    ${additional_args} \
    ${onnx_arg}

echo -e "${GREEN}Inference complete! Results saved to ${save_dir}${NC}"


# --- Style Control Example ---
echo
echo -e "${BLUE}SparkTTS Style Control Inference Example${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "• Model: $model_dir"
echo "• Device: $device"
echo "• Quantization: $quantization"
echo "• Text: Style Control Example Text"
echo "• Style Parameters:"
echo "  - Gender: male"
echo "  - Pitch: high"
echo "  - Speed: low"
echo "• Generation Parameters:"
echo "  - Max Tokens: $max_new_tokens"
echo "  - Temperature: $temperature"
echo "  - Top-k: $top_k"
echo "  - Top-p: $top_p"
echo "  - Sampling: $do_sample"
echo "  - Use Compile: $use_compile"
echo "  - Compile Mode: $compile_mode"
echo "  - Use LLM ONNX: $use_llm_onnx" # Note: Prompt-related ONNX flags are not relevant here
echo

echo -e "${GREEN}Starting style control inference...${NC}"

# Run style control inference
python -m cli.inference \
    --text "${text}" \
    --device "${device}" \
    --save_dir "${save_dir}/style_control" \
    --model_dir "${model_dir}" \
    --gender "female" \
    --pitch "high" \
    --speed "moderate" \
    --quantization "${quantization}" \
    --max_new_tokens "${max_new_tokens}" \
    --temperature "${temperature}" \
    --top_k "${top_k}" \
    --top_p "${top_p}" \
    --compile_mode "${compile_mode}" \
    ${additional_args} \
    ${onnx_arg} # Include relevant ONNX flags if used (e.g., LLM, BiCodec)

echo -e "${GREEN}Style control inference complete! Results saved to ${save_dir}/style_control${NC}"
    
    
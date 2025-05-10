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
prompt_text="Very well. Now that we've dispensed with introductions – which I assure you were not high on my list of priorities – do you have a *point*? Or are you simply cataloging the names of everyone present? Because unless your name is somehow pertinent to the sudden demise of Lord Ashworth, I suggest you move on to something that *is*."
text="Did you not *just* state it? 'Arghya', was it? Yes, I heard you. I acknowledged it."
# text="Did you not *just* state it? 'Arghya', was it? Yes, I heard you. I acknowledged it. Is there some reason you require this fact reiterated? Or are we quite finished with pointless exercises and ready to discuss something pertinent to *why* we are all trapped in this house with a dead man? "
prompt_speech_path="example/results/prompt.wav"

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
  onnx_arg="--use_wav2vec2_onnx"
fi
if [[ "$use_bicodec_onnx_model" == "true" ]]; then # This is the new conditional block
  onnx_arg="$onnx_arg --use_bicodec_onnx"
fi
if [ "$use_speaker_encoder_tokenizer_onnx" = true ]; then
  onnx_arg="$onnx_arg --use_speaker_encoder_tokenizer_onnx"
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
    
    
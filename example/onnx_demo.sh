#!/bin/bash

# Spark-TTS ONNX Demonstration Script
# This script demonstrates how to convert and use Spark-TTS ONNX models

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Define directories
MODEL_DIR="pretrained_models/Spark-TTS-0.5B"
OUTPUT_DIR="$MODEL_DIR/onnx"
PROMPT_PATH="$SCRIPT_DIR/prompt_audio.wav"
RESULTS_DIR="$SCRIPT_DIR/results"

# Voice settings
GENDER=0.0  # Male voice (0.0 is male, 1.0 is female)
PITCH=0.25  # Low pitch (0.0 very low to 1.0 very high)
SPEED=0.5   # Moderate speed (0.0 very slow to 1.0 very fast)

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model directory $MODEL_DIR does not exist."
  echo "Please download the pretrained models first."
  exit 1
fi

# Check if prompt audio exists
if [ ! -f "$PROMPT_PATH" ]; then
  echo "Error: Prompt audio file $PROMPT_PATH does not exist."
  exit 1
fi

# Step 1: Convert PyTorch model to ONNX
echo "===== Step 1: Converting PyTorch model to ONNX ====="
echo "Running: python convert_to_onnx.py --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR --skip_verification"
python "$PROJECT_DIR/convert_to_onnx.py" --model_dir "$MODEL_DIR" --output_dir "$OUTPUT_DIR" --skip_verification

# Check if conversion was successful
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Error: ONNX model conversion failed. Output directory $OUTPUT_DIR does not exist."
  exit 1
fi

# Count ONNX models
NUM_MODELS=$(ls -1 "$OUTPUT_DIR"/*.onnx 2>/dev/null | wc -l)
if [ "$NUM_MODELS" -eq 0 ]; then
  echo "Error: No ONNX models found in $OUTPUT_DIR"
  exit 1
fi

echo "ONNX conversion complete. $NUM_MODELS models created in $OUTPUT_DIR"
echo "Models:"
ls -lh "$OUTPUT_DIR"/*.onnx

# Step 2: Run inference with ONNX model
echo -e "\n===== Step 2: Running inference with ONNX model ====="
TEST_TEXT="This is a demonstration of Spark TTS using ONNX Runtime for faster inference. Now with male voice, low pitch, and moderate speed."

echo "Running: python -m cli.spark_tts_onnx --model-dir $MODEL_DIR --prompt-path $PROMPT_PATH --output-dir $RESULTS_DIR --gender $GENDER --pitch $PITCH --speed $SPEED --text \"$TEST_TEXT\""
python -m cli.spark_tts_onnx --model-dir "$MODEL_DIR" --prompt-path "$PROMPT_PATH" --output-dir "$RESULTS_DIR" --gender $GENDER --pitch $PITCH --speed $SPEED --text "$TEST_TEXT"

echo -e "\n===== Demo Complete ====="
echo "Output audio saved to $RESULTS_DIR"
echo "Voice settings: Gender=$GENDER (male), Pitch=$PITCH (low), Speed=$SPEED (moderate)"
echo "Try adjusting these settings in the script to generate different voice characteristics." 
#!/bin/bash

# Default values
MODEL_DIR="pretrained_models/Spark-TTS-0.5B-ONNX"
PROMPT_PATH="example/prompt_audio.wav"
OUTPUT_DIR="output"
TEXT="ONNX version of Spark-TTS is now speaking. This sounds pretty good!"
GENDER=0.5
PITCH=1.0
SPEED=1.0
DEVICE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --prompt-path)
      PROMPT_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --text)
      TEXT="$2"
      shift 2
      ;;
    --text-file)
      TEXT_FILE="$2"
      shift 2
      ;;
    --gender)
      GENDER="$2"
      shift 2
      ;;
    --pitch)
      PITCH="$2"
      shift 2
      ;;
    --speed)
      SPEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Construct command
CMD="python -m cli.spark_tts_onnx --model-dir ${MODEL_DIR} --prompt-path ${PROMPT_PATH} --output-dir ${OUTPUT_DIR} --gender ${GENDER} --pitch ${PITCH} --speed ${SPEED}"

# Add text or text file
if [ -n "$TEXT_FILE" ]; then
  CMD="${CMD} --text-file ${TEXT_FILE}"
else
  CMD="${CMD} --text \"${TEXT}\""
fi

# Add device if specified
if [ -n "$DEVICE" ]; then
  CMD="${CMD} --device ${DEVICE}"
fi

# Print the command
echo "Running: $CMD"

# Execute
eval $CMD 
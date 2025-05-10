#!/bin/bash
# Benchmark script for Spark-TTS

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
MODEL_DIR="pretrained_models/Spark-TTS-0.5B"
PROMPT_SPEECH="example/results/prompt.wav"
BENCHMARK_RUNS=3
SAVE_DIR="benchmark_results"
DEVICE=0
QUANTIZATION="none"
MAX_NEW_TOKENS=3000
TEMPERATURE=0.8
TOP_K=50
TOP_P=0.95
USE_COMPILE=false
COMPILE_MODE="reduce-overhead"
BENCHMARK_MODE="all" # New: 'all', 'pytorch', 'onnx'

# Create results directory if it doesn't exist
mkdir -p $SAVE_DIR

# Display banner
echo -e "${GREEN}"
echo "┌──────────────────────────────────────────────────┐"
echo "│                                                  │"
echo "│               Spark-TTS Benchmark                │"
echo "│                                                  │"
echo "└──────────────────────────────────────────────────┘"
echo -e "${NC}"

# Display help if requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo -e "${BLUE}Usage:${NC}"
  echo "  ./benchmark.sh [options]"
  echo ""
  echo -e "${BLUE}Options:${NC}"
  echo "  -m, --model DIR      Model directory (default: $MODEL_DIR)"
  echo "  -p, --prompt FILE    Prompt speech file (default: $PROMPT_SPEECH)"
  echo "  -r, --runs NUM       Number of benchmark runs (default: $BENCHMARK_RUNS)"
  echo "  -s, --save DIR       Directory to save results (default: $SAVE_DIR)"
  echo "  -d, --device NUM     Device ID (default: $DEVICE)"
  echo "  -q, --quantization   Quantization method: none, fp16, int8 (default: $QUANTIZATION)"
  echo "  -t, --temperature    Sampling temperature (default: $TEMPERATURE)"
  echo "  -k, --top-k          Top-k sampling parameter (default: $TOP_K)"
  echo "  -p, --top-p          Top-p sampling parameter (default: $TOP_P)"
  echo "  -n, --tokens         Max new tokens to generate (default: $MAX_NEW_TOKENS)"
  echo "  -u, --use-compile    Use torch.compile (default: $USE_COMPILE)"
  echo "  -m, --compile-mode   Compilation mode (default: $COMPILE_MODE)"
  echo "  --benchmark_mode     Benchmark mode: all, pytorch, onnx (default: $BENCHMARK_MODE)"
  echo "  -v, --voice          Run voice cloning benchmark (now combined with mode)"
  echo "  -c, --control        Run controlled voice benchmark (now combined with mode)"
  echo "  -h, --help           Show this help message"
  exit 0
fi

# Parse command line arguments
VOICE_CLONE_BENCH=false # Renamed to be more specific
CONTROL_VOICE_BENCH=false # Renamed

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_DIR="$2"
      shift 2
      ;;
    -p|--prompt)
      PROMPT_SPEECH="$2"
      shift 2
      ;;
    -r|--runs)
      BENCHMARK_RUNS="$2"
      shift 2
      ;;
    -s|--save)
      SAVE_DIR="$2"
      shift 2
      ;;
    -d|--device)
      DEVICE="$2"
      shift 2
      ;;
    -q|--quantization)
      QUANTIZATION="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -k|--top-k)
      TOP_K="$2"
      shift 2
      ;;
    -p|--top-p)
      TOP_P="$2"
      shift 2
      ;;
    -n|--tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    -u|--use-compile)
      USE_COMPILE="$2"
      shift 2
      ;;
    -m|--compile-mode)
      COMPILE_MODE="$2"
      shift 2
      ;;
    --benchmark_mode)
      BENCHMARK_MODE="$2"
      shift 2
      ;;
    -v|--voice) # Indicates to run the voice cloning type of benchmark
      VOICE_CLONE_BENCH=true
      shift
      ;;
    -c|--control) # Indicates to run the controlled voice type of benchmark
      CONTROL_VOICE_BENCH=true
      shift
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      exit 1
      ;;
  esac
done

# If no specific benchmark type (voice/control) is chosen, default to voice cloning
if [ "$VOICE_CLONE_BENCH" = false ] && [ "$CONTROL_VOICE_BENCH" = false ]; then
  echo -e "${YELLOW}No specific benchmark type (voice/control) selected, defaulting to voice cloning benchmark.${NC}"
  VOICE_CLONE_BENCH=true
fi

# Function to run a benchmark and parse results
# Accepts: 1. benchmark_name (e.g., "voice_cloning_pytorch"), 2. python_mode_args (onnx flags), 3. prompt_type_args
run_benchmark() {
  local benchmark_name=$1
  local python_mode_args=$2 # e.g., "--use_llm_onnx --use_bicodec_onnx ..." or ""
  local prompt_type_args=$3 # e.g., "--prompt_speech_path=..." or "--gender=..."
  
  local output_file="${SAVE_DIR}/${benchmark_name}_output.txt"
  
  echo -e "${YELLOW}Running ${benchmark_name} benchmark...${NC}"
  echo -e "${BLUE}Configuration:${NC}"
  echo "  Model: $MODEL_DIR"
  echo "  Runs: $BENCHMARK_RUNS"
  echo "  Device: $DEVICE"
  echo "  Quantization: $QUANTIZATION"
  echo "  Generation Parameters:"
  echo "    - Temperature: $TEMPERATURE"
  echo "    - Top-k: $TOP_K"
  echo "    - Top-p: $TOP_P"
  echo "    - Max New Tokens: $MAX_NEW_TOKENS"
  echo "  Acceleration:"
  echo "    - Use torch.compile: $USE_COMPILE"
  echo "    - Compile Mode: $COMPILE_MODE"
  echo "  Mode Args: $python_mode_args"
  echo "  Prompt Type Args: $prompt_type_args"
  echo ""
  
  # Run the benchmark command
  python -m cli.inference \
    --model_dir="$MODEL_DIR" \
    --save_dir="$SAVE_DIR" \
    --device="$DEVICE" \
    --benchmark \
    --benchmark_runs="$BENCHMARK_RUNS" \
    --quantization="$QUANTIZATION" \
    --temperature="$TEMPERATURE" \
    --top_k="$TOP_K" \
    --top_p="$TOP_P" \
    --max_new_tokens="$MAX_NEW_TOKENS" \
    $([ "$USE_COMPILE" = true ] && [[ ! "$python_mode_args" == *"--use_llm_onnx"* ]] && echo "--use_compile") \
    --compile_mode="$COMPILE_MODE" \
    --benchmark_texts \
      "This is the first sentence to synthesize." \
      "Here's the second sentence with the same voice." \
      "A third sentence to demonstrate efficiency." \
      "The fourth sentence shows how reusing tokenization helps." \
      "Finally, a fifth sentence to complete the benchmark." \
    $python_mode_args \
    $prompt_type_args | tee "$output_file"
  
  # Extract and format key metrics
  local speedup=$(grep "Speedup factor:" "$output_file" | awk '{print $3}')
  local std_time=$(grep "Standard flow total time:" "$output_file" | awk '{print $5}')
  local opt_time=$(grep "Optimized flow total time:" "$output_file" | awk '{print $5}')
  local subsequent_speedup=$(grep "Subsequent utterance speedup:" "$output_file" | awk '{print $4}')
  
  # Extract heaviest operations
  local std_heaviest_op=$(grep "Heaviest operation: " "$output_file" | head -1 | sed 's/.*Heaviest operation: \([^ ]*\).*/\1/')
  local std_heaviest_percent=$(grep "Heaviest operation: " "$output_file" | head -1 | sed 's/.*(\([0-9.]*\)%.*/\1/')
  local opt_heaviest_op=$(grep "Heaviest operation: " "$output_file" | tail -1 | sed 's/.*Heaviest operation: \([^ ]*\).*/\1/')
  local opt_heaviest_percent=$(grep "Heaviest operation: " "$output_file" | tail -1 | sed 's/.*(\([0-9.]*\)%.*/\1/')
  
  echo ""
  echo -e "${GREEN}${benchmark_name} Benchmark Results:${NC}"
  echo -e "  Standard Flow Time: ${YELLOW}${std_time}${NC}"
  echo -e "  Optimized Flow Time: ${YELLOW}${opt_time}${NC}"
  echo -e "  Overall Speedup: ${GREEN}${speedup}x${NC}"
  echo -e "  Subsequent Utterance Speedup: ${GREEN}${subsequent_speedup}x${NC}"
  
  # Display heaviest operations
  if [ -n "$std_heaviest_op" ]; then
    echo ""
    echo -e "  ${RED}Heaviest Operations:${NC}"
    echo -e "    Standard Flow: ${YELLOW}${std_heaviest_op}${NC} (${std_heaviest_percent}%)"
    echo -e "    Optimized Flow: ${YELLOW}${opt_heaviest_op}${NC} (${opt_heaviest_percent}%)"
  fi
  
  echo ""
  echo -e "Detailed results saved to: ${BLUE}${output_file}${NC}"
  echo "──────────────────────────────────────────────────"
}

# Time tracking
START_TIME=$(date +%s)

# Determine which benchmarks to run based on BENCHMARK_MODE and selected types

run_pytorch_benchmarks() {
  echo -e "${BLUE}--- Starting PyTorch Benchmarks ---${NC}"
  local pytorch_args="" # No ONNX flags for PyTorch
  if [ "$VOICE_CLONE_BENCH" = true ]; then
    run_benchmark "voice_cloning_pytorch" "$pytorch_args" "--prompt_speech_path=$PROMPT_SPEECH"
  fi
  if [ "$CONTROL_VOICE_BENCH" = true ]; then
    run_benchmark "controlled_voice_pytorch" "$pytorch_args" "--gender=female --pitch=moderate --speed=moderate"
  fi
}

run_onnx_benchmarks() {
  echo -e "${BLUE}--- Starting ONNX Benchmarks ---${NC}"
  local onnx_args="--use_wav2vec2_onnx --use_bicodec_onnx --use_speaker_encoder_tokenizer_onnx --use_llm_onnx"
  if [ "$VOICE_CLONE_BENCH" = true ]; then
    run_benchmark "voice_cloning_onnx" "$onnx_args" "--prompt_speech_path=$PROMPT_SPEECH"
  fi
  if [ "$CONTROL_VOICE_BENCH" = true ]; then
    run_benchmark "controlled_voice_onnx" "$onnx_args" "--gender=female --pitch=moderate --speed=moderate"
  fi
}

if [ "$BENCHMARK_MODE" = "all" ]; then
  run_pytorch_benchmarks
  run_onnx_benchmarks
elif [ "$BENCHMARK_MODE" = "pytorch" ]; then
  run_pytorch_benchmarks
elif [ "$BENCHMARK_MODE" = "onnx" ]; then
  run_onnx_benchmarks
else
  echo -e "${RED}Error: Invalid BENCHMARK_MODE '$BENCHMARK_MODE'. Choose 'all', 'pytorch', or 'onnx'.${NC}"
  exit 1
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo -e "${GREEN}All benchmarks completed in ${MINUTES}m ${SECONDS}s!${NC}"
echo -e "Results saved in ${BLUE}${SAVE_DIR}${NC} directory."
echo ""

# Check if gnuplot is available for visualizing results
if command -v gnuplot &> /dev/null; then
  echo -e "${YELLOW}You can visualize the benchmark results using:${NC}"
  echo "  gnuplot -e \"input='${SAVE_DIR}/benchmark_results_*.json'; output='${SAVE_DIR}/benchmark_chart.png'\" benchmark_visualize.gp"
  echo ""
fi 
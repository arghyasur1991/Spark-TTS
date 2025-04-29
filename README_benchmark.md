# SparkTTS Benchmarking Tools

This directory contains tools for benchmarking the performance of the SparkTTS system, particularly focusing on comparing the standard inference flow versus the optimized flow with tokenization reuse.

## Quick Start

Run a basic benchmark with default settings:

```bash
./benchmark.sh
```

Run all available benchmark types:

```bash
./benchmark.sh --all
```

## Available Benchmarks

### Voice Cloning Benchmark
Measures the performance of voice cloning using a prompt audio file:

```bash
./benchmark.sh --voice --prompt example/prompts/your_voice_sample.wav
```

### Controlled Voice Benchmark
Measures the performance of controllable TTS with specified voice attributes:

```bash
./benchmark.sh --control
```

## Benchmark Options

```
Usage:
  ./benchmark.sh [options]

Options:
  -m, --model DIR      Model directory (default: pretrained_models/Spark-TTS-0.5B)
  -p, --prompt FILE    Prompt speech file (default: example/prompts/voice_sample.wav)
  -r, --runs NUM       Number of benchmark runs (default: 3)
  -s, --save DIR       Directory to save results (default: benchmark_results)
  -d, --device NUM     Device ID (default: 0)
  -v, --voice          Run voice cloning benchmark
  -c, --control        Run controlled voice benchmark
  -a, --all            Run all benchmarks
  -h, --help           Show this help message
```

## Understanding Results

The benchmark measures and compares:

1. **Standard Flow**: Re-tokenizes and processes each input from scratch
2. **Optimized Flow**: Tokenizes once, then reuses/updates for subsequent texts

For each benchmark, you'll see:
- Time taken for both flows
- Overall speedup factor
- Tokenization overhead
- Subsequent utterance speedup

### Operation Timing Analysis

The benchmark now includes detailed timing analysis for each operation in the inference pipeline:

- **tokenization**: Time spent tokenizing text inputs
- **model_generation**: Time spent in the model's forward pass generating tokens
- **token_extraction**: Time spent parsing and extracting tokens from generated output
- **audio_conversion**: Time spent converting tokens to audio waveform

For each operation, you'll see:
- Absolute time (in seconds)
- Percentage of total inference time
- Identification of the heaviest (most time-consuming) operation

This helps identify bottlenecks and optimize the pipeline accordingly.

## Results Storage

Results are saved to:
- Text file: `benchmark_results/[benchmark_type]_benchmark_output.txt`
- JSON data: `benchmark_results/benchmark_results_[timestamp].json`

## Visualizing Results

If you have `gnuplot` and `jq` installed, you can visualize the benchmark results:

```bash
gnuplot -e "input='benchmark_results/benchmark_results_*.json';output='benchmark_results/benchmark_chart.png'" benchmark_visualize.gp
```

This creates a bar chart comparing the performance metrics of both flows.

## What to Expect

Typical results show that the optimized flow outperforms the standard flow for multi-utterance synthesis, especially when generating many sentences with the same voice. The first utterance includes tokenization overhead, but subsequent utterances are significantly faster.

For voice cloning, you might see:
- Overall speedup: 1.5-2x
- Subsequent utterance speedup: 2-3x
- Heaviest operation: Generally `model_generation` (60-80% of time)

For controlled voice synthesis, you might see:
- Overall speedup: 1.3-1.8x
- Subsequent utterance speedup: 1.5-2.5x
- Heaviest operation: Typically also `model_generation`

## Implementation Details

The optimized flow improves performance by:
1. Tokenizing the input only once per voice
2. Reusing global voice tokens across utterances
3. Efficiently updating only the text portion of tokenized inputs

This approach is particularly beneficial for applications like virtual assistants, audiobooks, or any scenario where multiple utterances use the same voice.

## Performance Bottlenecks

Based on the operation timing analysis, you may observe:

1. **Model Generation**: Usually the heaviest operation (typically 60-80% of total time)
   - Optimization strategies: Model quantization, distillation, or hardware acceleration

2. **Audio Conversion**: Often the second heaviest (typically 10-30% of total time)
   - Optimization strategies: Optimize the audio tokenizer or use faster codecs

3. **Tokenization**: Minimal impact for standard flow, but significant for first utterance of optimized flow
   - This is why the optimized flow shows better performance for multiple utterances 
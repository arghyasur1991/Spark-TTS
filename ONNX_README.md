# Spark-TTS ONNX Conversion

This repository provides tools to convert Spark-TTS PyTorch models to ONNX format for faster inference and cross-platform compatibility.

## Prerequisites

### System Requirements

Before installing the Python dependencies, you need the following system prerequisites:

- **cmake**: Required for building some packages
  - **macOS**: `brew install cmake`
  - **Ubuntu/Debian**: `sudo apt-get install cmake`
  - **Windows**: Download from [cmake.org](https://cmake.org/download/) or `choco install cmake`

- **Python 3.8+**: This script is tested with Python 3.8+

### Python Dependencies

After installing the system prerequisites, install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you encounter build errors, consider using our helper script:

```bash
# Make the script executable
chmod +x install_onnx_deps.sh

# Run the script
./install_onnx_deps.sh
```

Or install the key packages manually:

```bash
# Install ONNX avoiding build issues
pip install onnx>=1.15.0 --no-build-isolation

# Install ONNX Runtime (works for CPU, Apple Silicon, and with NVIDIA GPUs)
pip install onnxruntime>=1.17.0

# For NVIDIA GPUs specifically, you can use
pip install onnxruntime-gpu>=1.17.0
```

## Converting Models to ONNX

To convert all the Spark-TTS models to ONNX format, run:

```bash
python convert_to_onnx.py
```

This will:
1. Load the Spark-TTS PyTorch models
2. Convert each component to ONNX format
3. Save the converted models to `pretrained_models/Spark-TTS-0.5B/onnx/`
4. Verify the converted models using ONNX Runtime

## Command Line Arguments

The conversion script supports several command-line arguments:

- `--model_dir`: Directory containing PyTorch model files (default: `pretrained_models/Spark-TTS-0.5B`)
- `--output_dir`: Directory to save ONNX models (default: `pretrained_models/Spark-TTS-0.5B/onnx`)
- `--dynamic_axes`: Use dynamic axes for sequence dimensions (recommended for variable input sizes)
- `--cpu_only`: Force CPU usage even if CUDA is available
- `--opset_version`: ONNX opset version to use (default: 17)
- `--skip_verification`: Skip ONNX Runtime verification (useful if you have installation issues)

Example:

```bash
python convert_to_onnx.py --model_dir pretrained_models/Spark-TTS-0.5B --output_dir onnx_models --dynamic_axes
```

## Output Models

The script generates the following ONNX models:

- `encoder.onnx`: BiCodec Encoder
- `quantizer_tokenize.onnx`: BiCodec Quantizer (tokenize function)
- `quantizer_detokenize.onnx`: BiCodec Quantizer (detokenize function)
- `speaker_encoder.onnx`: Speaker Encoder
- `prenet.onnx`: BiCodec Prenet
- `postnet.onnx`: BiCodec Postnet
- `decoder.onnx`: BiCodec Wave Generator (Decoder)

## Using the ONNX Models

The ONNX models can be used with any ONNX-compatible runtime, including:

- ONNX Runtime (Python, C++, C#, Java)
- TensorRT
- OpenVINO
- CoreML
- TFLite (via ONNX converter)

Example usage with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("pretrained_models/Spark-TTS-0.5B/onnx/encoder.onnx")

# Prepare input
input_data = np.random.randn(1, 100, 80).astype(np.float32)  # [batch, seq_len, feature_dim]

# Run inference
outputs = session.run(None, {"input": input_data})
```

## Troubleshooting

### Error: Could not find cmake in PATH

If you see an error like:
```
AssertionError: Could not find cmake in PATH
```

Install cmake using the commands in the Prerequisites section, then try again.

### Error with CMake compatibility

If you see an error like:
```
CMake Error at CMakeLists.txt:2 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

This is likely happening because you're using Python 3.12+ which has compatibility issues with building ONNX from source. Use our helper script to install pre-built packages:

```bash
# Make the script executable
chmod +x install_onnx_deps.sh

# Run it
./install_onnx_deps.sh
```

Or install manually with the `--no-build-isolation` flag:

```bash
pip install onnx>=1.15.0 --no-build-isolation
pip install onnxruntime>=1.17.0  # Works for Apple Silicon and other platforms
```

### Error installing onnxruntime

If you encounter issues installing onnxruntime packages:

1. Try installing the latest version available:
   ```bash
   # Check available versions
   pip index versions onnxruntime
   
   # Install latest version (example)
   pip install onnxruntime>=1.17.0
   
   # For NVIDIA GPUs specifically
   pip install onnxruntime-gpu>=1.17.0
   ```

2. You can also use the CPU-only version of the script by adding the `--skip_verification` flag:
   ```bash
   python convert_to_onnx.py --cpu_only --skip_verification
   ```

### Python 3.12+ Compatibility Issues

Python 3.12 is relatively new and some packages may not have binary wheels available yet. If you encounter persistent issues:

1. Consider creating a new conda environment with Python 3.10:
   ```bash
   conda create -n sparktts-py310 python=3.10
   conda activate sparktts-py310
   pip install -r requirements.txt
   ```

2. Or use the `--skip_verification` flag to bypass the ONNX Runtime verification step:
   ```bash
   python convert_to_onnx.py --skip_verification
   ```

## Performance Benefits

Converting to ONNX format provides several benefits:

1. **Faster inference**: ONNX Runtime can optimize the model for your specific hardware
2. **Reduced memory usage**: Models are optimized for inference only
3. **Cross-platform support**: Run the same models on various devices and platforms
4. **Integration with other tools**: Easy integration with deployment tools and frameworks 